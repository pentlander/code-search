//! code-search: Standalone CLI for semantic code search and tree-sitter navigation.

use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Parser as ClapParser, Subcommand};

use code_search::db::SearchDB;
use code_search::indexer;
use code_search::model::Embedder;
use code_search::nav::{self, Categories, InspectOptions, InspectTarget, Role};
use code_search::parser::ParserCache;
use code_search::search::multi_query_search;
use code_search::symbols::{extract_symbols, format_outline, format_symbols, ExtractOptions};
use code_search::walker::collect_files;

const CACHE_DIR: &str = ".code-search-cache";
const DB_FILE: &str = "index.db";

const DEFAULT_MAX_LINES: usize = 2000;
const DEFAULT_MAX_BYTES: usize = 50 * 1024;

// ─── CLI structure ─────────────────────────────────────────────────────

#[derive(ClapParser)]
#[command(name = "code-search", about = "Semantic code search and tree-sitter navigation")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Incremental index a directory for semantic search
    Index {
        /// Directory to index (default: current directory)
        #[arg(long, default_value = ".")]
        path: String,

        /// Path to the SQLite database
        #[arg(long)]
        db: Option<String>,

        /// Force full rebuild (ignore incremental cache)
        #[arg(long)]
        force: bool,
    },

    /// Semantic search using natural language queries
    Search {
        /// Search queries (multiple queries are merged and deduplicated)
        queries: Vec<String>,

        /// Directory to search (default: current directory)
        #[arg(long, default_value = ".")]
        path: String,

        /// Number of results to return
        #[arg(long, default_value = "25")]
        top_k: i32,

        /// Minimum similarity score 0-1
        #[arg(long, default_value = "0.0")]
        threshold: f64,

        /// Filter by language
        #[arg(long)]
        language: Option<String>,

        /// Filter by symbol kind
        #[arg(long)]
        kind: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Analyze code structure using tree-sitter (like code_nav)
    Symbols {
        /// File or directory path to analyze
        path: String,

        /// "outline" (tree), "symbols" (flat list), "inspect" (inside one
        /// symbol) or "scope" (enclosing symbols at a line)
        #[arg(long, default_value = "outline",
              value_parser = ["outline", "symbols", "inspect", "scope"])]
        action: String,

        /// 1-based line: required for scope; selects the symbol for inspect
        #[arg(long)]
        line: Option<usize>,

        /// 1-based column on --line, for an exact position
        #[arg(long, requires = "line")]
        column: Option<usize>,

        /// Symbol to inspect, optionally qualified ("Type::method" or "Type.method")
        #[arg(long, conflicts_with = "line")]
        symbol: Option<String>,

        /// Inspect: maximum nesting depth of reported items (default 3).
        /// Outline/symbols: maximum symbol nesting depth, 0 = top level only
        #[arg(long)]
        depth: Option<usize>,

        /// Inspect: comma-separated categories (control_flow, calls, locals, nested_symbols)
        #[arg(long)]
        include: Option<String>,

        /// Filter by symbol kind
        #[arg(long)]
        kind: Option<String>,

        /// Regex pattern to filter symbols by name
        #[arg(long)]
        name_pattern: Option<String>,

        /// Include function/method signatures
        #[arg(long)]
        signatures: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Find all occurrences of an identifier in code
    FindIdents {
        /// Identifier name to search for (exact match)
        name: String,

        /// File or directory to search in
        path: String,

        /// Lines of context around each match
        #[arg(long, default_value = "0")]
        context: usize,

        /// Only report this syntactic role
        #[arg(long, value_parser = Role::ALL)]
        role: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show index statistics
    Stats {
        /// Directory to check (default: current directory)
        #[arg(long, default_value = ".")]
        path: String,
    },
}

// ─── Helpers ───────────────────────────────────────────────────────────

fn find_repo_root(dir: &Path) -> PathBuf {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(dir)
        .output();
    match output {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            PathBuf::from(s)
        }
        _ => dir.to_path_buf(),
    }
}

fn resolve_db_path(db_arg: Option<&str>, repo_root: &Path) -> PathBuf {
    match db_arg {
        Some(p) => PathBuf::from(p),
        None => repo_root.join(CACHE_DIR).join(DB_FILE),
    }
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("semantic-search")
        .join("models")
        .join("CodeRankEmbed-safetensors")
}

fn tokenizer_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("semantic-search")
        .join("models")
        .join("CodeRankEmbed-onnx-q8")
        .join("tokenizer.json")
}

fn truncate_output(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let total_lines = lines.len();
    let total_bytes = text.len();

    if total_lines <= DEFAULT_MAX_LINES && total_bytes <= DEFAULT_MAX_BYTES {
        return text.to_string();
    }

    let mut output = String::new();
    let mut byte_count = 0;
    let mut line_count = 0;

    for line in &lines {
        if line_count >= DEFAULT_MAX_LINES || byte_count + line.len() > DEFAULT_MAX_BYTES {
            break;
        }
        if line_count > 0 {
            output.push('\n');
            byte_count += 1;
        }
        output.push_str(line);
        byte_count += line.len();
        line_count += 1;
    }

    output.push_str(&format!(
        "\n\n[Output truncated: showing {} of {} lines ({} of {} bytes)]",
        line_count,
        total_lines,
        byte_count,
        total_bytes
    ));
    output
}

// ─── Command implementations ───────────────────────────────────────────

fn cmd_index(path: &str, db_arg: Option<&str>, force: bool) -> Result<(), String> {
    let target_dir = std::fs::canonicalize(path)
        .map_err(|e| format!("Cannot resolve path {}: {}", path, e))?;
    let repo_root = find_repo_root(&target_dir);
    let db_path = resolve_db_path(db_arg, &repo_root);

    let mut db = SearchDB::open(&db_path).map_err(|e| format!("DB error: {}", e))?;
    let mut embedder = Embedder::load(&model_dir(), &tokenizer_path())?;
    let mut parser_cache = ParserCache::new();

    let stats = indexer::index_directory(
        &target_dir,
        &repo_root,
        &mut db,
        &mut embedder,
        &mut parser_cache,
        force,
        &|msg| eprintln!("{}", msg),
    )?;

    eprintln!(
        "Index complete: {} symbols from {} files ({}ms, {} embedded, {} reused)",
        stats.symbols_indexed,
        stats.files_indexed,
        stats.index_time_ms,
        stats.symbols_embedded,
        stats.symbols_reused,
    );
    if stats.files_deleted > 0 {
        eprintln!("  {} files removed from index", stats.files_deleted);
    }

    Ok(())
}

fn cmd_search(
    queries: &[String],
    path: &str,
    top_k: i32,
    threshold: f64,
    language: Option<&str>,
    kind: Option<&str>,
    json: bool,
) -> Result<(), String> {
    let target_dir = std::fs::canonicalize(path)
        .map_err(|e| format!("Cannot resolve path {}: {}", path, e))?;
    let repo_root = find_repo_root(&target_dir);
    let db_path = resolve_db_path(None, &repo_root);

    let mut db = SearchDB::open(&db_path).map_err(|e| format!("DB error: {}", e))?;
    let mut embedder = Embedder::load(&model_dir(), &tokenizer_path())?;
    let mut parser_cache = ParserCache::new();

    // Incremental index update
    let _index_stats = indexer::index_directory(
        &target_dir,
        &repo_root,
        &mut db,
        &mut embedder,
        &mut parser_cache,
        false,
        &|msg| eprintln!("{}", msg),
    )?;

    // Path prefix for scoping
    let path_prefix = if target_dir != repo_root {
        target_dir
            .strip_prefix(&repo_root)
            .ok()
            .map(|p| p.to_string_lossy().to_string())
    } else {
        None
    };

    let results = multi_query_search(
        queries,
        top_k,
        threshold,
        language,
        kind,
        path_prefix.as_deref(),
        &db,
        &mut embedder,
    )?;

    if json {
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "file_path": r.file_path,
                    "name": r.name,
                    "kind": r.kind,
                    "language": r.language,
                    "line": r.line,
                    "end_line": r.end_line,
                    "signature": r.signature,
                    "score": r.score,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json_results).unwrap());
    } else {
        if results.is_empty() {
            let stats = db.get_stats().map_err(|e| format!("DB error: {}", e))?;
            println!(
                "No results (searched {} symbols across {} files)",
                stats.symbol_count, stats.file_count
            );
            return Ok(());
        }

        println!(
            "{:<7} {:<50} Symbol",
            "Score", "File"
        );
        for r in &results {
            let line_range = match r.end_line {
                Some(el) => format!("{}-{}", r.line, el),
                None => format!("{}", r.line),
            };
            let file_str = format!("{}:{}", r.file_path, line_range);
            let sig = r.signature.as_deref().unwrap_or(&r.name);
            println!("{:<7.2} {:<50} {}", r.score, file_str, sig);
        }
    }

    Ok(())
}

/// Flags that only `inspect` and `scope` take.
struct NavArgs<'a> {
    line: Option<usize>,
    column: Option<usize>,
    symbol: Option<&'a str>,
    depth: Option<usize>,
    include: Option<&'a str>,
}

fn cmd_symbols(
    path: &str,
    action: &str,
    kind: Option<&str>,
    name_pattern: Option<&str>,
    signatures: bool,
    nav_args: NavArgs,
    _json: bool,
) -> Result<(), String> {
    let abs_path =
        std::fs::canonicalize(path).map_err(|e| format!("Path not found: {}: {}", path, e))?;

    if action == "inspect" || action == "scope" {
        if kind.is_some() || name_pattern.is_some() {
            return Err("--kind and --name-pattern apply only to outline and symbols".to_string());
        }
        return cmd_navigate(path, &abs_path, action, signatures, &nav_args);
    }
    if nav_args.line.is_some()
        || nav_args.column.is_some()
        || nav_args.symbol.is_some()
        || nav_args.include.is_some()
    {
        return Err(
            "--line, --column, --symbol and --include apply only to inspect and scope".to_string(),
        );
    }

    let mut parser_cache = ParserCache::new();

    let metadata = std::fs::metadata(&abs_path)
        .map_err(|e| format!("Cannot stat {}: {}", path, e))?;

    if metadata.is_dir() {
        let has_filter = kind.is_some() || name_pattern.is_some();
        let max_files = if has_filter { 10000 } else { 200 };
        let result = collect_files(&abs_path, max_files);
        let cwd = std::env::current_dir().unwrap_or_else(|_| abs_path.clone());

        let mut sections = Vec::new();

        for file in &result.found {
            match parser_cache.parse_file(file) {
                Ok(parse_result) => {
                    let opts = ExtractOptions {
                        signatures,
                        kind: kind.map(|s| s.to_string()),
                        name_pattern: name_pattern.map(|s| s.to_string()),
                        top_level_only: true,
                        max_depth: Some(nav_args.depth.unwrap_or(2)),
                    };

                    let syms = extract_symbols(
                        &parse_result.tree,
                        parse_result.language.grammar,
                        &parse_result.source,
                        &opts,
                    );
                    if syms.is_empty() {
                        continue;
                    }
                    let rel_path = file
                        .strip_prefix(&cwd)
                        .unwrap_or(file)
                        .to_string_lossy();

                    let formatted = if action == "outline" {
                        format_outline(&syms, signatures)
                    } else {
                        format_symbols(&syms, signatures)
                    };

                    sections.push(format!(
                        "── {} ({}) ──\n{}",
                        rel_path, parse_result.language.name, formatted
                    ));
                }
                Err(_) => continue,
            }
        }

        let output = sections.join("\n\n");
        let truncated = truncate_output(&output);
        println!("{}", truncated);

        if result.skipped > 0 {
            eprintln!(
                "[Scanned {} of {} files. {} skipped (limit: {})]",
                result.found.len(),
                result.found.len() + result.skipped,
                result.skipped,
                max_files
            );
        }
    } else {
        let parse_result = parser_cache.parse_file(&abs_path)?;

        let opts = ExtractOptions {
            signatures,
            kind: kind.map(|s| s.to_string()),
            name_pattern: name_pattern.map(|s| s.to_string()),
            max_depth: nav_args.depth,
            ..Default::default()
        };

        let syms = extract_symbols(
            &parse_result.tree,
            parse_result.language.grammar,
            &parse_result.source,
            &opts,
        );

        let output = if action == "outline" {
            format_outline(&syms, signatures)
        } else {
            format_symbols(&syms, signatures)
        };

        if output.trim().is_empty() {
            let filters = [
                kind.map(|k| format!("kind=\"{}\"", k)),
                name_pattern.map(|p| format!("namePattern=\"{}\"", p)),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(", ");
            if !filters.is_empty() {
                println!("No symbols matching {} found in {}", filters, path);
            } else {
                println!("No symbols found in {}", path);
            }
        } else {
            let truncated = truncate_output(&output);
            println!("{}", truncated);
        }
    }

    Ok(())
}

fn cmd_navigate(
    path: &str,
    abs_path: &Path,
    action: &str,
    signatures: bool,
    args: &NavArgs,
) -> Result<(), String> {
    if abs_path.is_dir() {
        return Err(format!(
            "{} needs a file, not a directory: {}",
            action, path
        ));
    }
    let mut parser_cache = ParserCache::new();
    let parsed = parser_cache.parse_file(abs_path)?;
    let grammar = parsed.language.grammar;

    let output = if action == "scope" {
        if args.symbol.is_some() || args.depth.is_some() || args.include.is_some() {
            return Err("scope takes only --line and --column".to_string());
        }
        let line = args.line.ok_or("scope needs --line")?;
        nav::scope(
            &parsed.tree,
            grammar,
            &parsed.source,
            line,
            args.column,
            signatures,
        )?
    } else {
        let target = match (args.line, args.symbol) {
            (Some(line), None) => InspectTarget::Line {
                line,
                column: args.column,
            },
            (None, Some(symbol)) => InspectTarget::Symbol(symbol),
            _ => return Err("inspect needs exactly one of --line or --symbol".to_string()),
        };
        let include = match args.include {
            Some(list) => Categories::parse(list)?,
            None => Categories::all(),
        };
        nav::inspect(
            &parsed.tree,
            grammar,
            &parsed.source,
            path,
            target,
            &InspectOptions {
                depth: args.depth.unwrap_or(3),
                include,
                signatures,
            },
        )?
    };
    println!("{}", truncate_output(&output));
    Ok(())
}

/// A name search filters its own output and skips files without the name
/// before parsing, so it can scan far more than an unfiltered outline.
const FIND_IDENTS_MAX_FILES: usize = 100_000;

fn cmd_find_idents(
    name: &str,
    path: &str,
    context: usize,
    role: Option<&str>,
    _json: bool,
) -> Result<(), String> {
    let role = role.and_then(Role::parse);
    let abs_path =
        std::fs::canonicalize(path).map_err(|e| format!("Path not found: {}: {}", path, e))?;

    let mut parser_cache = ParserCache::new();
    let cwd = std::env::current_dir().unwrap_or_else(|_| abs_path.clone());
    let rel = |file: &Path| {
        file.strip_prefix(&cwd)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string()
    };

    let (files, skipped) = if abs_path.is_dir() {
        let result = collect_files(&abs_path, FIND_IDENTS_MAX_FILES);
        (result.found, result.skipped)
    } else {
        (vec![abs_path.clone()], 0)
    };
    let single_file = !abs_path.is_dir();

    let mut lines = Vec::new();
    let mut total = 0;
    let mut file_count = 0;
    for file in &files {
        // Most files cannot contain the name; skip them before parsing.
        if !single_file
            && !std::fs::read(file)
                .is_ok_and(|bytes| String::from_utf8_lossy(&bytes).contains(name))
        {
            continue;
        }
        let parsed = match parser_cache.parse_file(file) {
            Ok(p) => p,
            // An explicitly named file must parse; directory scans skip what they cannot.
            Err(e) if single_file => return Err(e),
            Err(_) => continue,
        };
        if parsed.language.grammar == "markdown" {
            continue;
        }
        let hits: Vec<nav::IdentMatch> =
            nav::find_identifiers(&parsed.tree, parsed.language.grammar, &parsed.source, name)
                .into_iter()
                .filter(|m| role.is_none_or(|r| m.role == r))
                .collect();
        if hits.is_empty() {
            continue;
        }
        let source_lines: Vec<&str> = parsed.source.lines().collect();
        if !lines.is_empty() {
            lines.push(String::new());
        }
        lines.push(rel(file));
        for (i, m) in hits.iter().enumerate() {
            let tag = match &m.scope {
                Some(scope) => format!("[{} in {}]", m.role.as_str(), scope),
                None => format!("[{}]", m.role.as_str()),
            };
            if context == 0 {
                let text = source_lines.get(m.line - 1).map(|l| l.trim()).unwrap_or("");
                lines.push(format!("  {}:{}  {}  {}", m.line, m.col, tag, text));
                continue;
            }
            if i > 0 {
                lines.push("  --".to_string());
            }
            lines.push(format!("  {}:{}  {}", m.line, m.col, tag));
            let first = m.line.saturating_sub(context).max(1);
            let last = (m.line + context).min(source_lines.len());
            for n in first..=last {
                let marker = if n == m.line { ">" } else { " " };
                let text = source_lines.get(n - 1).map(|l| l.trim_end()).unwrap_or("");
                lines.push(format!("{} {}:  {}", marker, n, text));
            }
        }
        total += hits.len();
        file_count += 1;
    }

    let role_note = role
        .map(|r| format!(" with role {}", r.as_str()))
        .unwrap_or_default();
    if total == 0 {
        println!(
            "No occurrences of \"{}\"{} found in {}",
            name, role_note, path
        );
    } else {
        println!("{}", truncate_output(&lines.join("\n")));
        eprintln!(
            "{} {} across {} {}{}",
            total,
            if total == 1 {
                "occurrence"
            } else {
                "occurrences"
            },
            file_count,
            if file_count == 1 { "file" } else { "files" },
            role.map(|r| format!(" (role: {})", r.as_str()))
                .unwrap_or_default()
        );
    }
    if skipped > 0 {
        eprintln!(
            "[Scanned {} of {} files. {} skipped (limit: {})]",
            files.len(),
            files.len() + skipped,
            skipped,
            FIND_IDENTS_MAX_FILES
        );
    }
    Ok(())
}

fn cmd_stats(path: &str) -> Result<(), String> {
    let target_dir = std::fs::canonicalize(path)
        .map_err(|e| format!("Cannot resolve path {}: {}", path, e))?;
    let repo_root = find_repo_root(&target_dir);
    let db_path = resolve_db_path(None, &repo_root);

    if !db_path.exists() {
        println!("No index found at {}", db_path.display());
        println!("Run `code-search index` to create one.");
        return Ok(());
    }

    let db = SearchDB::open(&db_path).map_err(|e| format!("DB error: {}", e))?;
    let stats = db.get_stats().map_err(|e| format!("DB error: {}", e))?;

    println!("Index: {}", db_path.display());
    println!("Files: {}", stats.file_count);
    println!("Symbols: {}", stats.symbol_count);

    Ok(())
}

// ─── Main ──────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Index { path, db, force } => cmd_index(&path, db.as_deref(), force),
        Commands::Search {
            queries,
            path,
            top_k,
            threshold,
            language,
            kind,
            json,
        } => cmd_search(
            &queries,
            &path,
            top_k,
            threshold,
            language.as_deref(),
            kind.as_deref(),
            json,
        ),
        Commands::Symbols {
            path,
            action,
            line,
            column,
            symbol,
            depth,
            include,
            kind,
            name_pattern,
            signatures,
            json,
        } => cmd_symbols(
            &path,
            &action,
            kind.as_deref(),
            name_pattern.as_deref(),
            signatures,
            NavArgs {
                line,
                column,
                symbol: symbol.as_deref(),
                depth,
                include: include.as_deref(),
            },
            json,
        ),
        Commands::FindIdents {
            name,
            path,
            context,
            role,
            json,
        } => cmd_find_idents(&name, &path, context, role.as_deref(), json),
        Commands::Stats { path } => cmd_stats(&path),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

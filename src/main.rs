//! code-search: Standalone CLI for semantic code search and tree-sitter navigation.

use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Parser as ClapParser, Subcommand};

use code_search::db::SearchDB;
use code_search::indexer;
use code_search::model::Embedder;
use code_search::parser::ParserCache;
use code_search::search::multi_query_search;
use code_search::symbols::{
    self, extract_symbols, format_outline, format_symbols, ExtractOptions,
};
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

        /// Output format: "outline" or "symbols"
        #[arg(long, default_value = "outline")]
        action: String,

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

fn cmd_symbols(
    path: &str,
    action: &str,
    kind: Option<&str>,
    name_pattern: Option<&str>,
    signatures: bool,
    _json: bool,
) -> Result<(), String> {
    let abs_path = std::fs::canonicalize(path)
        .map_err(|e| format!("Path not found: {}: {}", path, e))?;

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
                        max_depth: Some(2),
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

fn cmd_find_idents(name: &str, path: &str, context: usize, _json: bool) -> Result<(), String> {
    let abs_path = std::fs::canonicalize(path)
        .map_err(|e| format!("Path not found: {}: {}", path, e))?;

    let mut parser_cache = ParserCache::new();
    let cwd = std::env::current_dir().unwrap_or_else(|_| abs_path.clone());
    let mut all_matches = Vec::new();

    let metadata = std::fs::metadata(&abs_path)
        .map_err(|e| format!("Cannot stat {}: {}", path, e))?;

    if metadata.is_dir() {
        let result = collect_files(&abs_path, 200);
        for file in &result.found {
            match parser_cache.parse_file(file) {
                Ok(parse_result) => {
                    let rel_path = file
                        .strip_prefix(&cwd)
                        .unwrap_or(file)
                        .to_string_lossy()
                        .to_string();
                    let matches = symbols::find_identifiers_in_tree(
                        &parse_result.tree,
                        name,
                        &parse_result.source,
                        &rel_path,
                    );
                    all_matches.extend(matches);
                }
                Err(_) => continue,
            }
        }
    } else {
        let parse_result = parser_cache.parse_file(&abs_path)?;
        let rel_path = abs_path
            .strip_prefix(&cwd)
            .unwrap_or(&abs_path)
            .to_string_lossy()
            .to_string();
        let matches = symbols::find_identifiers_in_tree(
            &parse_result.tree,
            name,
            &parse_result.source,
            &rel_path,
        );
        all_matches.extend(matches);
    }

    if all_matches.is_empty() {
        println!("No occurrences of \"{}\" found in {}", name, path);
        return Ok(());
    }

    let mut lines = Vec::new();
    let mut current_file = String::new();
    for m in &all_matches {
        if m.file != current_file {
            if !current_file.is_empty() {
                lines.push(String::new());
            }
            lines.push(m.file.clone());
            current_file = m.file.clone();
        }

        if context > 0 {
            let start_line = m.line.saturating_sub(context + 1);
            let end_line = (m.line + context).min(m.source_lines.len());
            if lines.last().map_or(false, |l| l != &m.file) {
                lines.push("  --".to_string());
            }
            for i in start_line..end_line {
                let line_num = i + 1;
                let marker = if line_num == m.line { ">" } else { " " };
                let line_text = m.source_lines.get(i).map(|l| l.trim_end()).unwrap_or("");
                lines.push(format!("{} {}:  {}", marker, line_num, line_text));
            }
        } else {
            lines.push(format!("  {}:{}  {}", m.line, m.col, m.line_text));
        }
    }

    let output = lines.join("\n");
    let truncated = truncate_output(&output);
    println!("{}", truncated);

    let file_count: std::collections::HashSet<&str> =
        all_matches.iter().map(|m| m.file.as_str()).collect();
    eprintln!(
        "{} occurrences across {} files",
        all_matches.len(),
        file_count.len()
    );

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
            json,
        ),
        Commands::FindIdents {
            name,
            path,
            context,
            json,
        } => cmd_find_idents(&name, &path, context, json),
        Commands::Stats { path } => cmd_stats(&path),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

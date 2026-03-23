//! Tree-sitter symbol extraction → embedding text.
//!
//! Extracts symbols via tree-sitter and formats each as:
//! "<lang> | <relative-path> | <signature>"

use std::collections::HashSet;
use std::path::Path;

use crate::parser::ParserCache;
use crate::symbols::{extract_symbols, ExtractOptions, SymbolInfo, SymbolKind};

/// Config file extensions — only index blocks/sections, not leaf properties.
static CONFIG_EXTENSIONS: &[&str] = &[".toml", ".yaml", ".yml", ".hcl", ".tf", ".tfvars"];

/// Symbol kinds we want to index for semantic search.
static INDEXABLE_KINDS: &[SymbolKind] = &[
    SymbolKind::Function,
    SymbolKind::Method,
    SymbolKind::Type,
    SymbolKind::Struct,
    SymbolKind::Interface,
    SymbolKind::Class,
    SymbolKind::Enum,
    SymbolKind::Constant,
    SymbolKind::Trait,
    SymbolKind::Impl,
    SymbolKind::Module,
    SymbolKind::Property,
    SymbolKind::Block,
    SymbolKind::Resource,
    SymbolKind::Data,
];

#[derive(Debug, Clone)]
pub struct ChunkInfo {
    /// The text to embed: "go | path/to/file.go | func signature(...)"
    pub embedding_text: String,
    /// Symbol name
    pub name: String,
    /// Symbol kind
    pub kind: String,
    /// Language name (lowercase)
    pub language: String,
    /// Start line (1-based)
    pub line: i32,
    /// End line (1-based)
    pub end_line: Option<i32>,
    /// Signature string if available
    pub signature: Option<String>,
    /// Relative file path
    pub file_path: String,
}

/// Extract indexable symbols from a file and format them as embedding text.
pub fn extract_chunks(
    file_path: &Path,
    repo_root: &Path,
    parser_cache: &mut ParserCache,
) -> Result<Vec<ChunkInfo>, String> {
    let result = parser_cache.parse_file(file_path)?;
    let rel_path = file_path
        .strip_prefix(repo_root)
        .unwrap_or(file_path)
        .to_string_lossy()
        .to_string();

    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e))
        .unwrap_or_default();
    let is_config = CONFIG_EXTENSIONS.contains(&ext.as_str());

    let opts = ExtractOptions {
        signatures: true,
        top_level_only: is_config,
        ..Default::default()
    };

    let symbols = extract_symbols(&result.tree, result.language.grammar, &result.source, &opts);
    let mut chunks = Vec::new();
    let lang_name = result.language.name.to_lowercase();
    let indexable: HashSet<SymbolKind> = INDEXABLE_KINDS.iter().copied().collect();

    flatten_symbols(&symbols, &rel_path, &lang_name, &mut chunks, is_config, &indexable);

    Ok(chunks)
}

fn flatten_symbols(
    symbols: &[SymbolInfo],
    rel_path: &str,
    lang_name: &str,
    out: &mut Vec<ChunkInfo>,
    is_config: bool,
    indexable: &HashSet<SymbolKind>,
) {
    for sym in symbols {
        // For config files, skip leaf properties
        if is_config && sym.kind == SymbolKind::Property {
            continue;
        }

        if indexable.contains(&sym.kind) {
            let display_sig = sym.signature.as_deref().unwrap_or(&sym.name);
            let embedding_text = format!("{} | {} | {}", lang_name, rel_path, display_sig);

            out.push(ChunkInfo {
                embedding_text,
                name: sym.name.clone(),
                kind: sym.kind.as_str().to_string(),
                language: lang_name.to_string(),
                line: sym.start_line as i32,
                end_line: Some(sym.end_line as i32),
                signature: sym.signature.clone(),
                file_path: rel_path.to_string(),
            });
        }

        // Recurse into children (not for config files)
        if !sym.children.is_empty() && !is_config {
            flatten_symbols(&sym.children, rel_path, lang_name, out, is_config, indexable);
        }
    }
}

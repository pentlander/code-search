//! Tree-sitter parser initialization and parse caching.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use tree_sitter::{Parser, Tree};

use crate::languages::{get_language_for_extension, load_language, LanguageInfo};

struct CacheEntry {
    tree: Tree,
    mtime: SystemTime,
    grammar: String,
}

pub struct ParserCache {
    parser: Parser,
    cache: HashMap<PathBuf, CacheEntry>,
}

pub struct ParseResult {
    pub tree: Tree,
    pub language: LanguageInfo,
    pub source: String,
}

impl ParserCache {
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
            cache: HashMap::new(),
        }
    }

    /// Parse a file, returning the tree, language info, and source text.
    /// Uses mtime-based caching to avoid re-parsing unchanged files.
    pub fn parse_file(&mut self, file_path: &Path) -> Result<ParseResult, String> {
        let abs_path = fs::canonicalize(file_path)
            .map_err(|e| format!("Cannot resolve path {}: {}", file_path.display(), e))?;

        let ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{}", e))
            .unwrap_or_default();

        let lang_info = get_language_for_extension(&ext)
            .ok_or_else(|| format!("Unsupported language for extension \"{}\"", ext))?
            .clone();

        let metadata = fs::metadata(&abs_path)
            .map_err(|e| format!("Cannot stat {}: {}", abs_path.display(), e))?;
        let mtime = metadata
            .modified()
            .map_err(|e| format!("Cannot get mtime: {}", e))?;

        // Check cache
        if let Some(entry) = self.cache.get(&abs_path) {
            if entry.mtime == mtime && entry.grammar == lang_info.grammar {
                let source = fs::read_to_string(&abs_path)
                    .map_err(|e| format!("Cannot read {}: {}", abs_path.display(), e))?;
                return Ok(ParseResult {
                    tree: entry.tree.clone(),
                    language: lang_info,
                    source,
                });
            }
        }

        let source = fs::read_to_string(&abs_path)
            .map_err(|e| format!("Cannot read {}: {}", abs_path.display(), e))?;

        let ts_lang = load_language(lang_info.grammar)
            .ok_or_else(|| format!("Cannot load grammar for \"{}\"", lang_info.grammar))?;

        self.parser
            .set_language(&ts_lang)
            .map_err(|e| format!("Cannot set language: {}", e))?;

        let tree = self
            .parser
            .parse(&source, None)
            .ok_or_else(|| format!("Failed to parse {}", abs_path.display()))?;

        self.cache.insert(
            abs_path,
            CacheEntry {
                tree: tree.clone(),
                mtime,
                grammar: lang_info.grammar.to_string(),
            },
        );

        Ok(ParseResult {
            tree,
            language: lang_info,
            source,
        })
    }

    /// Clear the parse cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

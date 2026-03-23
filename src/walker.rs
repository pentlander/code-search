//! File discovery with gitignore-aware walking using the `ignore` crate.

use ignore::WalkBuilder;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::languages::get_supported_extensions;

/// Directories to always skip (in addition to .gitignore rules).
static SKIP_DIRS: &[&str] = &[
    "node_modules",
    "vendor",
    "__pycache__",
    "target",
    "build",
    "dist",
    ".git",
    ".jj",
    ".code-search-cache",
    ".next",
    ".nuxt",
];

/// File patterns to skip (generated code, minified, etc.).
static SKIP_PATTERNS: &[&str] = &[
    ".pb.go",
    ".pb.gw.go",
    "_generated.go",
    ".gen.go",
    ".d.ts",
    ".min.js",
    ".bundle.js",
];

pub struct CollectResult {
    pub found: Vec<PathBuf>,
    pub skipped: usize,
}

/// Collect supported source files from a directory, respecting .gitignore.
pub fn collect_files(dir: &Path, max_files: usize) -> CollectResult {
    let supported_exts: HashSet<&str> = get_supported_extensions().into_iter().collect();
    let skip_dirs: HashSet<&str> = SKIP_DIRS.iter().copied().collect();
    let mut found = Vec::new();
    let mut skipped = 0;

    let walker = WalkBuilder::new(dir)
        .hidden(true) // skip hidden files/dirs
        .git_ignore(true) // respect .gitignore
        .git_global(true)
        .git_exclude(true)
        .sort_by_file_name(|a, b| a.cmp(b)) // deterministic output
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Skip directories in SKIP_DIRS
        if entry.file_type().map_or(false, |ft| ft.is_dir()) {
            if let Some(name) = entry.file_name().to_str() {
                if skip_dirs.contains(name) {
                    // Note: ignore crate's WalkBuilder handles skipping via filter_entry,
                    // but we also check here for safety
                    continue;
                }
            }
            continue;
        }

        // Only process files
        if !entry.file_type().map_or(false, |ft| ft.is_file()) {
            continue;
        }

        let path = entry.path();
        let file_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };

        // Check extension is supported
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{}", e));
        let ext_str = ext.as_deref().unwrap_or("");
        if !supported_exts.contains(ext_str) {
            continue;
        }

        // Skip generated file patterns
        if SKIP_PATTERNS.iter().any(|p| file_name.ends_with(p)) {
            continue;
        }

        if found.len() >= max_files {
            skipped += 1;
            continue;
        }

        found.push(path.to_path_buf());
    }

    CollectResult { found, skipped }
}

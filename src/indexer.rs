//! Incremental indexing: file walking, hashing, and update orchestration.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::chunker::{extract_chunks, ChunkInfo};
use crate::db::{SearchDB, SymbolRow};
use crate::hasher::hash_content;
use crate::model::Embedder;
use crate::parser::ParserCache;
use crate::walker::collect_files;

pub struct IndexStats {
    pub files_scanned: usize,
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub symbols_indexed: usize,
    pub symbols_reused: usize,
    pub symbols_embedded: usize,
    pub index_time_ms: u64,
    pub embed_time_ms: u64,
}

/// Incrementally index a directory scope.
pub fn index_directory(
    scan_dir: &Path,
    repo_root: &Path,
    db: &mut SearchDB,
    embedder: &mut Embedder,
    parser_cache: &mut ParserCache,
    force: bool,
    on_progress: &dyn Fn(&str),
) -> Result<IndexStats, String> {
    let start = std::time::Instant::now();

    let mut stats = IndexStats {
        files_scanned: 0,
        files_indexed: 0,
        files_skipped: 0,
        files_deleted: 0,
        symbols_indexed: 0,
        symbols_reused: 0,
        symbols_embedded: 0,
        index_time_ms: 0,
        embed_time_ms: 0,
    };

    // Collect files
    on_progress("Scanning files...");
    let result = collect_files(scan_dir, 100_000);
    stats.files_scanned = result.found.len();

    // Scope prefix for filtering existing DB entries
    let scope_prefix = if scan_dir == repo_root {
        None
    } else {
        scan_dir
            .strip_prefix(repo_root)
            .ok()
            .map(|p| p.to_string_lossy().to_string())
    };

    // Get existing indexed files within scope
    let existing_files: HashMap<String, String> = db
        .get_all_files()
        .map_err(|e| format!("DB error: {}", e))?
        .into_iter()
        .filter(|f| match &scope_prefix {
            None => true,
            Some(prefix) => f.path.starts_with(&format!("{}/", prefix)),
        })
        .map(|f| (f.path, f.hash))
        .collect();

    // Diff: what changed, what's new
    struct FileToIndex {
        abs_path: std::path::PathBuf,
        rel_path: String,
        file_hash: String,
    }

    let mut to_index: Vec<FileToIndex> = Vec::new();
    let mut seen = HashSet::new();

    for abs_path in &result.found {
        let rel_path = abs_path
            .strip_prefix(repo_root)
            .unwrap_or(abs_path)
            .to_string_lossy()
            .to_string();
        seen.insert(rel_path.clone());

        let content = std::fs::read_to_string(abs_path)
            .map_err(|e| format!("Cannot read {}: {}", abs_path.display(), e))?;
        let file_hash = hash_content(&content);

        if !force && existing_files.get(&rel_path) == Some(&file_hash) {
            stats.files_skipped += 1;
            continue;
        }
        to_index.push(FileToIndex {
            abs_path: abs_path.clone(),
            rel_path,
            file_hash,
        });
    }

    // Deleted files
    let to_delete: Vec<String> = existing_files
        .keys()
        .filter(|p| !seen.contains(*p))
        .cloned()
        .collect();

    if to_index.is_empty() && to_delete.is_empty() {
        stats.index_time_ms = start.elapsed().as_millis() as u64;
        return Ok(stats);
    }

    let mut parts = Vec::new();
    if !to_index.is_empty() {
        parts.push(format!("{} changed", to_index.len()));
    }
    if !to_delete.is_empty() {
        parts.push(format!("{} deleted", to_delete.len()));
    }
    on_progress(&format!("Updating index: {}...", parts.join(", ")));

    // 1. Delete removed files
    if !to_delete.is_empty() {
        db.delete_files(&to_delete)
            .map_err(|e| format!("DB error: {}", e))?;
        stats.files_deleted = to_delete.len();
    }

    // 2. Extract chunks via tree-sitter
    struct FileChunks {
        chunks: Vec<ChunkInfo>,
        rel_path: String,
        file_hash: String,
        language: String,
    }

    let mut all_file_chunks: Vec<FileChunks> = Vec::new();
    let mut parse_failures: Vec<(String, String)> = Vec::new();

    for file_info in &to_index {
        match extract_chunks(&file_info.abs_path, repo_root, parser_cache) {
            Ok(chunks) => {
                let lang = chunks
                    .first()
                    .map(|c| c.language.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                all_file_chunks.push(FileChunks {
                    chunks,
                    rel_path: file_info.rel_path.clone(),
                    file_hash: file_info.file_hash.clone(),
                    language: lang,
                });
            }
            Err(_) => {
                parse_failures.push((file_info.rel_path.clone(), file_info.file_hash.clone()));
                stats.files_skipped += 1;
            }
        }
    }

    // Record parse failures so incremental re-index skips them
    if !parse_failures.is_empty() {
        let file_records: Vec<(String, String, Option<String>, i32)> = parse_failures
            .iter()
            .map(|(path, hash)| (path.clone(), hash.clone(), None, 0))
            .collect();
        db.upsert_files(&file_records)
            .map_err(|e| format!("DB error: {}", e))?;
    }

    let all_chunks: Vec<&ChunkInfo> = all_file_chunks
        .iter()
        .flat_map(|f| f.chunks.iter())
        .collect();
    let changed_file_paths: Vec<String> = all_file_chunks.iter().map(|f| f.rel_path.clone()).collect();

    // 3. Incremental embed + store: reuse embeddings for unchanged signatures
    if !all_chunks.is_empty() || !changed_file_paths.is_empty() {
        on_progress(&format!(
            "Updating {} symbols (reusing unchanged)...",
            all_chunks.len()
        ));

        let embed_start = std::time::Instant::now();

        // Load existing embeddings for reuse
        let path_refs: Vec<&str> = changed_file_paths.iter().map(|s| s.as_str()).collect();
        let existing_embeddings = db
            .get_symbol_embeddings_for_files(&path_refs)
            .map_err(|e| format!("DB error: {}", e))?;

        // Partition: reuse vs need embedding
        let mut reusable_indices = Vec::new();
        let mut need_embedding_indices = Vec::new();

        for (i, chunk) in all_chunks.iter().enumerate() {
            if existing_embeddings.contains_key(&chunk.embedding_text) {
                reusable_indices.push(i);
            } else {
                need_embedding_indices.push(i);
            }
        }

        stats.symbols_reused = reusable_indices.len();
        stats.symbols_embedded = need_embedding_indices.len();

        // Embed only truly new/changed symbols
        let new_embeddings: Vec<Vec<f32>> = if !need_embedding_indices.is_empty() {
            let texts: Vec<String> = need_embedding_indices
                .iter()
                .map(|&i| all_chunks[i].embedding_text.clone())
                .collect();
            embedder.embed(&texts, false)?
        } else {
            Vec::new()
        };

        stats.embed_time_ms = embed_start.elapsed().as_millis() as u64;

        // Delete old symbols for changed files
        db.delete_symbols_for_files(&changed_file_paths)
            .map_err(|e| format!("DB error: {}", e))?;

        // Build symbol rows for insertion
        let mut symbol_rows: Vec<SymbolRow> = Vec::new();

        // Reused symbols
        for &i in &reusable_indices {
            let chunk = &all_chunks[i];
            let emb_bytes = existing_embeddings.get(&chunk.embedding_text).unwrap();
            let embedding: Vec<f32> = bytemuck::cast_slice(emb_bytes).to_vec();
            symbol_rows.push(SymbolRow {
                file_path: chunk.file_path.clone(),
                line: chunk.line,
                name: chunk.name.clone(),
                kind: chunk.kind.clone(),
                language: chunk.language.clone(),
                end_line: chunk.end_line,
                signature: chunk.signature.clone(),
                embedding_text: chunk.embedding_text.clone(),
                embedding,
            });
        }

        // Newly embedded symbols
        for (seq, &i) in need_embedding_indices.iter().enumerate() {
            let chunk = &all_chunks[i];
            symbol_rows.push(SymbolRow {
                file_path: chunk.file_path.clone(),
                line: chunk.line,
                name: chunk.name.clone(),
                kind: chunk.kind.clone(),
                language: chunk.language.clone(),
                end_line: chunk.end_line,
                signature: chunk.signature.clone(),
                embedding_text: chunk.embedding_text.clone(),
                embedding: new_embeddings[seq].clone(),
            });
        }

        db.insert_symbols(&symbol_rows)
            .map_err(|e| format!("DB error: {}", e))?;

        stats.symbols_indexed = all_chunks.len();
    }

    // 5. Update file records
    let mut file_symbol_counts: HashMap<String, i32> = HashMap::new();
    for chunk in &all_chunks {
        *file_symbol_counts
            .entry(chunk.file_path.clone())
            .or_insert(0) += 1;
    }

    let file_records: Vec<(String, String, Option<String>, i32)> = all_file_chunks
        .iter()
        .map(|f| {
            (
                f.rel_path.clone(),
                f.file_hash.clone(),
                Some(f.language.clone()),
                *file_symbol_counts.get(&f.rel_path).unwrap_or(&0),
            )
        })
        .collect();

    db.upsert_files(&file_records)
        .map_err(|e| format!("DB error: {}", e))?;

    stats.files_indexed = all_file_chunks.len();
    stats.index_time_ms = start.elapsed().as_millis() as u64;
    Ok(stats)
}

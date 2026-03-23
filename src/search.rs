//! Multi-query search with dedup and merge.

use std::collections::HashMap;

use crate::db::{SearchDB, SearchResult};
use crate::model::Embedder;

/// Multi-query search: embed all queries, run each against DB,
/// deduplicate by (file_path, line, name) keeping the best score,
/// return top_k sorted by score descending.
pub fn multi_query_search(
    queries: &[String],
    top_k: i32,
    threshold: f64,
    language: Option<&str>,
    kind: Option<&str>,
    path_prefix: Option<&str>,
    db: &SearchDB,
    embedder: &mut Embedder,
) -> Result<Vec<SearchResult>, String> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }

    // Batch-embed all queries at once
    let query_embeddings = embedder.embed(&queries.to_vec(), true)?;

    // Fetch more per-query so we have enough after dedup
    let per_query_k = if queries.len() > 1 {
        (top_k as f64 * 1.5).ceil() as i32
    } else {
        top_k
    };

    // Run each query and merge results, keeping best score per symbol
    let mut best_by_key: HashMap<String, SearchResult> = HashMap::new();

    for emb in &query_embeddings {
        let results = db
            .search(emb, per_query_k, language, kind, path_prefix)
            .map_err(|e| format!("Search error: {}", e))?;

        for r in results {
            let key = format!("{}:{}:{}", r.file_path, r.line, r.name);
            let existing = best_by_key.get(&key);
            if existing.map_or(true, |e| r.score > e.score) {
                best_by_key.insert(key, r);
            }
        }
    }

    // Sort by best score, filter by threshold, take top_k
    let mut merged: Vec<_> = best_by_key
        .into_values()
        .filter(|r| r.score >= threshold)
        .collect();
    merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    merged.truncate(top_k as usize);

    Ok(merged)
}

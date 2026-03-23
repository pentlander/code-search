//! SQLite database layer with simsimd brute-force vector search.
//!
//! Embeddings stored as BLOBs in a regular table. Search uses simsimd
//! L2² distance with a top-K heap. No sqlite-vec dependency needed.

use rusqlite::{params, Connection, Result as SqlResult};
use simsimd::SpatialSimilarity;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

const SCHEMA_VERSION: i32 = 4;

#[derive(Debug, Clone)]
pub struct FileRow {
    pub path: String,
    pub hash: String,
    pub language: Option<String>,
    pub symbol_count: Option<i32>,
    pub indexed_at: i64,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file_path: String,
    pub name: String,
    pub kind: String,
    pub language: String,
    pub line: i32,
    pub end_line: Option<i32>,
    pub signature: Option<String>,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct Stats {
    pub symbol_count: i64,
    pub file_count: i64,
}

pub struct SearchDB {
    conn: Connection,
}

impl SearchDB {
    pub fn open(db_path: &Path) -> SqlResult<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let conn = Connection::open(db_path)?;

        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "mmap_size", 3_000_000_000i64)?;
        conn.pragma_update(None, "temp_store", 2)?;
        conn.pragma_update(None, "cache_size", -64000)?;

        let mut db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&mut self) -> SqlResult<()> {
        let has_meta: bool = self
            .conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='meta'",
                [],
                |r| r.get::<_, i64>(0),
            )
            .map(|c| c > 0)?;

        if has_meta {
            let version: Option<String> = self
                .conn
                .query_row(
                    "SELECT value FROM meta WHERE key = 'schema_version'",
                    [],
                    |r| r.get(0),
                )
                .ok();

            if let Some(v) = version {
                if v.parse::<i32>().unwrap_or(0) == SCHEMA_VERSION {
                    return Ok(());
                }
            }

            self.conn.execute_batch(
                "DROP TABLE IF EXISTS files;
                 DROP TABLE IF EXISTS symbols;
                 DROP TABLE IF EXISTS vec_symbols;
                 DROP TABLE IF EXISTS meta;",
            )?;
        }

        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                language TEXT,
                symbol_count INTEGER,
                indexed_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS symbols (
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                language TEXT NOT NULL,
                end_line INTEGER,
                signature TEXT,
                embedding_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (file_path, line)
            ) WITHOUT ROWID;

            CREATE INDEX IF NOT EXISTS idx_symbols_language ON symbols(language);
            CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);",
        )?;

        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            params!["schema_version", SCHEMA_VERSION.to_string()],
        )?;
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            params!["model", "CodeRankEmbed"],
        )?;
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            params!["dimensions", "768"],
        )?;

        Ok(())
    }

    pub fn get_all_files(&self) -> SqlResult<Vec<FileRow>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT path, hash, language, symbol_count, indexed_at FROM files",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(FileRow {
                path: r.get(0)?,
                hash: r.get(1)?,
                language: r.get(2)?,
                symbol_count: r.get(3)?,
                indexed_at: r.get(4)?,
            })
        })?;
        rows.collect()
    }

    pub fn delete_files(&mut self, paths: &[String]) -> SqlResult<()> {
        let tx = self.conn.transaction()?;
        for path in paths {
            tx.execute("DELETE FROM symbols WHERE file_path = ?", params![path])?;
            tx.execute("DELETE FROM files WHERE path = ?", params![path])?;
        }
        tx.commit()
    }

    pub fn upsert_files(&mut self, files: &[(String, String, Option<String>, i32)]) -> SqlResult<()> {
        let tx = self.conn.transaction()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        for (path, hash, language, symbol_count) in files {
            tx.execute(
                "INSERT OR REPLACE INTO files (path, hash, language, symbol_count, indexed_at) VALUES (?, ?, ?, ?, ?)",
                params![path, hash, language, symbol_count, now],
            )?;
        }
        tx.commit()
    }

    /// Insert symbols with pre-computed embeddings.
    pub fn insert_symbols(
        &mut self,
        symbols: &[SymbolRow],
    ) -> SqlResult<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO symbols (file_path, line, name, kind, language, end_line, signature, embedding_text, embedding)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )?;
            for sym in symbols {
                let embedding_bytes: &[u8] = bytemuck::cast_slice(&sym.embedding);
                stmt.execute(params![
                    sym.file_path,
                    sym.line,
                    sym.name,
                    sym.kind,
                    sym.language,
                    sym.end_line,
                    sym.signature,
                    sym.embedding_text,
                    embedding_bytes
                ])?;
            }
        }
        tx.commit()
    }

    /// Delete symbols for specific files.
    pub fn delete_symbols_for_files(&mut self, file_paths: &[String]) -> SqlResult<()> {
        let tx = self.conn.transaction()?;
        for path in file_paths {
            tx.execute("DELETE FROM symbols WHERE file_path = ?", params![path])?;
        }
        tx.commit()
    }

    /// Get existing symbol embeddings for a set of files (for incremental reuse).
    pub fn get_symbol_embeddings_for_files(
        &self,
        file_paths: &[&str],
    ) -> SqlResult<HashMap<String, Vec<u8>>> {
        let mut map = HashMap::new();
        for path in file_paths {
            let mut stmt = self.conn.prepare_cached(
                "SELECT embedding_text, embedding FROM symbols WHERE file_path = ?",
            )?;
            let rows = stmt.query_map(params![path], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
            })?;
            for row in rows {
                let (text, emb) = row?;
                map.insert(text, emb);
            }
        }
        Ok(map)
    }

    /// Search using simsimd L2².
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: i32,
        language: Option<&str>,
        kind: Option<&str>,
        path_prefix: Option<&str>,
    ) -> SqlResult<Vec<SearchResult>> {
        let mut where_clauses = Vec::new();
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(prefix) = path_prefix {
            where_clauses.push("file_path LIKE ?");
            param_values.push(Box::new(format!("{}/%", prefix)));
        }
        if let Some(lang) = language {
            where_clauses.push("language = ?");
            param_values.push(Box::new(lang.to_string()));
        }
        if let Some(k) = kind {
            where_clauses.push("kind = ?");
            param_values.push(Box::new(k.to_string()));
        }

        let where_str = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let sql = format!(
            "SELECT file_path, line, name, kind, language, end_line, signature, embedding
             FROM symbols {}",
            where_str
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let mut rows = stmt.query(params_ref.as_slice())?;

        let top_k = top_k as usize;
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(top_k + 1);

        while let Some(row) = rows.next()? {
            let blob = row.get_ref(7)?.as_blob()?;
            let emb: &[f32] = bytemuck::cast_slice(blob);
            let dist = f32::l2sq(query_embedding, emb).unwrap_or(f64::MAX);

            if heap.len() < top_k {
                heap.push(HeapItem {
                    dist,
                    file_path: row.get(0)?,
                    line: row.get(1)?,
                    name: row.get(2)?,
                    kind: row.get(3)?,
                    language: row.get(4)?,
                    end_line: row.get(5)?,
                    signature: row.get(6)?,
                });
            } else if dist < heap.peek().unwrap().dist {
                heap.pop();
                heap.push(HeapItem {
                    dist,
                    file_path: row.get(0)?,
                    line: row.get(1)?,
                    name: row.get(2)?,
                    kind: row.get(3)?,
                    language: row.get(4)?,
                    end_line: row.get(5)?,
                    signature: row.get(6)?,
                });
            }
        }

        let mut results: Vec<_> = heap.into_vec();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());

        Ok(results
            .into_iter()
            .map(|item| SearchResult {
                file_path: item.file_path,
                name: item.name,
                kind: item.kind,
                language: item.language,
                line: item.line,
                end_line: item.end_line,
                signature: item.signature,
                score: 1.0 - (item.dist / 2.0),
            })
            .collect())
    }

    pub fn get_stats(&self) -> SqlResult<Stats> {
        let symbol_count: i64 = self
            .conn
            .query_row("SELECT count(*) FROM symbols", [], |r| r.get(0))?;
        let file_count: i64 = self
            .conn
            .query_row("SELECT count(*) FROM files", [], |r| r.get(0))?;
        Ok(Stats {
            symbol_count,
            file_count,
        })
    }

    pub fn transaction(&mut self) -> SqlResult<rusqlite::Transaction<'_>> {
        self.conn.transaction()
    }
}

// ─── Symbol row for batch insert ───────────────────────────────────────

pub struct SymbolRow {
    pub file_path: String,
    pub line: i32,
    pub name: String,
    pub kind: String,
    pub language: String,
    pub end_line: Option<i32>,
    pub signature: Option<String>,
    pub embedding_text: String,
    pub embedding: Vec<f32>,
}

// ─── Top-K heap item ───────────────────────────────────────────────────

struct HeapItem {
    dist: f64,
    file_path: String,
    name: String,
    kind: String,
    language: String,
    line: i32,
    end_line: Option<i32>,
    signature: Option<String>,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

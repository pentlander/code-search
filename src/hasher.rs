//! Fast content hashing using xxhash for incremental index diffing.

use xxhash_rust::xxh64::xxh64;

/// Hash file content, returning a hex string.
pub fn hash_content(content: &str) -> String {
    let h = xxh64(content.as_bytes(), 0);
    format!("{:016x}", h)
}

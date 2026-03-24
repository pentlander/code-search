//! Language detection and grammar mapping.
//!
//! Maps file extensions to tree-sitter grammars and human-readable names.

use tree_sitter::Language;

#[derive(Debug, Clone)]
pub struct LanguageInfo {
    /// Tree-sitter grammar name (used as key in symbol specs)
    pub grammar: &'static str,
    /// Human-readable language name
    pub name: &'static str,
}

/// Map of file extensions (with dot) to language info.
static EXTENSION_MAP: &[(&str, LanguageInfo)] = &[
    // TypeScript / JavaScript
    (".ts", LanguageInfo { grammar: "typescript", name: "TypeScript" }),
    (".tsx", LanguageInfo { grammar: "tsx", name: "TSX" }),
    (".js", LanguageInfo { grammar: "javascript", name: "JavaScript" }),
    (".jsx", LanguageInfo { grammar: "javascript", name: "JavaScript" }),
    (".mjs", LanguageInfo { grammar: "javascript", name: "JavaScript" }),
    (".cjs", LanguageInfo { grammar: "javascript", name: "JavaScript" }),
    (".mts", LanguageInfo { grammar: "typescript", name: "TypeScript" }),
    (".cts", LanguageInfo { grammar: "typescript", name: "TypeScript" }),
    // Python
    (".py", LanguageInfo { grammar: "python", name: "Python" }),
    (".pyi", LanguageInfo { grammar: "python", name: "Python" }),
    // Rust
    (".rs", LanguageInfo { grammar: "rust", name: "Rust" }),
    // Go
    (".go", LanguageInfo { grammar: "go", name: "Go" }),
    // Java
    (".java", LanguageInfo { grammar: "java", name: "Java" }),
    // Kotlin
    (".kt", LanguageInfo { grammar: "kotlin", name: "Kotlin" }),
    (".kts", LanguageInfo { grammar: "kotlin", name: "Kotlin" }),
    // Swift
    (".swift", LanguageInfo { grammar: "swift", name: "Swift" }),
    // Ruby
    (".rb", LanguageInfo { grammar: "ruby", name: "Ruby" }),
    // PHP
    (".php", LanguageInfo { grammar: "php", name: "PHP" }),
    // C#
    (".cs", LanguageInfo { grammar: "c_sharp", name: "C#" }),
    // Scala
    (".scala", LanguageInfo { grammar: "scala", name: "Scala" }),
    // Lua
    (".lua", LanguageInfo { grammar: "lua", name: "Lua" }),
    // Bash / Shell
    (".sh", LanguageInfo { grammar: "bash", name: "Bash" }),
    (".bash", LanguageInfo { grammar: "bash", name: "Bash" }),
    (".zsh", LanguageInfo { grammar: "bash", name: "Bash" }),
    // Zig
    (".zig", LanguageInfo { grammar: "zig", name: "Zig" }),
    // Elixir
    (".ex", LanguageInfo { grammar: "elixir", name: "Elixir" }),
    (".exs", LanguageInfo { grammar: "elixir", name: "Elixir" }),
    // Dart
    (".dart", LanguageInfo { grammar: "dart", name: "Dart" }),
    // OCaml
    (".ml", LanguageInfo { grammar: "ocaml", name: "OCaml" }),
    (".mli", LanguageInfo { grammar: "ocaml", name: "OCaml" }),
    // YAML
    (".yml", LanguageInfo { grammar: "yaml", name: "YAML" }),
    (".yaml", LanguageInfo { grammar: "yaml", name: "YAML" }),
    // TOML
    (".toml", LanguageInfo { grammar: "toml", name: "TOML" }),
    // Protobuf
    (".proto", LanguageInfo { grammar: "proto", name: "Protobuf" }),
    // HCL / Terraform
    (".hcl", LanguageInfo { grammar: "hcl", name: "HCL" }),
    (".tf", LanguageInfo { grammar: "terraform", name: "Terraform" }),
    (".tfvars", LanguageInfo { grammar: "terraform", name: "Terraform" }),
];

/// Get language info for a file extension (including the dot).
pub fn get_language_for_extension(ext: &str) -> Option<&'static LanguageInfo> {
    let ext_lower = ext.to_lowercase();
    EXTENSION_MAP
        .iter()
        .find(|(e, _)| *e == ext_lower)
        .map(|(_, info)| info)
}

/// Get all supported file extensions.
pub fn get_supported_extensions() -> Vec<&'static str> {
    EXTENSION_MAP.iter().map(|(ext, _)| *ext).collect()
}

/// Load a tree-sitter Language for a grammar name.
pub fn load_language(grammar: &str) -> Option<Language> {
    match grammar {
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "javascript" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "typescript" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "kotlin" => Some(tree_sitter_kotlin_ng::LANGUAGE.into()),
        "swift" => Some(tree_sitter_swift::LANGUAGE.into()),
        "ruby" => Some(tree_sitter_ruby::LANGUAGE.into()),
        "php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),
        "c_sharp" => Some(tree_sitter_c_sharp::LANGUAGE.into()),
        "scala" => Some(tree_sitter_scala::LANGUAGE.into()),
        "lua" => Some(tree_sitter_lua::LANGUAGE.into()),
        "bash" => Some(tree_sitter_bash::LANGUAGE.into()),
        "zig" => Some(tree_sitter_zig::LANGUAGE.into()),
        "elixir" => Some(tree_sitter_elixir::LANGUAGE.into()),
        "dart" => Some(tree_sitter_dart::LANGUAGE.into()),
        "ocaml" => Some(tree_sitter_ocaml::LANGUAGE_OCAML.into()),
        "yaml" => Some(tree_sitter_yaml::LANGUAGE.into()),
        "toml" => Some(tree_sitter_toml_ng::LANGUAGE.into()),
        "proto" => Some(tree_sitter_proto::LANGUAGE.into()),
        "hcl" => Some(tree_sitter_hcl::LANGUAGE.into()),
        "terraform" => Some(tree_sitter_hcl::LANGUAGE.into()),
        _ => None,
    }
}

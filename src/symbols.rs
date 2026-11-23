//! Symbol extraction from tree-sitter ASTs.
//!
//! Walks the AST and extracts meaningful symbols (functions, classes, methods, etc.)
//! with their names, kinds, line ranges, nesting, and optional signatures.

use regex::Regex;
use tree_sitter::{Node, Tree};

// ─── Types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Interface,
    Type,
    Enum,
    Struct,
    Trait,
    Impl,
    Module,
    Variable,
    Constant,
    Property,
    Block,
    Resource,
    Data,
    Provider,
    Output,
    Locals,
    Section,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Method => "method",
            Self::Class => "class",
            Self::Interface => "interface",
            Self::Type => "type",
            Self::Enum => "enum",
            Self::Struct => "struct",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Module => "module",
            Self::Variable => "variable",
            Self::Constant => "constant",
            Self::Property => "property",
            Self::Block => "block",
            Self::Resource => "resource",
            Self::Data => "data",
            Self::Provider => "provider",
            Self::Output => "output",
            Self::Locals => "locals",
            Self::Section => "section",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Function => "fn",
            Self::Method => "method",
            Self::Class => "class",
            Self::Interface => "iface",
            Self::Type => "type",
            Self::Enum => "enum",
            Self::Struct => "struct",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Module => "mod",
            Self::Variable => "var",
            Self::Constant => "const",
            Self::Property => "prop",
            Self::Block => "block",
            Self::Resource => "resource",
            Self::Data => "data",
            Self::Provider => "provider",
            Self::Output => "output",
            Self::Locals => "locals",
            Self::Section => "section",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "function" => Some(Self::Function),
            "method" => Some(Self::Method),
            "class" => Some(Self::Class),
            "interface" => Some(Self::Interface),
            "type" => Some(Self::Type),
            "enum" => Some(Self::Enum),
            "struct" => Some(Self::Struct),
            "trait" => Some(Self::Trait),
            "impl" => Some(Self::Impl),
            "module" => Some(Self::Module),
            "variable" => Some(Self::Variable),
            "constant" => Some(Self::Constant),
            "property" => Some(Self::Property),
            "block" => Some(Self::Block),
            "resource" => Some(Self::Resource),
            "data" => Some(Self::Data),
            "provider" => Some(Self::Provider),
            "output" => Some(Self::Output),
            "locals" => Some(Self::Locals),
            "section" => Some(Self::Section),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: SymbolKind,
    pub start_line: usize, // 1-based
    pub end_line: usize,   // 1-based
    pub signature: Option<String>,
    pub children: Vec<SymbolInfo>,
}

pub struct ExtractOptions {
    pub signatures: bool,
    pub kind: Option<String>,
    pub name_pattern: Option<String>,
    pub top_level_only: bool,
    pub max_depth: Option<usize>,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            signatures: false,
            kind: None,
            name_pattern: None,
            top_level_only: false,
            max_depth: None,
        }
    }
}

// ─── Language Spec ─────────────────────────────────────────────────────

struct LanguageSpec {
    node_kind: fn(&str) -> Option<SymbolKind>,
    get_name: for<'a> fn(Node<'a>, SymbolKind, &[u8]) -> Option<String>,
    get_signature: for<'a> fn(Node<'a>, &[u8]) -> Option<String>,
    resolve_kind: for<'a> fn(Node<'a>, SymbolKind) -> SymbolKind,
    is_container: fn(&str) -> bool,
}

fn get_spec(grammar: &str) -> Option<&'static LanguageSpec> {
    match grammar {
        "typescript" | "tsx" => Some(&TYPESCRIPT_SPEC),
        "javascript" => Some(&JAVASCRIPT_SPEC),
        "python" => Some(&PYTHON_SPEC),
        "rust" => Some(&RUST_SPEC),
        "go" => Some(&GO_SPEC),
        "java" => Some(&JAVA_SPEC),
        "kotlin" => Some(&KOTLIN_SPEC),
        "ruby" => Some(&RUBY_SPEC),
        "swift" => Some(&SWIFT_SPEC),
        "c_sharp" => Some(&CSHARP_SPEC),
        "lua" => Some(&LUA_SPEC),
        "bash" => Some(&BASH_SPEC),
        "scala" => Some(&SCALA_SPEC),
        "php" => Some(&PHP_SPEC),
        "zig" => Some(&ZIG_SPEC),
        "elixir" => Some(&ELIXIR_SPEC),
        "dart" => Some(&DART_SPEC),
        "ocaml" => Some(&OCAML_SPEC),
        "yaml" => Some(&YAML_SPEC),
        "toml" => Some(&TOML_SPEC),
        "proto" => Some(&PROTO_SPEC),
        "hcl" => Some(&HCL_SPEC),
        "terraform" => Some(&TERRAFORM_SPEC),
        "markdown" => Some(&MARKDOWN_SPEC),
        _ => None,
    }
}

// ─── Node helpers ──────────────────────────────────────────────────────

fn field_name<'a>(node: Node<'a>, field: &str, source: &[u8]) -> Option<String> {
    node.child_by_field_name(field)
        .and_then(|n| n.utf8_text(source).ok())
        .map(|s| s.to_string())
}

fn first_child_of_type<'a>(node: Node<'a>, type_name: &str) -> Option<Node<'a>> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if child.kind() == type_name {
                return Some(child);
            }
        }
    }
    None
}

fn node_text<'a>(node: Node<'a>, source: &'a [u8]) -> &'a str {
    node.utf8_text(source).unwrap_or("")
}

/// Shorten `text` to at most `max` bytes, ending in "..." when cut.
/// Cuts on a char boundary, so multi-byte names never panic.
pub fn truncate_display(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut cut = max.saturating_sub(3);
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}...", &text[..cut])
}

/// 1-based inclusive line range of a node. A node whose end sits at column 0
/// of a later row (it swallowed a trailing newline) ends on the previous line.
pub fn line_range(node: Node) -> (usize, usize) {
    let start = node.start_position();
    let end = node.end_position();
    let end_line = if end.column == 0 && end.row > start.row {
        end.row
    } else {
        end.row + 1
    };
    (start.row + 1, end_line)
}

/// A symbol recognized at a single AST node, named and kinded as the outline shows it.
pub struct NodeSymbol<'a> {
    /// The node the language spec recognized.
    pub node: Node<'a>,
    /// The node whose lines the outline reports: an enclosing `export` or decorator wrapper.
    pub range_node: Node<'a>,
    pub name: String,
    pub kind: SymbolKind,
}

/// Recognize the symbol a node defines, if any. Python decorator wrappers return
/// `None`; their inner definition carries the symbol and is widened to cover them.
pub fn node_symbol<'a>(node: Node<'a>, grammar: &str, source: &[u8]) -> Option<NodeSymbol<'a>> {
    let spec = get_spec(grammar)?;
    if node.kind() == "decorated_definition" {
        return None;
    }
    let raw = (spec.node_kind)(node.kind())?;
    let kind = (spec.resolve_kind)(node, raw);
    let name = truncate_display(&(spec.get_name)(node, kind, source)?, 80);
    let kind = if grammar == "hcl" || grammar == "terraform" {
        map_hcl_kind(kind, &name)
    } else {
        kind
    };
    Some(NodeSymbol {
        node,
        range_node: range_owner(node),
        name,
        kind,
    })
}

/// Signature for a symbol node, truncated the way the outline truncates it.
pub fn node_signature(node: Node, grammar: &str, source: &[u8]) -> Option<String> {
    let spec = get_spec(grammar)?;
    let owner = range_owner(node);
    let target = if owner.kind() == "decorated_definition" {
        owner
    } else {
        node
    };
    (spec.get_signature)(target, source).map(|s| truncate_display(&s, 120))
}

/// The node whose range the outline reports for a symbol node.
fn range_owner(node: Node) -> Node {
    if let Some(parent) = node.parent() {
        let wraps = match parent.kind() {
            "export_statement" | "export_declaration" => {
                parent.child_by_field_name("declaration") == Some(node)
            }
            "decorated_definition" => parent.child_by_field_name("definition") == Some(node),
            _ => false,
        };
        if wraps {
            return parent;
        }
    }
    node
}

fn is_inside_class(node: Node) -> bool {
    let mut current = node.parent();
    while let Some(n) = current {
        match n.kind() {
            "class_definition" | "class_declaration" | "class_body" => return true,
            _ => {}
        }
        current = n.parent();
    }
    false
}

// ─── Signature helpers ─────────────────────────────────────────────────

fn extract_signature_line(node: Node, source: &[u8]) -> Option<String> {
    let start = node.start_byte();
    let end = (start + 500).min(source.len());
    let text = std::str::from_utf8(&source[start..end]).unwrap_or("");
    let brace_idx = text.find('{');
    let newline_idx = text.find('\n');

    let mut end_pos = text.len();
    if let Some(bi) = brace_idx {
        end_pos = end_pos.min(bi);
    }
    if let Some(ni) = newline_idx {
        end_pos = end_pos.min(ni);
    }

    let sig = text[..end_pos].trim().trim_end_matches(|c: char| c == '{' || c == ':' || c.is_whitespace());
    if sig.is_empty() {
        None
    } else {
        Some(sig.to_string())
    }
}

fn extract_params(node: Node, source: &[u8]) -> Option<String> {
    node.child_by_field_name("parameters")
        .or_else(|| node.child_by_field_name("params"))
        .or_else(|| first_child_of_type(node, "formal_parameters"))
        .or_else(|| first_child_of_type(node, "parameters"))
        .or_else(|| first_child_of_type(node, "parameter_list"))
        .map(|n| node_text(n, source).to_string())
}

fn extract_return_type(node: Node, source: &[u8]) -> Option<String> {
    node.child_by_field_name("return_type")
        .or_else(|| node.child_by_field_name("result"))
        .map(|n| node_text(n, source).to_string())
}

fn build_signature(name: &str, node: Node, source: &[u8]) -> Option<String> {
    let params = extract_params(node, source)?;
    match extract_return_type(node, source) {
        Some(ret) => {
            let ret_text = ret.trim_start_matches(':').trim();
            Some(format!("{}{}: {}", name, params, ret_text))
        }
        None => Some(format!("{}{}", name, params)),
    }
}

// ─── TypeScript / JavaScript ───────────────────────────────────────────

fn ts_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_declaration" | "generator_function_declaration" => Some(SymbolKind::Function),
        "class_declaration" | "abstract_class_declaration" => Some(SymbolKind::Class),
        "method_definition" => Some(SymbolKind::Method),
        "interface_declaration" => Some(SymbolKind::Interface),
        "type_alias_declaration" => Some(SymbolKind::Type),
        "enum_declaration" => Some(SymbolKind::Enum),
        "lexical_declaration" | "variable_declaration" => Some(SymbolKind::Function),
        "module" => Some(SymbolKind::Module),
        _ => None,
    }
}

fn ts_get_name(node: Node, kind: SymbolKind, source: &[u8]) -> Option<String> {
    let nk = node.kind();
    if nk == "lexical_declaration" || nk == "variable_declaration" {
        let declarator = first_child_of_type(node, "variable_declarator");
        if let Some(decl) = declarator {
            let init = decl.child_by_field_name("value");
            if let Some(init_node) = init {
                let init_type = init_node.kind();
                if init_type == "arrow_function"
                    || init_type == "function_expression"
                    || init_type == "function"
                {
                    return field_name(decl, "name", source);
                }
            }
        }
        return None;
    }
    if nk == "export_statement" {
        let decl = node.child_by_field_name("declaration")?;
        return ts_get_name(decl, kind, source);
    }
    field_name(node, "name", source)
}

fn ts_get_signature(node: Node, source: &[u8]) -> Option<String> {
    let name = ts_get_name(node, SymbolKind::Function, source)?;
    let nk = node.kind();
    if nk == "lexical_declaration" || nk == "variable_declaration" {
        let declarator = first_child_of_type(node, "variable_declarator")?;
        let init = declarator.child_by_field_name("value")?;
        return build_signature(&name, init, source);
    }
    build_signature(&name, node, source)
}

fn ts_is_container(node_type: &str) -> bool {
    matches!(
        node_type,
        "class_declaration"
            | "abstract_class_declaration"
            | "interface_declaration"
            | "enum_declaration"
            | "module"
    )
}

static TYPESCRIPT_SPEC: LanguageSpec = LanguageSpec {
    node_kind: ts_node_kind,
    get_name: ts_get_name,
    get_signature: ts_get_signature,
    resolve_kind: |_, kind| kind,
    is_container: ts_is_container,
};

fn js_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_declaration" | "generator_function_declaration" => Some(SymbolKind::Function),
        "class_declaration" => Some(SymbolKind::Class),
        "method_definition" => Some(SymbolKind::Method),
        "lexical_declaration" | "variable_declaration" => Some(SymbolKind::Function),
        _ => None,
    }
}

static JAVASCRIPT_SPEC: LanguageSpec = LanguageSpec {
    node_kind: js_node_kind,
    get_name: ts_get_name,
    get_signature: ts_get_signature,
    resolve_kind: |_, kind| kind,
    is_container: |t| t == "class_declaration",
};

// ─── Python ────────────────────────────────────────────────────────────

fn py_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_definition" => Some(SymbolKind::Function),
        "class_definition" => Some(SymbolKind::Class),
        "decorated_definition" => Some(SymbolKind::Function),
        _ => None,
    }
}

fn py_get_name(node: Node, _kind: SymbolKind, source: &[u8]) -> Option<String> {
    if node.kind() == "decorated_definition" {
        let def = node.child_by_field_name("definition")?;
        return field_name(def, "name", source);
    }
    field_name(node, "name", source)
}

fn py_resolve_kind(node: Node, kind: SymbolKind) -> SymbolKind {
    if node.kind() == "decorated_definition" {
        if let Some(def) = node.child_by_field_name("definition") {
            if def.kind() == "class_definition" {
                return SymbolKind::Class;
            }
            if def.kind() == "function_definition" {
                return if is_inside_class(node) {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
            }
        }
    }
    if node.kind() == "function_definition" && is_inside_class(node) {
        return SymbolKind::Method;
    }
    kind
}

fn py_get_signature(node: Node, source: &[u8]) -> Option<String> {
    let target = if node.kind() == "decorated_definition" {
        node.child_by_field_name("definition")?
    } else {
        node
    };
    let name = field_name(target, "name", source)?;
    let params = target
        .child_by_field_name("parameters")
        .map(|n| node_text(n, source).to_string())?;
    match target.child_by_field_name("return_type") {
        Some(ret) => Some(format!("{}{} -> {}", name, params, node_text(ret, source))),
        None => Some(format!("{}{}", name, params)),
    }
}

fn py_is_container(node_type: &str) -> bool {
    node_type == "class_definition"
}

static PYTHON_SPEC: LanguageSpec = LanguageSpec {
    node_kind: py_node_kind,
    get_name: py_get_name,
    get_signature: py_get_signature,
    resolve_kind: py_resolve_kind,
    is_container: py_is_container,
};

// ─── Rust ──────────────────────────────────────────────────────────────

fn rust_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_item" => Some(SymbolKind::Function),
        "struct_item" => Some(SymbolKind::Struct),
        "enum_item" => Some(SymbolKind::Enum),
        "impl_item" => Some(SymbolKind::Impl),
        "trait_item" => Some(SymbolKind::Trait),
        "mod_item" => Some(SymbolKind::Module),
        "type_item" => Some(SymbolKind::Type),
        "const_item" | "static_item" => Some(SymbolKind::Constant),
        "macro_definition" => Some(SymbolKind::Function),
        _ => None,
    }
}

fn rust_get_name(node: Node, kind: SymbolKind, source: &[u8]) -> Option<String> {
    if kind == SymbolKind::Impl {
        let type_node = node.child_by_field_name("type")?;
        let trait_node = node.child_by_field_name("trait");
        if let Some(tr) = trait_node {
            return Some(format!(
                "{} for {}",
                node_text(tr, source),
                node_text(type_node, source)
            ));
        }
        return Some(node_text(type_node, source).to_string());
    }
    field_name(node, "name", source)
}

fn rust_get_signature(node: Node, source: &[u8]) -> Option<String> {
    let name = field_name(node, "name", source)?;
    build_signature(&name, node, source)
}

fn rust_is_container(node_type: &str) -> bool {
    matches!(
        node_type,
        "impl_item" | "trait_item" | "mod_item" | "struct_item" | "enum_item"
    )
}

static RUST_SPEC: LanguageSpec = LanguageSpec {
    node_kind: rust_node_kind,
    get_name: rust_get_name,
    get_signature: rust_get_signature,
    resolve_kind: |_, kind| kind,
    is_container: rust_is_container,
};

// ─── Go ────────────────────────────────────────────────────────────────

fn go_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_declaration" => Some(SymbolKind::Function),
        "method_declaration" => Some(SymbolKind::Method),
        "type_declaration" => Some(SymbolKind::Type),
        _ => None,
    }
}

fn go_get_name(node: Node, _kind: SymbolKind, source: &[u8]) -> Option<String> {
    if node.kind() == "type_declaration" {
        let spec = first_child_of_type(node, "type_spec")?;
        return field_name(spec, "name", source);
    }
    field_name(node, "name", source)
}

fn go_resolve_kind(node: Node, kind: SymbolKind) -> SymbolKind {
    if node.kind() == "type_declaration" {
        if let Some(spec) = first_child_of_type(node, "type_spec") {
            if let Some(value) = spec.child_by_field_name("type") {
                match value.kind() {
                    "struct_type" => return SymbolKind::Struct,
                    "interface_type" => return SymbolKind::Interface,
                    _ => {}
                }
            }
        }
    }
    kind
}

fn go_get_signature(node: Node, source: &[u8]) -> Option<String> {
    let name = field_name(node, "name", source)?;
    if node.kind() == "method_declaration" {
        let receiver = node
            .child_by_field_name("receiver")
            .map(|n| node_text(n, source).to_string())
            .unwrap_or_default();
        let params = node
            .child_by_field_name("parameters")
            .map(|n| node_text(n, source).to_string())
            .unwrap_or_else(|| "()".to_string());
        let result = node
            .child_by_field_name("result")
            .map(|n| format!(" {}", node_text(n, source)));
        return Some(format!(
            "{} {}{}{}",
            receiver,
            name,
            params,
            result.unwrap_or_default()
        ));
    }
    build_signature(&name, node, source)
}

static GO_SPEC: LanguageSpec = LanguageSpec {
    node_kind: go_node_kind,
    get_name: go_get_name,
    get_signature: go_get_signature,
    resolve_kind: go_resolve_kind,
    is_container: |_| false,
};

// ─── Java ──────────────────────────────────────────────────────────────

fn java_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "class_declaration" | "record_declaration" => Some(SymbolKind::Class),
        "interface_declaration" | "annotation_type_declaration" => Some(SymbolKind::Interface),
        "enum_declaration" => Some(SymbolKind::Enum),
        "method_declaration" | "constructor_declaration" => Some(SymbolKind::Method),
        _ => None,
    }
}

fn java_get_signature(node: Node, source: &[u8]) -> Option<String> {
    let name = field_name(node, "name", source)?;
    build_signature(&name, node, source)
}

fn java_is_container(node_type: &str) -> bool {
    matches!(
        node_type,
        "class_declaration"
            | "interface_declaration"
            | "enum_declaration"
            | "record_declaration"
    )
}

static JAVA_SPEC: LanguageSpec = LanguageSpec {
    node_kind: java_node_kind,
    get_name: |node, _, source| field_name(node, "name", source),
    get_signature: java_get_signature,
    resolve_kind: |_, kind| kind,
    is_container: java_is_container,
};

// ─── Kotlin ────────────────────────────────────────────────────────────

fn kotlin_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_declaration" => Some(SymbolKind::Function),
        "class_declaration" | "object_declaration" => Some(SymbolKind::Class),
        "interface_declaration" => Some(SymbolKind::Interface),
        "property_declaration" => Some(SymbolKind::Property),
        _ => None,
    }
}

fn kotlin_get_name(node: Node, _kind: SymbolKind, source: &[u8]) -> Option<String> {
    first_child_of_type(node, "simple_identifier")
        .map(|n| node_text(n, source).to_string())
        .or_else(|| field_name(node, "name", source))
}

fn kotlin_is_container(node_type: &str) -> bool {
    matches!(
        node_type,
        "class_declaration" | "object_declaration" | "interface_declaration"
    )
}

static KOTLIN_SPEC: LanguageSpec = LanguageSpec {
    node_kind: kotlin_node_kind,
    get_name: kotlin_get_name,
    get_signature: |node, source| extract_signature_line(node, source),
    resolve_kind: |_, kind| kind,
    is_container: kotlin_is_container,
};

// ─── Ruby ──────────────────────────────────────────────────────────────

fn ruby_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "method" | "singleton_method" => Some(SymbolKind::Method),
        "class" => Some(SymbolKind::Class),
        "module" => Some(SymbolKind::Module),
        _ => None,
    }
}

static RUBY_SPEC: LanguageSpec = LanguageSpec {
    node_kind: ruby_node_kind,
    get_name: |node, _, source| field_name(node, "name", source),
    get_signature: |node, source| extract_signature_line(node, source),
    resolve_kind: |_, kind| kind,
    is_container: |t| matches!(t, "class" | "module"),
};

// ─── Swift ─────────────────────────────────────────────────────────────

fn swift_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_declaration" => Some(SymbolKind::Function),
        "class_declaration" => Some(SymbolKind::Class),
        "struct_declaration" => Some(SymbolKind::Struct),
        "enum_declaration" => Some(SymbolKind::Enum),
        "protocol_declaration" => Some(SymbolKind::Interface),
        "extension_declaration" => Some(SymbolKind::Impl),
        _ => None,
    }
}

fn swift_is_container(node_type: &str) -> bool {
    matches!(
        node_type,
        "class_declaration"
            | "struct_declaration"
            | "enum_declaration"
            | "protocol_declaration"
            | "extension_declaration"
    )
}

static SWIFT_SPEC: LanguageSpec = LanguageSpec {
    node_kind: swift_node_kind,
    get_name: |node, _, source| field_name(node, "name", source),
    get_signature: |node, source| extract_signature_line(node, source),
    resolve_kind: |_, kind| kind,
    is_container: swift_is_container,
};

// ─── C# ────────────────────────────────────────────────────────────────

fn csharp_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "class_declaration" | "record_declaration" => Some(SymbolKind::Class),
        "interface_declaration" => Some(SymbolKind::Interface),
        "struct_declaration" => Some(SymbolKind::Struct),
        "enum_declaration" => Some(SymbolKind::Enum),
        "method_declaration" | "constructor_declaration" => Some(SymbolKind::Method),
        "property_declaration" => Some(SymbolKind::Property),
        "namespace_declaration" => Some(SymbolKind::Module),
        _ => None,
    }
}

fn csharp_is_container(node_type: &str) -> bool {
    matches!(
        node_type,
        "class_declaration"
            | "interface_declaration"
            | "struct_declaration"
            | "namespace_declaration"
    )
}

static CSHARP_SPEC: LanguageSpec = LanguageSpec {
    node_kind: csharp_node_kind,
    get_name: |node, _, source| field_name(node, "name", source),
    get_signature: |node, source| {
        let name = field_name(node, "name", source)?;
        build_signature(&name, node, source)
    },
    resolve_kind: |_, kind| kind,
    is_container: csharp_is_container,
};

// ─── Lua ───────────────────────────────────────────────────────────────

fn lua_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_declaration" | "local_function_declaration" => Some(SymbolKind::Function),
        _ => None,
    }
}

static LUA_SPEC: LanguageSpec = LanguageSpec {
    node_kind: lua_node_kind,
    get_name: |node, _, source| field_name(node, "name", source),
    get_signature: |node, source| extract_signature_line(node, source),
    resolve_kind: |_, kind| kind,
    is_container: |_| false,
};

// ─── Bash ──────────────────────────────────────────────────────────────

fn bash_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_definition" => Some(SymbolKind::Function),
        _ => None,
    }
}

static BASH_SPEC: LanguageSpec = LanguageSpec {
    node_kind: bash_node_kind,
    get_name: |node, _, source| field_name(node, "name", source),
    get_signature: |_, _| None,
    resolve_kind: |_, kind| kind,
    is_container: |_| false,
};

// ─── Scala ─────────────────────────────────────────────────────────────

fn scala_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_definition" => Some(SymbolKind::Function),
        "val_definition" | "var_definition" => Some(SymbolKind::Variable),
        "class_definition" | "object_definition" => Some(SymbolKind::Class),
        "trait_definition" => Some(SymbolKind::Trait),
        _ => None,
    }
}

fn scala_is_container(node_type: &str) -> bool {
    matches!(
        node_type,
        "class_definition" | "object_definition" | "trait_definition"
    )
}

static SCALA_SPEC: LanguageSpec = LanguageSpec {
    node_kind: scala_node_kind,
    get_name: |node, _, source| field_name(node, "name", source),
    get_signature: |node, source| extract_signature_line(node, source),
    resolve_kind: |_, kind| kind,
    is_container: scala_is_container,
};

// ─── PHP ───────────────────────────────────────────────────────────────

fn php_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_definition" => Some(SymbolKind::Function),
        "method_declaration" => Some(SymbolKind::Method),
        "class_declaration" => Some(SymbolKind::Class),
        "interface_declaration" => Some(SymbolKind::Interface),
        "trait_declaration" => Some(SymbolKind::Trait),
        "enum_declaration" => Some(SymbolKind::Enum),
        "namespace_definition" => Some(SymbolKind::Module),
        _ => None,
    }
}

fn php_is_container(node_type: &str) -> bool {
    matches!(
        node_type,
        "class_declaration"
            | "interface_declaration"
            | "trait_declaration"
            | "enum_declaration"
    )
}

static PHP_SPEC: LanguageSpec = LanguageSpec {
    node_kind: php_node_kind,
    get_name: |node, _, source| field_name(node, "name", source),
    get_signature: |node, source| {
        let name = field_name(node, "name", source)?;
        build_signature(&name, node, source)
    },
    resolve_kind: |_, kind| kind,
    is_container: php_is_container,
};

// ─── Zig ───────────────────────────────────────────────────────────────

fn zig_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "FnProto" => Some(SymbolKind::Function),
        "TestDecl" => Some(SymbolKind::Function),
        _ => None,
    }
}

fn zig_get_name(node: Node, _kind: SymbolKind, source: &[u8]) -> Option<String> {
    field_name(node, "name", source)
        .or_else(|| first_child_of_type(node, "IDENTIFIER").map(|n| node_text(n, source).to_string()))
}

static ZIG_SPEC: LanguageSpec = LanguageSpec {
    node_kind: zig_node_kind,
    get_name: zig_get_name,
    get_signature: |node, source| extract_signature_line(node, source),
    resolve_kind: |_, kind| kind,
    is_container: |_| false,
};

// ─── Elixir ────────────────────────────────────────────────────────────

fn elixir_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "call" => Some(SymbolKind::Function),
        _ => None,
    }
}

fn elixir_get_name(node: Node, _kind: SymbolKind, source: &[u8]) -> Option<String> {
    let target = first_child_of_type(node, "identifier")?;
    let name = node_text(target, source);
    if !["def", "defp", "defmodule", "defprotocol", "defimpl", "defmacro"].contains(&name) {
        return None;
    }
    let args = node.child_by_field_name("arguments")?;
    let first_arg = args.child(0)?;
    if name == "defmodule" {
        return Some(node_text(first_arg, source).to_string());
    }
    if first_arg.kind() == "call" {
        return first_child_of_type(first_arg, "identifier")
            .map(|n| node_text(n, source).to_string());
    }
    if first_arg.kind() == "identifier" {
        return Some(node_text(first_arg, source).to_string());
    }
    None
}

static ELIXIR_SPEC: LanguageSpec = LanguageSpec {
    node_kind: elixir_node_kind,
    get_name: elixir_get_name,
    get_signature: |node, source| extract_signature_line(node, source),
    resolve_kind: |_, kind| kind,
    is_container: |_| false,
};

// ─── Dart ──────────────────────────────────────────────────────────────

fn dart_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_signature" => Some(SymbolKind::Function),
        "method_signature" => Some(SymbolKind::Method),
        "class_definition" => Some(SymbolKind::Class),
        "enum_declaration" => Some(SymbolKind::Enum),
        "mixin_declaration" => Some(SymbolKind::Class),
        "extension_declaration" => Some(SymbolKind::Impl),
        _ => None,
    }
}

fn dart_is_container(node_type: &str) -> bool {
    matches!(
        node_type,
        "class_definition" | "enum_declaration" | "mixin_declaration"
    )
}

static DART_SPEC: LanguageSpec = LanguageSpec {
    node_kind: dart_node_kind,
    get_name: |node, _, source| field_name(node, "name", source),
    get_signature: |node, source| extract_signature_line(node, source),
    resolve_kind: |_, kind| kind,
    is_container: dart_is_container,
};

// ─── OCaml ─────────────────────────────────────────────────────────────

fn ocaml_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "let_binding" => Some(SymbolKind::Function),
        "type_binding" => Some(SymbolKind::Type),
        "module_binding" => Some(SymbolKind::Module),
        "class_binding" => Some(SymbolKind::Class),
        "method_definition" => Some(SymbolKind::Method),
        _ => None,
    }
}

fn ocaml_get_name(node: Node, _kind: SymbolKind, source: &[u8]) -> Option<String> {
    node.child_by_field_name("pattern")
        .map(|n| node_text(n, source).to_string())
        .or_else(|| field_name(node, "name", source))
}

fn ocaml_is_container(node_type: &str) -> bool {
    matches!(node_type, "module_binding" | "class_binding")
}

static OCAML_SPEC: LanguageSpec = LanguageSpec {
    node_kind: ocaml_node_kind,
    get_name: ocaml_get_name,
    get_signature: |node, source| extract_signature_line(node, source),
    resolve_kind: |_, kind| kind,
    is_container: ocaml_is_container,
};

// ─── YAML ──────────────────────────────────────────────────────────────

fn yaml_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "block_mapping_pair" => Some(SymbolKind::Property),
        _ => None,
    }
}

static YAML_SPEC: LanguageSpec = LanguageSpec {
    node_kind: yaml_node_kind,
    get_name: |node, _, source| {
        node.child_by_field_name("key")
            .map(|n| node_text(n, source).to_string())
    },
    get_signature: |_, _| None,
    resolve_kind: |_, kind| kind,
    is_container: |t| t == "block_mapping_pair",
};

// ─── TOML ──────────────────────────────────────────────────────────────

fn toml_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "table" | "table_array_element" => Some(SymbolKind::Block),
        "pair" => Some(SymbolKind::Property),
        _ => None,
    }
}

fn toml_get_name(node: Node, _kind: SymbolKind, source: &[u8]) -> Option<String> {
    let nk = node.kind();
    if nk == "table" || nk == "table_array_element" {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "bare_key" | "dotted_key" | "quoted_key" => {
                        return Some(node_text(child, source).to_string());
                    }
                    _ => {}
                }
            }
        }
        return None;
    }
    // pair
    first_child_of_type(node, "bare_key")
        .or_else(|| first_child_of_type(node, "quoted_key"))
        .map(|n| node_text(n, source).to_string())
}

static TOML_SPEC: LanguageSpec = LanguageSpec {
    node_kind: toml_node_kind,
    get_name: toml_get_name,
    get_signature: |_, _| None,
    resolve_kind: |_, kind| kind,
    is_container: |t| matches!(t, "table" | "table_array_element"),
};

// ─── Protobuf ──────────────────────────────────────────────────────────

fn proto_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "message" => Some(SymbolKind::Struct),
        "enum" => Some(SymbolKind::Enum),
        "service" => Some(SymbolKind::Interface),
        "rpc" => Some(SymbolKind::Method),
        "field" | "map_field" => Some(SymbolKind::Property),
        "oneof" => Some(SymbolKind::Property),
        "package" => Some(SymbolKind::Module),
        _ => None,
    }
}

fn proto_get_name(node: Node, _kind: SymbolKind, source: &[u8]) -> Option<String> {
    let nk = node.kind();
    match nk {
        "message" => first_child_of_type(node, "message_name").map(|n| node_text(n, source).to_string()),
        "enum" => first_child_of_type(node, "enum_name").map(|n| node_text(n, source).to_string()),
        "service" => first_child_of_type(node, "service_name").map(|n| node_text(n, source).to_string()),
        "rpc" => first_child_of_type(node, "rpc_name").map(|n| node_text(n, source).to_string()),
        "package" => first_child_of_type(node, "full_ident").map(|n| node_text(n, source).to_string()),
        "field" | "map_field" | "oneof" => {
            // Fields have identifier children — find the field name
            first_child_of_type(node, "identifier").map(|n| node_text(n, source).to_string())
        }
        _ => None,
    }
}

fn proto_get_signature(node: Node, source: &[u8]) -> Option<String> {
    let nk = node.kind();
    match nk {
        "rpc" => {
            // rpc Name(Request) returns (Response)
            Some(extract_signature_line(node, source).unwrap_or_default())
        }
        "field" => {
            // type name = number;
            let type_node = first_child_of_type(node, "type").map(|n| node_text(n, source)).unwrap_or("");
            let name = first_child_of_type(node, "identifier").map(|n| node_text(n, source)).unwrap_or("");
            let num = first_child_of_type(node, "field_number").map(|n| node_text(n, source)).unwrap_or("");
            Some(format!("{} {} = {}", type_node, name, num))
        }
        "map_field" => {
            let key = first_child_of_type(node, "key_type").map(|n| node_text(n, source)).unwrap_or("");
            let val = first_child_of_type(node, "type").map(|n| node_text(n, source)).unwrap_or("");
            let name = first_child_of_type(node, "identifier").map(|n| node_text(n, source)).unwrap_or("");
            let num = first_child_of_type(node, "field_number").map(|n| node_text(n, source)).unwrap_or("");
            Some(format!("map<{}, {}> {} = {}", key, val, name, num))
        }
        _ => None,
    }
}

fn proto_is_container(node_type: &str) -> bool {
    matches!(node_type, "message" | "enum" | "service")
}

static PROTO_SPEC: LanguageSpec = LanguageSpec {
    node_kind: proto_node_kind,
    get_name: proto_get_name,
    get_signature: proto_get_signature,
    resolve_kind: |_, kind| kind,
    is_container: proto_is_container,
};

// ─── HCL / Terraform ──────────────────────────────────────────────────

fn hcl_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "block" => Some(SymbolKind::Block),
        _ => None,
    }
}

fn hcl_get_name(node: Node, _kind: SymbolKind, source: &[u8]) -> Option<String> {
    let mut parts = Vec::new();
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            match child.kind() {
                "identifier" => parts.push(node_text(child, source).to_string()),
                "string_lit" => {
                    parts.push(node_text(child, source).trim_matches('"').to_string())
                }
                "body" | "{" => break,
                _ => {}
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

/// Map generic "block" kind to more specific HCL/Terraform kinds.
fn map_hcl_kind(kind: SymbolKind, name: &str) -> SymbolKind {
    if kind != SymbolKind::Block {
        return kind;
    }
    let block_type = name.split_whitespace().next().unwrap_or("");
    match block_type {
        "resource" => SymbolKind::Resource,
        "data" => SymbolKind::Data,
        "provider" => SymbolKind::Provider,
        "output" => SymbolKind::Output,
        "variable" => SymbolKind::Variable,
        "locals" => SymbolKind::Locals,
        "module" => SymbolKind::Module,
        _ => SymbolKind::Block,
    }
}

static HCL_SPEC: LanguageSpec = LanguageSpec {
    node_kind: hcl_node_kind,
    get_name: hcl_get_name,
    get_signature: |_, _| None,
    resolve_kind: |_, kind| kind,
    is_container: |t| t == "block",
};

static TERRAFORM_SPEC: LanguageSpec = LanguageSpec {
    node_kind: hcl_node_kind,
    get_name: hcl_get_name,
    get_signature: |_, _| None,
    resolve_kind: |_, kind| kind,
    is_container: |t| t == "block",
};

// ─── Markdown ──────────────────────────────────────────────────────────
//
// tree-sitter-md nests a `section` node per heading, with deeper headings as
// child sections. Text before the first heading sits in a section with no
// heading; that section is not a symbol, and the walk looks through it.

fn markdown_node_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "section" => Some(SymbolKind::Section),
        _ => None,
    }
}

fn markdown_get_name(node: Node, _kind: SymbolKind, source: &[u8]) -> Option<String> {
    let heading = (0..node.named_child_count())
        .filter_map(|i| node.named_child(i))
        .find(|c| matches!(c.kind(), "atx_heading" | "setext_heading"))?;
    let raw = heading
        .child_by_field_name("heading_content")
        .map(|n| node_text(n, source))
        .unwrap_or_else(|| node_text(heading, source).trim_start_matches('#'));
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    // ATX headings may close with a run of '#'.
    let name = collapsed.trim_end_matches('#').trim_end();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

static MARKDOWN_SPEC: LanguageSpec = LanguageSpec {
    node_kind: markdown_node_kind,
    get_name: markdown_get_name,
    get_signature: |_, _| None,
    resolve_kind: |_, kind| kind,
    is_container: |t| t == "section",
};

// ─── Symbol extraction ─────────────────────────────────────────────────

/// Extract symbols from a parsed tree.
pub fn extract_symbols(
    tree: &Tree,
    grammar: &str,
    source: &str,
    options: &ExtractOptions,
) -> Vec<SymbolInfo> {
    let spec = match get_spec(grammar) {
        Some(s) => s,
        None => return Vec::new(),
    };

    let source_bytes = source.as_bytes();
    let mut symbols = Vec::new();
    walk_node(
        tree.root_node(),
        spec,
        grammar,
        source_bytes,
        &mut symbols,
        options,
        0,
    );

    // Apply kind filter
    if let Some(ref kind_str) = options.kind {
        symbols = filter_by_kind(&symbols, kind_str);
    }

    // Apply name pattern filter
    if let Some(ref pattern) = options.name_pattern {
        // Strip inline flags like (?i)
        let cleaned = pattern.replace("(?i)", "").replace("(?m)", "").replace("(?s)", "");
        if let Ok(re) = Regex::new(&format!("(?i){}", cleaned)) {
            symbols = filter_by_name(&symbols, &re);
        }
    }

    symbols
}

fn walk_node(
    node: Node,
    spec: &LanguageSpec,
    grammar: &str,
    source: &[u8],
    results: &mut Vec<SymbolInfo>,
    options: &ExtractOptions,
    depth: usize,
) {
    if let Some(max) = options.max_depth {
        if depth > max {
            return;
        }
    }

    for i in 0..node.child_count() {
        let child = match node.child(i) {
            Some(c) => c,
            None => continue,
        };
        if !child.is_named() {
            continue;
        }

        if let Some(raw_kind) = (spec.node_kind)(child.kind()) {
            let kind = (spec.resolve_kind)(child, raw_kind);
            if let Some(name) = (spec.get_name)(child, kind, source) {
                let name = truncate_display(&name, 80);

                // Map HCL block kinds
                let final_kind = if grammar == "hcl" || grammar == "terraform" {
                    map_hcl_kind(kind, &name)
                } else {
                    kind
                };

                let (start_line, end_line) = line_range(child);
                let mut symbol = SymbolInfo {
                    name,
                    kind: final_kind,
                    start_line,
                    end_line,
                    signature: None,
                    children: Vec::new(),
                };

                if options.signatures {
                    symbol.signature =
                        (spec.get_signature)(child, source).map(|s| truncate_display(&s, 120));
                }

                // Extract children from container types
                if !options.top_level_only && (spec.is_container)(child.kind()) {
                    walk_node(
                        child,
                        spec,
                        grammar,
                        source,
                        &mut symbol.children,
                        options,
                        depth + 1,
                    );
                } else if !options.top_level_only && child.kind() == "decorated_definition" {
                    if let Some(inner_def) = child.child_by_field_name("definition") {
                        if (spec.is_container)(inner_def.kind()) {
                            walk_node(
                                inner_def,
                                spec,
                                grammar,
                                source,
                                &mut symbol.children,
                                options,
                                depth + 1,
                            );
                        }
                    }
                }

                results.push(symbol);
                continue;
            }
        }

        // Handle export statements
        if child.kind() == "export_statement" || child.kind() == "export_declaration" {
            if let Some(decl) = child.child_by_field_name("declaration") {
                if let Some(kind) = (spec.node_kind)(decl.kind()) {
                    if let Some(name) = (spec.get_name)(decl, kind, source) {
                        let mut symbol = SymbolInfo {
                            name,
                            kind,
                            start_line: line_range(child).0,
                            end_line: line_range(child).1,
                            signature: None,
                            children: Vec::new(),
                        };
                        if options.signatures {
                            symbol.signature = (spec.get_signature)(decl, source)
                                .map(|s| truncate_display(&s, 120));
                        }
                        if !options.top_level_only && (spec.is_container)(decl.kind()) {
                            walk_node(
                                decl,
                                spec,
                                grammar,
                                source,
                                &mut symbol.children,
                                options,
                                depth + 1,
                            );
                        }
                        results.push(symbol);
                        continue;
                    }
                }
            }
        }

        // Handle Python decorated_definition
        if child.kind() == "decorated_definition" {
            if let Some(def) = child.child_by_field_name("definition") {
                if let Some(def_kind) = (spec.node_kind)(def.kind()) {
                    if let Some(name) = field_name(def, "name", source) {
                        let mut symbol = SymbolInfo {
                            name,
                            kind: def_kind,
                            start_line: line_range(child).0,
                            end_line: line_range(child).1,
                            signature: None,
                            children: Vec::new(),
                        };
                        if options.signatures {
                            symbol.signature = (spec.get_signature)(child, source)
                                .map(|s| truncate_display(&s, 120));
                        }
                        if !options.top_level_only && (spec.is_container)(def.kind()) {
                            walk_node(
                                def,
                                spec,
                                grammar,
                                source,
                                &mut symbol.children,
                                options,
                                depth + 1,
                            );
                        }
                        results.push(symbol);
                        continue;
                    }
                }
            }
        }

        // Recurse into non-symbol nodes. A Markdown section with no heading is
        // not a symbol but still holds the headed sections after it.
        if (spec.node_kind)(child.kind()).is_none() || grammar == "markdown" {
            walk_node(child, spec, grammar, source, results, options, depth);
        }
    }
}

// ─── Filtering ─────────────────────────────────────────────────────────

fn filter_by_kind(symbols: &[SymbolInfo], kind: &str) -> Vec<SymbolInfo> {
    let mut result = Vec::new();
    for sym in symbols {
        if sym.kind.as_str() == kind {
            result.push(sym.clone());
        } else if !sym.children.is_empty() {
            let filtered = filter_by_kind(&sym.children, kind);
            result.extend(filtered);
        }
    }
    result
}

fn filter_by_name(symbols: &[SymbolInfo], re: &Regex) -> Vec<SymbolInfo> {
    let mut result = Vec::new();
    for sym in symbols {
        if re.is_match(&sym.name) {
            result.push(sym.clone());
        } else if !sym.children.is_empty() {
            let filtered = filter_by_name(&sym.children, re);
            if !filtered.is_empty() {
                result.push(SymbolInfo {
                    children: filtered,
                    ..sym.clone()
                });
            }
        }
    }
    result
}

// ─── Formatting ────────────────────────────────────────────────────────

/// Format symbols as a flat list.
pub fn format_symbols(symbols: &[SymbolInfo], show_signatures: bool) -> String {
    let mut lines = Vec::new();
    format_symbols_flat(symbols, &mut lines, 0, show_signatures);
    lines.join("\n")
}

fn format_symbols_flat(
    symbols: &[SymbolInfo],
    lines: &mut Vec<String>,
    indent: usize,
    show_signatures: bool,
) {
    for sym in symbols {
        let pad = "  ".repeat(indent);
        let kind_label = format!("{:9}", sym.kind.icon());
        let line_range = if sym.start_line == sym.end_line {
            format!("L{}", sym.start_line)
        } else {
            format!("L{}-L{}", sym.start_line, sym.end_line)
        };
        let display = if show_signatures {
            sym.signature.as_deref().unwrap_or(&sym.name)
        } else {
            &sym.name
        };
        let padded_display = format!("{:<width$}", display, width = 50 - indent * 2);
        lines.push(format!("{}{} {} {}", pad, kind_label, padded_display, line_range));
        if !sym.children.is_empty() {
            format_symbols_flat(&sym.children, lines, indent + 1, show_signatures);
        }
    }
}

/// Format symbols as a tree outline.
pub fn format_outline(symbols: &[SymbolInfo], show_signatures: bool) -> String {
    let mut lines = Vec::new();
    format_outline_tree(symbols, &mut lines, "", true, show_signatures);
    lines.join("\n")
}

fn format_outline_tree(
    symbols: &[SymbolInfo],
    lines: &mut Vec<String>,
    prefix: &str,
    is_root: bool,
    show_signatures: bool,
) {
    for (i, sym) in symbols.iter().enumerate() {
        let is_last = i == symbols.len() - 1;
        let connector = if is_root {
            ""
        } else if is_last {
            "└── "
        } else {
            "├── "
        };
        let child_prefix = if is_root {
            String::new()
        } else if is_last {
            format!("{}    ", prefix)
        } else {
            format!("{}│   ", prefix)
        };
        let kind_label = sym.kind.icon();
        let line_range = if sym.start_line == sym.end_line {
            format!("L{}", sym.start_line)
        } else {
            format!("L{}-L{}", sym.start_line, sym.end_line)
        };
        let display = if show_signatures {
            sym.signature.as_deref().unwrap_or(&sym.name)
        } else {
            &sym.name
        };
        lines.push(format!(
            "{}{}{} ({} {})",
            prefix, connector, display, kind_label, line_range
        ));
        if !sym.children.is_empty() {
            format_outline_tree(&sym.children, lines, &child_prefix, false, show_signatures);
        }
    }
}

/// Count total symbols including nested children.
pub fn count_symbols(symbols: &[SymbolInfo]) -> usize {
    let mut count = 0;
    for sym in symbols {
        count += 1;
        count += count_symbols(&sym.children);
    }
    count
}

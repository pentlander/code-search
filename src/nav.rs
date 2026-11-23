//! Navigation inside one file:
//!
//! - `scope`: the breadcrumb of symbols enclosing a line.
//! - `inspect`: control flow, calls, local bindings and nested symbols inside one symbol.
//! - `find_identifiers`: identifier occurrences, each classified by syntactic role
//!   and labelled with its enclosing symbol.
//!
//! A "symbol" here is exactly what the outline shows: everything goes through
//! `symbols::node_symbol`. Role and category tables are syntactic and shared
//! across grammars; they do not resolve names.

use tree_sitter::{Node, Point, Tree};

use crate::symbols::{
    elixir_defines_function, format_outline, line_range, node_signature, node_symbol,
    truncate_display, wrapped_declaration, NodeSymbol, SymbolInfo,
};

// ─── Shared helpers ────────────────────────────────────────────────────

/// Symbols enclosing `node` (inclusive), outermost first.
fn enclosing_symbols<'a>(node: Node<'a>, grammar: &str, source: &[u8]) -> Vec<NodeSymbol<'a>> {
    let mut chain = Vec::new();
    let mut came_from = None;
    let mut current = Some(node);
    while let Some(n) = current {
        if let Some(sym) = node_symbol(n, grammar, source) {
            chain.push(sym);
        } else if let Some(decl) = wrapped_declaration(n).filter(|&d| Some(d) != came_from) {
            // On `export` or a decorator: inside the wrapped symbol's range.
            chain.extend(node_symbol(decl, grammar, source));
        }
        came_from = Some(n);
        current = n.parent();
    }
    chain.reverse();
    chain
}

/// Qualified name segments for a chain. Go methods are top-level syntactically,
/// so their receiver type is added as the qualifier.
fn qualified_names(chain: &[NodeSymbol], grammar: &str, source: &[u8]) -> Vec<String> {
    let mut names = Vec::new();
    for sym in chain {
        if grammar == "go" && sym.node.kind() == "method_declaration" {
            if let Some(receiver) = go_receiver_type(sym.node, source) {
                names.push(receiver);
            }
        }
        names.push(sym.name.clone());
    }
    names
}

fn go_receiver_type(method: Node, source: &[u8]) -> Option<String> {
    let receiver = method.child_by_field_name("receiver")?;
    let mut cursor = receiver.walk();
    let mut stack = vec![receiver];
    while let Some(n) = stack.pop() {
        if n.kind() == "type_identifier" {
            return n.utf8_text(source).ok().map(str::to_string);
        }
        let children: Vec<Node> = n.named_children(&mut cursor).collect();
        stack.extend(children.into_iter().rev());
    }
    None
}

/// The field name under which `node` hangs from its parent.
fn field_in_parent(node: Node) -> Option<&'static str> {
    let parent = node.parent()?;
    for i in 0..parent.child_count() {
        if parent.child(i) == Some(node) {
            return parent.field_name_for_child(i as u32);
        }
    }
    None
}

fn first_named_child_is(parent: Node, node: Node) -> bool {
    parent.named_child(0) == Some(node)
}

/// First line of a node's text, whitespace collapsed, shortened for display.
fn one_line(node: Node, source: &[u8], max: usize) -> String {
    let text = node.utf8_text(source).unwrap_or("");
    let first = text.lines().next().unwrap_or("");
    let collapsed = first.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_display(&collapsed, max)
}

fn range_label(start: usize, end: usize) -> String {
    if start == end {
        format!("L{}", start)
    } else {
        format!("L{}-L{}", start, end)
    }
}

fn location_label(line: usize, column: Option<usize>) -> String {
    match column {
        Some(c) => format!("L{}:{}", line, c),
        None => format!("L{}", line),
    }
}

/// Point for a 1-based line and optional 1-based column. Without a column, the
/// line's first non-blank character is used, so leading indentation never
/// lands in the enclosing block instead of the statement on the line.
fn point_for(source: &str, line: usize, column: Option<usize>) -> Result<Point, String> {
    let total = source.lines().count();
    if line == 0 || line > total.max(1) {
        return Err(format!(
            "Line {} is outside the file ({} lines)",
            line, total
        ));
    }
    let text = source.lines().nth(line - 1).unwrap_or("");
    let col = match column {
        Some(0) => return Err("Columns are 1-based".to_string()),
        Some(c) if c - 1 > text.len() => {
            return Err(format!(
                "Column {} is past the end of line {} ({} bytes)",
                c,
                line,
                text.len()
            ))
        }
        Some(c) => c - 1,
        None => text.len() - text.trim_start().len(),
    };
    Ok(Point::new(line - 1, col))
}

fn node_at<'a>(tree: &'a Tree, point: Point) -> Node<'a> {
    let root = tree.root_node();
    root.descendant_for_point_range(point, point)
        .unwrap_or(root)
}

/// The node whose enclosing symbols answer for `point`. With an explicit
/// column that is the node under it. For a whole line, the first non-blank
/// token is often a wrapper that holds the symbol as a child (`export`, a
/// YAML `-`, a Swift attribute), so prefer the first symbol starting on the
/// line inside the widest node that starts at that token.
fn anchor_at<'a>(
    tree: &'a Tree,
    grammar: &str,
    source: &[u8],
    point: Point,
    whole_line: bool,
) -> Node<'a> {
    let node = node_at(tree, point);
    if !whole_line {
        return node;
    }
    let mut widest = node;
    while let Some(parent) = widest.parent() {
        if parent.start_position() != point {
            break;
        }
        widest = parent;
    }
    first_symbol_on_row(widest, grammar, source, point.row).unwrap_or(node)
}

fn first_symbol_on_row<'a>(
    node: Node<'a>,
    grammar: &str,
    source: &[u8],
    row: usize,
) -> Option<Node<'a>> {
    if node.start_position().row > row || node.end_position().row < row {
        return None;
    }
    if node.start_position().row == row && node_symbol(node, grammar, source).is_some() {
        return Some(node);
    }
    let mut cursor = node.walk();
    let children: Vec<Node<'a>> = node.children(&mut cursor).collect();
    children
        .into_iter()
        .find_map(|child| first_symbol_on_row(child, grammar, source, row))
}

fn symbol_info(
    sym: &NodeSymbol,
    display: String,
    grammar: &str,
    source: &[u8],
    signatures: bool,
) -> SymbolInfo {
    let (start_line, end_line) = line_range(sym.range_node);
    SymbolInfo {
        name: display,
        kind: sym.kind,
        start_line,
        end_line,
        signature: if signatures {
            node_signature(sym.node, grammar, source)
        } else {
            None
        },
        children: Vec::new(),
    }
}

// ─── scope ─────────────────────────────────────────────────────────────

/// Breadcrumb of the symbols enclosing a line, outermost first:
///
/// ```text
/// L4 scope:
/// Box (class L2-L7)
/// └── alpha (method L3-L5)
/// ```
///
/// A line outside every symbol prints `L1: file scope`.
pub fn scope(
    tree: &Tree,
    grammar: &str,
    source: &str,
    line: usize,
    column: Option<usize>,
    signatures: bool,
) -> Result<String, String> {
    let point = point_for(source, line, column)?;
    let at = location_label(line, column);
    let bytes = source.as_bytes();
    let anchor = anchor_at(tree, grammar, bytes, point, column.is_none());
    let chain = enclosing_symbols(anchor, grammar, bytes);
    if chain.is_empty() {
        return Ok(format!("{}: file scope", at));
    }
    let mut nested: Option<SymbolInfo> = None;
    for sym in chain.iter().rev() {
        let mut info = symbol_info(sym, sym.name.clone(), grammar, bytes, signatures);
        if let Some(child) = nested.take() {
            info.children.push(child);
        }
        nested = Some(info);
    }
    let tree_text = format_outline(&[nested.expect("chain is non-empty")], signatures);
    Ok(format!("{} scope:\n{}", at, tree_text))
}

// ─── inspect ───────────────────────────────────────────────────────────

/// Which inspect categories to report.
#[derive(Debug, Clone, Copy)]
pub struct Categories {
    pub control_flow: bool,
    pub calls: bool,
    pub locals: bool,
    pub nested_symbols: bool,
}

impl Categories {
    pub fn all() -> Self {
        Self {
            control_flow: true,
            calls: true,
            locals: true,
            nested_symbols: true,
        }
    }

    /// Parse a comma-separated list such as `calls,locals`.
    pub fn parse(list: &str) -> Result<Self, String> {
        let mut c = Self {
            control_flow: false,
            calls: false,
            locals: false,
            nested_symbols: false,
        };
        for part in list.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match part {
                "control_flow" => c.control_flow = true,
                "calls" => c.calls = true,
                "locals" => c.locals = true,
                "nested_symbols" => c.nested_symbols = true,
                other => {
                    return Err(format!(
                        "Unknown --include category \"{}\" (expected control_flow, calls, locals, nested_symbols)",
                        other
                    ))
                }
            }
        }
        Ok(c)
    }

    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.control_flow {
            parts.push("control flow");
        }
        if self.calls {
            parts.push("calls");
        }
        if self.locals {
            parts.push("locals");
        }
        if self.nested_symbols {
            parts.push("nested symbols");
        }
        parts.join(", ")
    }
}

/// How deep and what to report inside the inspected symbol.
#[derive(Debug, Clone, Copy)]
pub struct InspectOptions {
    /// Maximum nesting depth of reported items; deeper items are counted.
    pub depth: usize,
    pub include: Categories,
    /// Show the target's signature instead of its qualified name.
    pub signatures: bool,
}

pub enum InspectTarget<'q> {
    Line { line: usize, column: Option<usize> },
    Symbol(&'q str),
}

struct Item {
    text: String,
    tag: String,
    start: usize,
    end: usize,
    children: Vec<Item>,
}

/// Control flow, calls, locals and nested symbols inside one symbol. The first
/// row is the target as an outline row; matched nodes nest beneath it:
///
/// ```text
/// Box::alpha (method L3-L5)
/// └── this.beta() [call:call L4]
/// ```
///
/// Nested symbols are listed but not descended into; inspect them directly.
pub fn inspect(
    tree: &Tree,
    grammar: &str,
    source: &str,
    file_label: &str,
    target: InspectTarget,
    options: &InspectOptions,
) -> Result<String, String> {
    let InspectOptions {
        depth,
        include,
        signatures,
    } = *options;
    let bytes = source.as_bytes();
    let (sym, names) = match target {
        InspectTarget::Line { line, column } => {
            let point = point_for(source, line, column)?;
            let anchor = anchor_at(tree, grammar, bytes, point, column.is_none());
            let chain = enclosing_symbols(anchor, grammar, bytes);
            if chain.is_empty() {
                return Err(format!(
                    "No symbol encloses {} of {}; use scope or outline",
                    location_label(line, column),
                    file_label
                ));
            }
            let names = qualified_names(&chain, grammar, bytes);
            (chain.into_iter().last().expect("non-empty"), names)
        }
        InspectTarget::Symbol(query) => resolve_symbol(tree, grammar, bytes, query, file_label)?,
    };

    let display = if signatures {
        node_signature(sym.node, grammar, bytes).unwrap_or_else(|| names.join("::"))
    } else {
        names.join("::")
    };
    let (start, end) = line_range(sym.range_node);
    let mut out = vec![format!(
        "{} ({} {})",
        display,
        sym.kind.icon(),
        range_label(start, end)
    )];

    let mut walk = InspectWalk {
        grammar,
        source: bytes,
        include,
        max_depth: depth,
        hidden: 0,
    };
    let mut items = Vec::new();
    walk.children(sym.node, 1, &mut items);
    render_items(&items, "", &mut out);
    if items.is_empty() && walk.hidden == 0 {
        out.push(format!("(no {} inside)", include.describe()));
    }
    if walk.hidden > 0 {
        out.push(format!(
            "[{} more items deeper than --depth {}; raise --depth or inspect a nested symbol]",
            walk.hidden, depth
        ));
    }
    Ok(out.join("\n"))
}

fn resolve_symbol<'a>(
    tree: &'a Tree,
    grammar: &str,
    source: &[u8],
    query: &str,
    file_label: &str,
) -> Result<(NodeSymbol<'a>, Vec<String>), String> {
    let query = query.trim();
    let mut found = Vec::new();
    let mut cursor = tree.walk();
    'walk: loop {
        let node = cursor.node();
        if node_symbol(node, grammar, source).is_some() {
            let chain = enclosing_symbols(node, grammar, source);
            let names = qualified_names(&chain, grammar, source);
            found.push((chain.into_iter().last().expect("node is a symbol"), names));
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                continue 'walk;
            }
            if !cursor.goto_parent() {
                break 'walk;
            }
        }
    }

    let by = |query: &str, separator: &str| -> Vec<usize> {
        let segments: Vec<&str> = query.split(separator).filter(|s| !s.is_empty()).collect();
        if segments.is_empty() {
            return Vec::new();
        }
        found
            .iter()
            .enumerate()
            .filter(|(_, (_, names))| {
                names.len() >= segments.len()
                    && names[names.len() - segments.len()..]
                        .iter()
                        .zip(&segments)
                        .all(|(name, seg)| segment_matches(name, seg))
            })
            .map(|(i, _)| i)
            .collect()
    };
    // Whole-string suffix match, for names that contain a separator
    // themselves (Elixir `Shop.Cart`).
    let joined = |query: &str| -> Vec<usize> {
        (0..found.len())
            .filter(|&i| {
                ["::", "."].iter().any(|sep| {
                    let full = found[i].1.join(sep);
                    full == query || full.ends_with(&format!("{}{}", sep, query))
                })
            })
            .collect()
    };
    let lookup = |query: &str| {
        let mut matches = by(query, "::");
        if matches.is_empty() && query.contains('.') {
            matches = by(query, ".");
        }
        if matches.is_empty() {
            matches = joined(query);
        }
        matches
    };
    let asked = query;
    // A name that itself contains `::` or `.` (a YAML key like `fe80::/10`)
    // matches whole before the query is split into segments.
    let mut matches: Vec<usize> = (0..found.len())
        .filter(|&i| found[i].1.last().is_some_and(|name| name == query))
        .collect();
    if matches.is_empty() {
        matches = lookup(query);
    }
    let mut query = query.to_string();
    if matches.is_empty() {
        // `nav::inspect` or `crate::nav::inspect`: leading segments may name
        // the file's module rather than a symbol inside it.
        if let Some(rest) = strip_module_prefix(&query, file_label) {
            matches = lookup(&rest);
            query = rest;
        }
    }
    if matches.len() > 1 {
        // A query naming the full path wins over suffix matches.
        let exact: Vec<usize> = matches
            .iter()
            .copied()
            .filter(|&i| found[i].1.join("::") == query || found[i].1.join(".") == query)
            .collect();
        if exact.len() == 1 {
            matches = exact;
        }
    }
    match matches.len() {
        0 => Err(format!("No symbol named \"{}\" in {}", asked, file_label)),
        1 => Ok(found.swap_remove(matches[0])),
        _ => {
            let listed: Vec<String> = matches
                .iter()
                .take(10)
                .map(|&i| {
                    let (sym, names) = &found[i];
                    let (s, e) = line_range(sym.range_node);
                    format!(
                        "  {} ({} {})",
                        names.join("::"),
                        sym.kind.icon(),
                        range_label(s, e)
                    )
                })
                .collect();
            Err(format!(
                "Symbol \"{}\" is ambiguous in {}; qualify it or use --line:\n{}",
                query,
                file_label,
                listed.join("\n")
            ))
        }
    }
}

/// Drops leading `crate`/`self`/`super` and file or directory module names from
/// a qualified query, keeping at least one segment. None when nothing drops.
fn strip_module_prefix(query: &str, file_label: &str) -> Option<String> {
    let separator = if query.contains("::") {
        "::"
    } else if query.contains('.') {
        "."
    } else {
        return None;
    };
    let path = std::path::Path::new(file_label);
    let stem = path.file_stem().and_then(|s| s.to_str());
    let dir = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str());
    let is_module = |seg: &str| {
        matches!(seg, "crate" | "self" | "super") || Some(seg) == stem || Some(seg) == dir
    };
    let segments: Vec<&str> = query.split(separator).collect();
    let skip = segments[..segments.len() - 1]
        .iter()
        .take_while(|s| is_module(s))
        .count();
    (skip > 0).then(|| segments[skip..].join(separator))
}

/// `seg` names `name` exactly, names the type of a Rust `impl Trait for Type`,
/// or names a generic type without its parameters.
fn segment_matches(name: &str, seg: &str) -> bool {
    if name == seg {
        return true;
    }
    if let Some(ty) = name.rsplit(" for ").next() {
        if ty != name && (ty == seg || ty.split('<').next() == Some(seg)) {
            return true;
        }
    }
    name.split('<').next() == Some(seg)
}

struct InspectWalk<'s> {
    grammar: &'s str,
    source: &'s [u8],
    include: Categories,
    max_depth: usize,
    hidden: usize,
}

#[derive(Clone, Copy)]
enum Category {
    Control,
    Call,
    Local,
}

impl InspectWalk<'_> {
    fn children(&mut self, node: Node, depth: usize, out: &mut Vec<Item>) {
        let mut cursor = node.walk();
        let kids: Vec<Node> = node.named_children(&mut cursor).collect();
        for child in kids {
            self.visit(child, depth, out);
        }
    }

    fn visit(&mut self, node: Node, depth: usize, out: &mut Vec<Item>) {
        if let Some(sym) = node_symbol(node, self.grammar, self.source) {
            // A nested symbol is its own scope: list it, never descend.
            if self.include.nested_symbols {
                if depth > self.max_depth {
                    self.hidden += 1;
                } else {
                    let (start, end) = line_range(sym.range_node);
                    out.push(Item {
                        text: sym.name.clone(),
                        tag: format!("symbol:{}", sym.kind.icon()),
                        start,
                        end,
                        children: Vec::new(),
                    });
                }
            }
            return;
        }
        let included = match self.classify(node) {
            Some((Category::Control, item)) if self.include.control_flow => Some(item),
            Some((Category::Call, item)) if self.include.calls => Some(item),
            Some((Category::Local, item)) if self.include.locals => Some(item),
            _ => None,
        };
        match included {
            Some(mut item) => {
                if depth > self.max_depth {
                    self.hidden += 1;
                    self.children(node, depth + 1, &mut Vec::new());
                } else {
                    self.children(node, depth + 1, &mut item.children);
                    out.push(item);
                }
            }
            // Unmatched nodes are transparent.
            None => self.children(node, depth, out),
        }
    }

    fn classify(&self, node: Node) -> Option<(Category, Item)> {
        let (start, end) = line_range(node);
        let item = |text: String, tag: String| Item {
            text,
            tag,
            start,
            end,
            children: Vec::new(),
        };
        if let Some(label) = control_label(node, self.grammar) {
            let header = one_line(node, self.source, 72);
            let header = header
                .trim_end_matches(&['{', ':'][..])
                .trim_end()
                .to_string();
            let text = if header.is_empty() {
                label.to_string()
            } else {
                header
            };
            return Some((Category::Control, item(text, format!("control:{}", label))));
        }
        if let Some(label) = call_label(node.kind(), self.grammar) {
            let callee = callee_text(node, self.source);
            let text = match label {
                "macro" => format!("{}!()", callee),
                "new" => format!("new {}()", callee),
                "command" => one_line(node, self.source, 72),
                _ => format!("{}()", callee),
            };
            return Some((Category::Call, item(text, format!("call:{}", label))));
        }
        if let Some(label) = local_label(node, self.grammar) {
            let names = binding_names(node, self.source, label);
            return Some((Category::Local, item(names, format!("local:{}", label))));
        }
        None
    }
}

fn render_items(items: &[Item], prefix: &str, out: &mut Vec<String>) {
    for (i, item) in items.iter().enumerate() {
        let last = i + 1 == items.len();
        let connector = if last { "└── " } else { "├── " };
        out.push(format!(
            "{}{}{} [{} {}]",
            prefix,
            connector,
            item.text,
            item.tag,
            range_label(item.start, item.end)
        ));
        let child_prefix = format!("{}{}", prefix, if last { "    " } else { "│   " });
        render_items(&item.children, &child_prefix, out);
    }
}

fn control_label(node: Node, grammar: &str) -> Option<&'static str> {
    let label = match node.kind() {
        "if_statement" | "if_expression" | "if_let_expression" | "if" | "unless" => "if",
        "elif_clause" | "else_if_clause" => "elif",
        "else_clause" | "else" => "else",
        "for_statement"
        | "for_in_statement"
        | "for_of_statement"
        | "enhanced_for_statement"
        | "for_expression"
        | "foreach_statement"
        | "for_each_statement"
        | "for_range_loop"
        | "for" => "for",
        "while_statement" | "while_expression" | "while" | "until" => "while",
        "do_statement" | "do_while_statement" | "repeat_statement" => "do",
        "loop_expression" => "loop",
        "switch_statement"
        | "switch_expression"
        | "expression_switch_statement"
        | "type_switch_statement" => "switch",
        "select_statement" => "select",
        "match_expression" | "match_statement" => "match",
        "when_expression" => "when",
        "case_statement" | "case" => "case",
        "try_statement" | "try_block" | "begin" => "try",
        "try_expression" if grammar == "rust" => "?",
        "try_expression" => "try",
        "catch_clause" | "except_clause" | "rescue" => "catch",
        "finally_clause" | "ensure" => "finally",
        "return_statement" | "return_expression" | "return" => "return",
        "throw_statement" | "throw_expression" | "raise_statement" => "throw",
        "break_statement" | "break_expression" | "break" => "break",
        "continue_statement" | "continue_expression" | "next" => "continue",
        "yield_statement" | "yield_expression" | "yield" => "yield",
        "await_expression" | "await" => "await",
        "go_statement" => "go",
        "defer_statement" => "defer",
        "with_statement" => "with",
        "ternary_expression" | "conditional_expression" => "ternary",
        _ => return None,
    };
    Some(label)
}

const CALL_KINDS: &[&str] = &[
    "call_expression",
    "call",
    "method_invocation",
    "invocation_expression",
    "function_call",
    "function_call_expression",
    "member_call_expression",
    "nullsafe_member_call_expression",
    "scoped_call_expression",
    "macro_invocation",
    "new_expression",
    "object_creation_expression",
];

fn call_label(kind: &str, grammar: &str) -> Option<&'static str> {
    match kind {
        "macro_invocation" => Some("macro"),
        "new_expression" | "object_creation_expression" => Some("new"),
        "command" if grammar == "bash" => Some("command"),
        k if CALL_KINDS.contains(&k) => Some("call"),
        _ => None,
    }
}

fn callee_text(call: Node, source: &[u8]) -> String {
    let field = |name: &str| call.child_by_field_name(name);
    let text = |n: Node| n.utf8_text(source).unwrap_or("").to_string();
    let raw = if let (Some(object), Some(name)) = (
        field("object").or_else(|| field("receiver")),
        field("name").or_else(|| field("method")),
    ) {
        format!("{}.{}", text(object), text(name))
    } else if let (Some(scope), Some(name)) = (field("scope"), field("name")) {
        format!("{}::{}", text(scope), text(name))
    } else if let Some(n) = field("function")
        .or_else(|| field("macro"))
        .or_else(|| field("constructor"))
        .or_else(|| field("type"))
        .or_else(|| field("name"))
        .or_else(|| field("method"))
        .or_else(|| field("target"))
        .or_else(|| call.named_child(0))
    {
        text(n)
    } else {
        text(call)
    };
    let first = raw.lines().next().unwrap_or("");
    truncate_display(&first.split_whitespace().collect::<Vec<_>>().join(" "), 60)
}

fn local_label(node: Node, grammar: &str) -> Option<&'static str> {
    let label = match node.kind() {
        "lexical_declaration" | "variable_declaration" => match node.child(0).map(|c| c.kind()) {
            Some("const") => "const",
            Some("let") => "let",
            Some("local") => "local",
            _ => "var",
        },
        "let_declaration" => "let",
        "short_var_declaration" => ":=",
        "var_declaration" => "var",
        "const_declaration" => "const",
        "local_variable_declaration" | "local_declaration_statement" => "local",
        "assignment" if grammar == "python" => {
            // Only plain name bindings; `self.x = ...` is an attribute write.
            let left = node.child_by_field_name("left")?;
            if matches!(left.kind(), "identifier" | "pattern_list" | "tuple_pattern") {
                "="
            } else {
                return None;
            }
        }
        "variable_assignment" if grammar == "bash" => "=",
        _ => return None,
    };
    Some(label)
}

fn binding_names(node: Node, source: &[u8], label: &str) -> String {
    let text = |n: Node| n.utf8_text(source).unwrap_or("").to_string();
    let mut names: Vec<String> = Vec::new();
    let mut cursor = node.walk();
    match node.kind() {
        "lexical_declaration" | "variable_declaration" | "local_variable_declaration" => {
            for child in node.named_children(&mut cursor) {
                if child.kind().ends_with("declarator") {
                    if let Some(n) = child
                        .child_by_field_name("name")
                        .or_else(|| child.named_child(0))
                    {
                        names.push(text(n));
                    }
                }
            }
        }
        "let_declaration" => names.extend(node.child_by_field_name("pattern").map(text)),
        "short_var_declaration" | "assignment" => {
            names.extend(node.child_by_field_name("left").map(text))
        }
        "variable_assignment" => names.extend(node.child_by_field_name("name").map(text)),
        "var_declaration" | "const_declaration" | "local_declaration_statement" => {
            // Go specs and C# declarators sit one or two levels down.
            let mut stack = vec![node];
            while let Some(n) = stack.pop() {
                match n.kind() {
                    "var_spec" | "const_spec" => {
                        let mut c = n.walk();
                        names.extend(n.children_by_field_name("name", &mut c).map(text));
                    }
                    "variable_declarator" => {
                        if let Some(id) = n.child_by_field_name("name").or_else(|| n.named_child(0))
                        {
                            names.push(text(id));
                        }
                    }
                    _ => {
                        let mut c = n.walk();
                        let kids: Vec<Node> = n.named_children(&mut c).collect();
                        stack.extend(kids.into_iter().rev());
                    }
                }
            }
        }
        _ => {}
    }
    let joined = if names.is_empty() {
        // Fallback: the text before '=', without the declaring keyword.
        let first = text(node).lines().next().unwrap_or("").to_string();
        let lhs = first.split('=').next().unwrap_or("").trim().to_string();
        lhs.strip_prefix(label)
            .map(str::trim)
            .unwrap_or(&lhs)
            .to_string()
    } else {
        names.join(", ")
    };
    truncate_display(&joined.split_whitespace().collect::<Vec<_>>().join(" "), 60)
}

// ─── find-idents ───────────────────────────────────────────────────────

/// Syntactic role of an identifier occurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Declaration,
    Read,
    Write,
    Call,
    Import,
}

impl Role {
    pub const ALL: [&'static str; 5] = ["declaration", "read", "write", "call", "import"];

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Declaration => "declaration",
            Self::Read => "read",
            Self::Write => "write",
            Self::Call => "call",
            Self::Import => "import",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "declaration" => Some(Self::Declaration),
            "read" => Some(Self::Read),
            "write" => Some(Self::Write),
            "call" => Some(Self::Call),
            "import" => Some(Self::Import),
            _ => None,
        }
    }
}

/// Node types that represent identifiers across grammars.
const IDENTIFIER_TYPES: &[&str] = &[
    "identifier",
    "type_identifier",
    "property_identifier",
    "shorthand_property_identifier",
    "shorthand_property_identifier_pattern",
    "field_identifier",
    "package_identifier",
    "simple_identifier",
    "name",
    "constant",
    "symbol",
];

pub struct IdentMatch {
    /// 1-based line.
    pub line: usize,
    /// 1-based byte column.
    pub col: usize,
    pub role: Role,
    /// Qualified enclosing symbol, e.g. `Box::alpha`; `None` at file scope.
    pub scope: Option<String>,
}

/// Every identifier node whose text is exactly `name`, in source order.
pub fn find_identifiers(tree: &Tree, grammar: &str, source: &str, name: &str) -> Vec<IdentMatch> {
    let mut matches: Vec<IdentMatch> = Vec::new();
    if name.is_empty() || !source.contains(name) {
        return matches;
    }
    let bytes = source.as_bytes();
    let mut cursor = tree.walk();
    loop {
        let node = cursor.node();
        if node.is_named()
            && IDENTIFIER_TYPES.contains(&node.kind())
            && node.utf8_text(bytes) == Ok(name)
        {
            let pos = node.start_position();
            let duplicate = matches
                .last()
                .is_some_and(|m| m.line == pos.row + 1 && m.col == pos.column + 1);
            if !duplicate {
                let role = classify_role(node, grammar, bytes);
                matches.push(IdentMatch {
                    line: pos.row + 1,
                    col: pos.column + 1,
                    role,
                    scope: enclosing_label(node, role, grammar, bytes),
                });
            }
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return matches;
            }
        }
    }
}

fn enclosing_label(node: Node, role: Role, grammar: &str, source: &[u8]) -> Option<String> {
    let mut chain = enclosing_symbols(node, grammar, source);
    // A declaration's own symbol is not its scope: `alpha` in `function alpha`
    // belongs to whatever encloses the function. The name sits at most two
    // levels down (`const f = ...` goes through a declarator); a parameter
    // that shares the function's name sits deeper and stays inside it.
    if role == Role::Declaration {
        if let Some(last) = chain.last() {
            let own_name = node.utf8_text(source).ok() == Some(last.name.as_str());
            let names_it = if grammar == "elixir" {
                is_elixir_def_name(node, source)
            } else {
                is_near_descendant(node, last.node, 2)
            };
            if own_name && names_it {
                chain.pop();
            }
        }
    }
    if chain.is_empty() {
        None
    } else {
        Some(qualified_names(&chain, grammar, source).join("::"))
    }
}

fn is_near_descendant(node: Node, ancestor: Node, hops: usize) -> bool {
    let mut current = node;
    for _ in 0..hops {
        match current.parent() {
            Some(p) if p == ancestor => return true,
            Some(p) => current = p,
            None => return false,
        }
    }
    false
}

const IMPORT_KINDS: &[&str] = &[
    "import_statement",
    "import_from_statement",
    "future_import_statement",
    "import_declaration",
    "import_spec",
    "import_clause",
    "import_specifier",
    "namespace_import",
    "named_imports",
    "import_header",
    "import_list",
    "use_declaration",
    "using_directive",
    "extern_crate_declaration",
];

const MEMBER_KINDS: &[&str] = &[
    "member_expression",
    "selector_expression",
    "field_expression",
    "attribute",
    "scoped_identifier",
    "member_access_expression",
    "navigation_expression",
];

const MEMBER_FIELDS: &[&str] = &["property", "field", "attribute", "name"];

fn classify_role(node: Node, grammar: &str, source: &[u8]) -> Role {
    let mut ancestor = node.parent();
    while let Some(a) = ancestor {
        if IMPORT_KINDS.contains(&a.kind()) {
            return Role::Import;
        }
        ancestor = a.parent();
    }
    if is_declaration(node, grammar, source) {
        Role::Declaration
    } else if is_call(node) {
        Role::Call
    } else if is_write(node) {
        Role::Write
    } else {
        Role::Read
    }
}

fn declaration_like(kind: &str) -> bool {
    kind.ends_with("_declaration")
        || kind.ends_with("_definition")
        || kind.ends_with("_declarator")
        || kind.ends_with("_item")
        || kind.ends_with("_spec")
        || kind.ends_with("_signature")
        || kind.ends_with("parameter")
        || kind.ends_with("_variant")
        || kind.ends_with("_binding")
        || matches!(
            kind,
            "method" | "singleton_method" | "class" | "module" | "enum_assignment"
        )
}

/// Destructuring wrappers that keep an identifier in binding position.
fn climbs_pattern(parent_kind: &str, field: Option<&str>) -> bool {
    match parent_kind {
        "pair_pattern" => field == Some("value"),
        "assignment_pattern" => field == Some("left"),
        // `Foo { x: y }` binds y; the field name x is not a binding.
        "field_pattern" => field == Some("pattern"),
        // `Some(x)` / `Foo { .. }`: the type path is read, the inner patterns bind.
        "tuple_struct_pattern" | "struct_pattern" => field != Some("type"),
        "array_pattern"
        | "object_pattern"
        | "rest_pattern"
        | "tuple_pattern"
        | "pattern_list"
        | "slice_pattern"
        | "reference_pattern"
        | "mut_pattern"
        | "list_splat_pattern"
        | "dictionary_splat_pattern"
        | "captured_pattern"
        | "or_pattern" => true,
        _ => false,
    }
}

const BINDING_SITES: &[&str] = &[
    "parameter",
    "required_parameter",
    "optional_parameter",
    "closure_parameters",
    "parameters",
    "lambda_parameters",
    "typed_parameter",
    "formal_parameters",
    "let_condition",
    "for_expression",
];

/// `head` is a function definer's first argument (`name` or `name(params)`),
/// possibly under a `when` guard.
fn is_elixir_def_head(head: Node, source: &[u8]) -> bool {
    let mut up = head.parent();
    if let Some(op) =
        up.filter(|p| p.kind() == "binary_operator" && p.child_by_field_name("left") == Some(head))
    {
        up = op.parent();
    }
    up.filter(|a| a.kind() == "arguments")
        .and_then(|a| a.parent())
        .is_some_and(|definer| elixir_defines_function(definer, source))
}

/// The function name in `def name` or `def name(params)`.
fn is_elixir_def_name(node: Node, source: &[u8]) -> bool {
    is_elixir_def_head(node, source)
        || node.parent().is_some_and(|p| {
            p.kind() == "call"
                && p.child_by_field_name("target") == Some(node)
                && is_elixir_def_head(p, source)
        })
}

/// The name or a parameter in the head of `def name(params)`.
fn is_elixir_binding(node: Node, source: &[u8]) -> bool {
    if is_elixir_def_name(node, source) {
        return true;
    }
    let mut current = node;
    while let Some(parent) = current.parent() {
        if parent.kind() == "arguments" {
            return parent
                .parent()
                .is_some_and(|head| head.kind() == "call" && is_elixir_def_head(head, source));
        }
        current = parent;
    }
    false
}

fn is_declaration(node: Node, grammar: &str, source: &[u8]) -> bool {
    if grammar == "elixir" && is_elixir_binding(node, source) {
        return true;
    }
    let Some(parent) = node.parent() else {
        return false;
    };
    let field = field_in_parent(node);
    if field == Some("name") && declaration_like(parent.kind()) {
        return true;
    }

    // Climb destructuring patterns to the site that binds them.
    let (mut site, mut site_field) = (parent, field);
    while climbs_pattern(site.kind(), site_field) {
        site_field = field_in_parent(site);
        match site.parent() {
            Some(p) => site = p,
            None => return false,
        }
    }
    match site.kind() {
        "variable_declarator" if site_field == Some("name") => return true,
        "let_declaration" if site_field == Some("pattern") => return true,
        "for_in_statement" if site_field == Some("left") => {
            // `for (const x of xs)` declares; `for (x of xs)` assigns.
            if site.child_by_field_name("kind").is_some() {
                return true;
            }
        }
        k if BINDING_SITES.contains(&k) => {
            if matches!(
                site_field,
                None | Some("pattern") | Some("name") | Some("parameter")
            ) {
                return true;
            }
        }
        _ => {}
    }
    if site.kind() == "arrow_function" && site_field == Some("parameter") {
        return true;
    }

    // Go: `a, b := f()` and `for k, v := range m` declare.
    let (go_site, go_field) = if parent.kind() == "expression_list" {
        match parent.parent() {
            Some(gp) => (gp, field_in_parent(parent)),
            None => (parent, field),
        }
    } else {
        (parent, field)
    };
    if go_field == Some("left") {
        if go_site.kind() == "short_var_declaration" {
            return true;
        }
        if go_site.kind() == "range_clause" {
            let header = go_site.utf8_text(source).unwrap_or("");
            if let Some(range_at) = header.find("range") {
                if header[..range_at].contains(":=") {
                    return true;
                }
            }
        }
    }

    // Grammars without a `name` field: the symbol's first identifier child.
    if let Some(sym) = node_symbol(parent, grammar, source) {
        if node.utf8_text(source).ok() == Some(sym.name.as_str()) {
            let mut cursor = parent.walk();
            let first_ident = parent
                .named_children(&mut cursor)
                .find(|c| IDENTIFIER_TYPES.contains(&c.kind()));
            if first_ident == Some(node) {
                return true;
            }
        }
    }
    false
}

fn is_call(node: Node) -> bool {
    let Some(parent) = node.parent() else {
        return false;
    };
    let field = field_in_parent(node);
    // Java/PHP/Ruby name the method as a field of the call node itself.
    if matches!(
        parent.kind(),
        "method_invocation"
            | "member_call_expression"
            | "nullsafe_member_call_expression"
            | "scoped_call_expression"
    ) && field == Some("name")
    {
        return true;
    }
    if parent.kind() == "call" && field == Some("method") {
        return true;
    }

    let mut callee = node;
    if MEMBER_KINDS.contains(&parent.kind()) && field.is_some_and(|f| MEMBER_FIELDS.contains(&f)) {
        callee = parent;
    } else if parent.kind() == "navigation_suffix" {
        if let Some(nav) = parent.parent() {
            callee = nav;
        }
    }
    if let Some(p) = callee.parent() {
        if p.kind() == "generic_function" && field_in_parent(callee) == Some("function") {
            callee = p;
        }
    }
    let Some(call) = callee.parent() else {
        return false;
    };
    if !CALL_KINDS.contains(&call.kind()) {
        return false;
    }
    match field_in_parent(callee) {
        Some("function" | "macro" | "constructor" | "name" | "method" | "target" | "type") => true,
        None => first_named_child_is(call, callee),
        _ => false,
    }
}

const ASSIGNMENT_KINDS: &[&str] = &[
    "assignment_expression",
    "assignment",
    "augmented_assignment_expression",
    "augmented_assignment",
    "compound_assignment_expr",
    "assignment_statement",
    "operator_assignment",
];

fn is_write(node: Node) -> bool {
    let mut current = node;
    while let Some(parent) = current.parent() {
        let field = field_in_parent(current);
        let kind = parent.kind();
        let climb = if MEMBER_KINDS.contains(&kind) {
            // `obj.x = 1` writes x; obj itself is only read.
            field.is_some_and(|f| MEMBER_FIELDS.contains(&f))
        } else if matches!(
            kind,
            "subscript_expression" | "index_expression" | "subscript" | "element_access_expression"
        ) {
            // `xs[i] = 1` mutates xs; the index is read.
            matches!(field, Some("object" | "value" | "operand" | "expression"))
                || (field.is_none() && first_named_child_is(parent, current))
        } else if matches!(
            kind,
            "expression_list"
                | "pattern_list"
                | "tuple_pattern"
                | "parenthesized_expression"
                | "tuple"
                | "tuple_expression"
                | "array_pattern"
                | "object_pattern"
                | "left_hand_side"
                | "directly_assignable_expression"
        ) {
            // Transparent grouping on either side; the assignment decides.
            true
        } else if ASSIGNMENT_KINDS.contains(&kind) {
            return field == Some("left")
                || (field.is_none() && first_named_child_is(parent, current));
        } else {
            return matches!(
                kind,
                "update_expression" | "inc_statement" | "dec_statement"
            ) || (matches!(kind, "for_statement" | "range_clause")
                && field == Some("left"))
                || (kind == "variable_assignment" && field == Some("name"));
        };
        if !climb {
            return false;
        }
        current = parent;
    }
    false
}

//! Output formats here are parsed by the pi tree-sitter-nav extension
//! (`source-excerpts.ts`): outline rows `name (kind Lstart-Lend)`, a scope
//! header `L<line> scope:` or `L<line>: file scope`, and an inspect target row
//! first. Change them together.

use code_search::languages::load_language;
use code_search::nav::{self, Categories, InspectOptions, InspectTarget, Role};
use code_search::symbols::{extract_symbols, format_outline, ExtractOptions};
use tree_sitter::{Parser, Tree};

fn parse(grammar: &str, source: &str) -> Tree {
    let mut parser = Parser::new();
    parser
        .set_language(&load_language(grammar).unwrap())
        .unwrap();
    parser.parse(source, None).unwrap()
}

fn outline(grammar: &str, source: &str) -> String {
    let tree = parse(grammar, source);
    format_outline(
        &extract_symbols(&tree, grammar, source, &ExtractOptions::default()),
        false,
    )
}

const BOX_TS: &str =
    "// scope\nclass Box {\n  alpha() {\n    return this.beta();\n  }\n  beta() { return 1; }\n}\n";

#[test]
fn scope_prints_the_enclosing_breadcrumb() {
    let tree = parse("typescript", BOX_TS);
    assert_eq!(
        nav::scope(&tree, "typescript", BOX_TS, 4, None, false).unwrap(),
        "L4 scope:\nBox (class L2-L7)\n└── alpha (method L3-L5)"
    );
    assert_eq!(
        nav::scope(&tree, "typescript", BOX_TS, 4, Some(12), false).unwrap(),
        "L4:12 scope:\nBox (class L2-L7)\n└── alpha (method L3-L5)"
    );
}

#[test]
fn scope_outside_every_symbol_is_file_scope() {
    let tree = parse("typescript", BOX_TS);
    assert_eq!(
        nav::scope(&tree, "typescript", BOX_TS, 1, None, false).unwrap(),
        "L1: file scope"
    );
    assert!(nav::scope(&tree, "typescript", BOX_TS, 99, None, false).is_err());
}

#[test]
fn whole_line_lookups_see_symbols_behind_leading_wrappers() {
    // The first token is `export` or a YAML `-`, which holds the symbol as a
    // child rather than sitting inside it.
    let ts = "export interface Base { id: string }\nexport function run() {\n  return 1;\n}\n";
    let tree = parse("typescript", ts);
    assert_eq!(
        nav::scope(&tree, "typescript", ts, 1, None, false).unwrap(),
        "L1 scope:\nBase (iface L1)"
    );
    assert_eq!(
        nav::scope(&tree, "typescript", ts, 2, None, false).unwrap(),
        "L2 scope:\nrun (fn L2-L4)"
    );
    // An explicit column still means the node under it.
    assert_eq!(
        nav::scope(&tree, "typescript", ts, 2, Some(1), false).unwrap(),
        "L2:1 scope:\nrun (fn L2-L4)"
    );
    let py = "@cache\ndef load():\n    return 1\n";
    let tree = parse("python", py);
    assert_eq!(
        nav::scope(&tree, "python", py, 1, Some(2), false).unwrap(),
        "L1:2 scope:\nload (fn L1-L3)"
    );
    let yaml = "tasks:\n  - name: build\n    run: make\n";
    let tree = parse("yaml", yaml);
    let out = nav::scope(&tree, "yaml", yaml, 2, None, false).unwrap();
    assert!(out.ends_with("name (prop L2)"), "{}", out);
    let keyed = "fe80::/10: local\n";
    let tree = parse("yaml", keyed);
    let out = nav::inspect(
        &tree,
        "yaml",
        keyed,
        "k.yaml",
        InspectTarget::Symbol("fe80::/10"),
        &InspectOptions {
            depth: 3,
            include: Categories::all(),
            signatures: false,
        },
    )
    .unwrap();
    assert!(out.starts_with("fe80::/10 (prop L1)"), "{}", out);
}

#[test]
fn inspect_lists_only_included_categories_under_the_target() {
    let tree = parse("typescript", BOX_TS);
    let calls = Categories::parse("calls").unwrap();
    let out = nav::inspect(
        &tree,
        "typescript",
        BOX_TS,
        "box.ts",
        InspectTarget::Symbol("Box::alpha"),
        &InspectOptions {
            depth: 3,
            include: calls,
            signatures: false,
        },
    )
    .unwrap();
    assert_eq!(
        out,
        "Box::alpha (method L3-L5)\n└── this.beta() [call:call L4]"
    );

    // Dotted qualification and --line resolve the same way.
    let dotted = nav::inspect(
        &tree,
        "typescript",
        BOX_TS,
        "box.ts",
        InspectTarget::Symbol("Box.alpha"),
        &InspectOptions {
            depth: 3,
            include: calls,
            signatures: false,
        },
    )
    .unwrap();
    let by_line = nav::inspect(
        &tree,
        "typescript",
        BOX_TS,
        "box.ts",
        InspectTarget::Line {
            line: 4,
            column: None,
        },
        &InspectOptions {
            depth: 3,
            include: calls,
            signatures: false,
        },
    )
    .unwrap();
    assert_eq!(dotted, out);
    assert_eq!(by_line, out);
}

#[test]
fn inspect_lists_nested_symbols_without_descending() {
    let tree = parse("typescript", BOX_TS);
    let out = nav::inspect(
        &tree,
        "typescript",
        BOX_TS,
        "box.ts",
        InspectTarget::Symbol("Box"),
        &InspectOptions {
            depth: 3,
            include: Categories::all(),
            signatures: false,
        },
    )
    .unwrap();
    assert_eq!(
        out,
        "Box (class L2-L7)\n├── alpha [symbol:method L3-L5]\n└── beta [symbol:method L6]"
    );
}

#[test]
fn inspect_reports_items_hidden_by_depth() {
    let tree = parse("typescript", BOX_TS);
    let out = nav::inspect(
        &tree,
        "typescript",
        BOX_TS,
        "box.ts",
        InspectTarget::Symbol("alpha"),
        &InspectOptions {
            depth: 0,
            include: Categories::all(),
            signatures: false,
        },
    )
    .unwrap();
    assert!(out.starts_with("Box::alpha (method L3-L5)\n["), "{}", out);
    assert!(out.contains("deeper than --depth 0"), "{}", out);
}

#[test]
fn inspect_errors_name_the_problem() {
    let src = "class A { run() {} }\nclass B { run() {} }\n";
    let tree = parse("typescript", src);
    let missing = nav::inspect(
        &tree,
        "typescript",
        src,
        "a.ts",
        InspectTarget::Symbol("nope"),
        &InspectOptions {
            depth: 3,
            include: Categories::all(),
            signatures: false,
        },
    );
    assert_eq!(missing.unwrap_err(), "No symbol named \"nope\" in a.ts");
    // A leading module segment naming the file itself is dropped.
    let prefixed = nav::inspect(
        &tree,
        "typescript",
        src,
        "src/a.ts",
        InspectTarget::Symbol("crate::a::nope"),
        &InspectOptions {
            depth: 3,
            include: Categories::all(),
            signatures: false,
        },
    );
    assert_eq!(
        prefixed.unwrap_err(),
        "No symbol named \"crate::a::nope\" in src/a.ts"
    );
    let ambiguous = nav::inspect(
        &tree,
        "typescript",
        src,
        "a.ts",
        InspectTarget::Symbol("run"),
        &InspectOptions {
            depth: 3,
            include: Categories::all(),
            signatures: false,
        },
    )
    .unwrap_err();
    assert!(ambiguous.contains("ambiguous"), "{}", ambiguous);
    assert!(
        ambiguous.contains("A::run (method L1)") && ambiguous.contains("B::run (method L2)"),
        "{}",
        ambiguous
    );
    assert!(nav::inspect(
        &tree,
        "typescript",
        src,
        "a.ts",
        InspectTarget::Symbol("a::B::run"),
        &InspectOptions {
            depth: 3,
            include: Categories::all(),
            signatures: false
        }
    )
    .is_ok());
}

#[test]
fn markdown_sections_nest_and_stop_before_the_next_heading() {
    let src = "Preamble text.\n\n# Title\nIntro.\n\n## Details\nMore.\n\n# Next\nEnd.\n";
    assert_eq!(
        outline("markdown", src),
        "Title (section L3-L8)\n└── Details (section L6-L8)\nNext (section L9-L10)"
    );
    let single = "# No symbols found in this title\nActual section content.\n";
    assert_eq!(
        outline("markdown", single),
        "No symbols found in this title (section L1-L2)"
    );
}

#[test]
fn outline_depth_zero_keeps_only_top_level_symbols() {
    let src = "# Title\n\n## Details\n\n# Next\n";
    let tree = parse("markdown", src);
    let opts = ExtractOptions {
        max_depth: Some(0),
        ..Default::default()
    };
    assert_eq!(
        format_outline(&extract_symbols(&tree, "markdown", src, &opts), false),
        "Title (section L1-L4)\nNext (section L5)"
    );
}

#[test]
fn long_multibyte_names_truncate_without_panicking() {
    let heading = format!("# {}\n", "🦊".repeat(40));
    let out = outline("markdown", &heading);
    assert!(out.ends_with("... (section L1)"), "{}", out);
}

fn roles(grammar: &str, source: &str, name: &str) -> Vec<(usize, Role, Option<String>)> {
    let tree = parse(grammar, source);
    nav::find_identifiers(&tree, grammar, source, name)
        .into_iter()
        .map(|m| (m.line, m.role, m.scope))
        .collect()
}

fn s(scope: &str) -> Option<String> {
    Some(scope.to_string())
}

#[test]
fn typescript_roles() {
    let src = r#"import { count } from "./lib";
class Box {
  count = 0;
  bump(count: number) {
    this.count = count + 1;
    return count(1);
  }
}
const fmt = (n) => n;
"#;
    use Role::*;
    assert_eq!(
        roles("typescript", src, "count"),
        vec![
            (1, Import, None),
            (3, Declaration, s("Box")),
            (4, Declaration, s("Box::bump")),
            (5, Write, s("Box::bump")),
            (5, Read, s("Box::bump")),
            (6, Call, s("Box::bump")),
        ]
    );
    assert_eq!(
        roles("typescript", src, "fmt"),
        vec![(9, Declaration, None)]
    );
}

#[test]
fn go_roles_qualify_methods_by_receiver() {
    let src = "package main\n\nfunc (s *Server) Run(n int) int {\n\tn = 2\n\tm, n := n, 3\n\ts.Run(n)\n\treturn m\n}\n";
    use Role::*;
    assert_eq!(
        roles("go", src, "n"),
        vec![
            (3, Declaration, s("Server::Run")),
            (4, Write, s("Server::Run")),
            (5, Declaration, s("Server::Run")),
            (5, Read, s("Server::Run")),
            (6, Read, s("Server::Run")),
        ]
    );
    assert_eq!(
        roles("go", src, "Run"),
        vec![(3, Declaration, None), (6, Call, s("Server::Run"))]
    );
}

#[test]
fn python_and_rust_roles() {
    use Role::*;
    let py = "from lib import x\n\ndef f(x):\n    x = 1\n    return g(x)\n";
    assert_eq!(
        roles("python", py, "x"),
        vec![
            (1, Import, None),
            (3, Declaration, s("f")),
            (4, Write, s("f")),
            (5, Read, s("f"))
        ]
    );
    let rs = "fn count(count: u32) -> u32 {\n    let n = count;\n    count::<u32>(n)\n}\n";
    assert_eq!(
        roles("rust", rs, "count"),
        vec![
            (1, Declaration, None),
            (1, Declaration, s("count")),
            (2, Read, s("count")),
            (3, Call, s("count")),
        ]
    );
}

#[test]
fn elixir_modules_nest_defs_and_bind_parameters() {
    let src = "defmodule Shop.Cart do\n  def add(cart, item) when is_map(item) do\n    Map.put(cart, item.id, item)\n  end\n  def empty, do: %{}\nend\n";
    assert_eq!(
        outline("elixir", src),
        "Shop.Cart (mod L1-L6)\n├── add (fn L2-L4)\n└── empty (fn L5)"
    );
    let tree = parse("elixir", src);
    let out = nav::inspect(
        &tree,
        "elixir",
        src,
        "cart.ex",
        InspectTarget::Symbol("Cart.add"),
        &InspectOptions {
            depth: 3,
            include: Categories::parse("calls").unwrap(),
            signatures: false,
        },
    )
    .unwrap();
    assert!(out.starts_with("Shop.Cart::add (fn L2-L4)"), "{}", out);
    use Role::*;
    assert_eq!(
        roles("elixir", src, "cart"),
        vec![
            (2, Declaration, s("Shop.Cart::add")),
            (3, Read, s("Shop.Cart::add"))
        ]
    );
    assert_eq!(
        roles("elixir", src, "add"),
        vec![(2, Declaration, s("Shop.Cart"))]
    );
    assert_eq!(
        roles("elixir", src, "empty"),
        vec![(5, Declaration, s("Shop.Cart"))]
    );
}

#[test]
fn struct_literal_fields_are_writes_but_map_keys_are_reads() {
    use Role::*;
    let go = "package p\n\nfunc f() {\n\t_ = T{deadline: 1}\n\t_ = map[string]int{deadline: 1}\n\t_ = []T{{deadline: 2}}\n\t_ = &pkg.T{deadline: 3}\n}\n";
    assert_eq!(
        roles("go", go, "deadline"),
        vec![
            (4, Write, s("f")),
            (5, Read, s("f")),
            (6, Write, s("f")),
            (7, Write, s("f")),
        ]
    );
    let rs =
        "fn f(deadline: u32) {\n    let t = T { deadline: 1 };\n    let u = T { deadline };\n}\n";
    assert_eq!(
        roles("rust", rs, "deadline"),
        vec![
            (1, Declaration, s("f")),
            (2, Write, s("f")),
            (3, Read, s("f"))
        ]
    );
}

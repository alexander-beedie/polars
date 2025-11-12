# Custom Sphinx Extensions

## Railroad Diagrams

The `railroad_diagrams.py` extension provides syntax railroad diagrams for SQL documentation.

### About the Railroad Library

This extension uses the **railroad-diagrams** library by Tab Atkins:

- **PyPI**: [railroad-diagrams](https://pypi.org/project/railroad-diagrams/)
- **GitHub**:
  [github.com/tabatkins/railroad-diagrams](https://github.com/tabatkins/railroad-diagrams)
- **Purpose**: Generate SVG railroad syntax diagrams (like those on JSON.org)
- **Version**: 3.0.1

The library provides a Python API for programmatically creating railroad diagrams. Our extension
adds a convenient RST directive interface with EBNF-like syntax parsing.

### Usage

Add the `.. railroad::` directive to any RST file:

```rst
.. railroad::

   select_stmt ::= 'SELECT' distinct? column_list
                   'FROM' table_name

   distinct ::= 'DISTINCT'
   column_list ::= '*' | column_expr (',' column_expr)*
```

### Syntax

The directive uses EBNF-like notation:

- **Terminals**: Literal strings in single or double quotes: `'SELECT'`, `"FROM"`
- **Non-terminals**: Unquoted identifiers that reference other rules: `column_list`, `table_name`
- **Optional**: Add `?` suffix: `distinct?`
- **Zero or more**: Add `*` suffix: `column_expr*`
- **One or more**: Add `+` suffix: `column_expr+`
- **Alternatives**: Use `|` separator: `'LEFT' | 'RIGHT'`
- **Grouping**: Use parentheses: `('FULL' | 'LEFT' | 'RIGHT')`
- **Grouped operators**: Apply operators to groups: `('LEFT' | 'RIGHT')?`
- **Rule definitions**: Use `::=` to define rules: `rule_name ::= definition`

### Examples

**Simple clause:**

```rst
.. railroad::

   where_clause ::= 'WHERE' condition
```

**Complex with grouped alternatives:**

```rst
.. railroad::

   join_type ::= 'CROSS'
               | 'INNER'
               | 'NATURAL'? ('FULL' | 'LEFT' | 'RIGHT')
               | ('LEFT' | 'RIGHT')? ('SEMI' | 'ANTI')
```

This creates a more compact diagram where `NATURAL` is shown once as optional before the choice of
FULL/LEFT/RIGHT, rather than repeating it for each alternative.

**With optional and repeating elements:**

```rst
.. railroad::

   count_expr ::= 'COUNT' '(' ('*' | distinct? expression) ')'

   distinct ::= 'DISTINCT'
```

### Requirements

- `railroad-diagrams==3.0.1` (installed via `requirements-docs.txt`)

### Styling

The extension automatically includes `_static/css/railroad.css` which provides:

- Theme-aware colors using CSS variables (`--pst-color-text-base`, `--pst-color-primary-bg`)
- Transparent backgrounds that work with both light and dark themes
- Responsive container styling with overflow handling
- Proper fallbacks for browsers without CSS variable support

To customize the appearance, edit `source/_static/css/railroad.css`.

### Notes

- The first rule defined becomes the main diagram
- Additional rules can be referenced by the main rule
- If the railroad-diagrams package is not available, the directive falls back to rendering as EBNF
  code blocks
- The extension generates clean SVG diagrams without embedded CSS
- CSS is loaded once per page, not embedded in each diagram

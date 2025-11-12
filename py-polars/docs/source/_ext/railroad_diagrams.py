"""
Sphinx extension for adding railroad diagrams.

Provides a `.. railroad::` directive for embedding diagrams using EBNF-like notation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from railroad import (
    Choice,
    Diagram,
    NonTerminal,
    OneOrMore,
    Optional,
    Sequence,
    Terminal,
    ZeroOrMore,
)

if TYPE_CHECKING:
    from sphinx.application import Sphinx


class RailroadDirective(Directive):
    """Directive for railroad diagrams using EBNF-like syntax."""

    has_content = True
    optional_arguments = 1
    option_spec: ClassVar[dict[str, Any]] = {
        "alt": directives.unchanged,
        "caption": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        """Generate the railroad diagram from EBNF content."""
        content = "\n".join(self.content)
        rules = self._parse_ebnf(content)

        if not rules:
            error = self.state_machine.reporter.error(
                "No valid EBNF rules found in railroad directive",
                nodes.literal_block(content, content),
                line=self.lineno,
            )
            return [error]

        # Generate SVG for the first rule (main diagram)
        main_rule_name = next(iter(rules.keys()))
        diagram_svg = self._generate_diagram(rules, main_rule_name)

        # Create a raw HTML node with the SVG
        raw_node = nodes.raw("", diagram_svg, format="html")

        # Wrap in a container
        container = nodes.container()
        container["classes"].append("railroad-diagram")
        container.append(raw_node)
        return [container]

    def _generate_diagram(self, rules: dict[str, Any], rule_name: str) -> str:
        """Generate SVG diagram for a rule."""
        import io

        rule_def = rules.get(rule_name, "")
        elements = self._parse_rule_elements(rule_def, rules, expand_refs=True)

        # Create diagram
        diagram = (
            Diagram(*elements) if isinstance(elements, list) else Diagram(elements)
        )

        # Generate SVG (CSS will be loaded from external file)
        buffer = io.StringIO()
        diagram.writeSvg(buffer.write)
        svg = buffer.getvalue()

        # Wrap in container div
        return f'<div class="railroad-container">{svg}</div>'

    def _parse_ebnf(self, content: str) -> dict[str, Any]:
        """Parse EBNF-like notation into a dict of rules."""
        rules = {}
        current_rule = None
        current_rhs = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check if this is a rule definition (contains ::=)
            if "::=" in line:
                # Save previous rule if any
                if current_rule:
                    rules[current_rule] = " ".join(current_rhs)

                # Start new rule
                lhs, rhs = line.split("::=", 1)
                current_rule = lhs.strip()
                current_rhs = [rhs.strip()]
            elif current_rule and (line.startswith("|") or not line.startswith("::")):
                # Continuation of current rule
                current_rhs.append(line)

        # Save last rule
        if current_rule:
            rules[current_rule] = " ".join(current_rhs)

        return rules

    def _parse_rule_elements(
        self, rule_def: str, rules: dict[str, Any], *, expand_refs: bool = False
    ) -> list[Any]:
        """Parse rule definition into railroad elements."""
        # Tokenize the rule definition
        tokens = self._tokenize(rule_def)

        # Parse with recursive descent for parentheses
        elements, _ = self._parse_sequence(tokens, 0, rules, expand_refs=expand_refs)
        return elements

    def _parse_sequence(
        self, tokens: list[str], start: int, rules: dict[str, Any], *, expand_refs: bool
    ) -> tuple[list[Any], int]:
        """Parse a sequence of elements, handling alternatives and groups."""
        alternatives = []
        current_seq = []
        i = start

        while i < len(tokens):
            token = tokens[i].strip()

            if token == ")":
                # End of group
                break
            elif token == "|":
                # End of this alternative
                if current_seq:
                    alternatives.append(current_seq)
                    current_seq = []
                i += 1
            elif token == "(":
                # Start of grouped expression
                group_elements, new_i = self._parse_sequence(
                    tokens, i + 1, rules, expand_refs=expand_refs
                )
                i = new_i + 1  # Skip past the closing )

                # Check for operators after the group
                if i < len(tokens) and tokens[i] in ("?", "*", "+"):
                    op = tokens[i]
                    i += 1
                    if len(group_elements) == 1:
                        elem = group_elements[0]
                    elif group_elements:
                        elem = Sequence(*group_elements)
                    else:
                        elem = Sequence()  # Empty sequence for empty groups

                    if op == "?":
                        current_seq.append(Optional(elem))
                    elif op == "*":
                        current_seq.append(ZeroOrMore(elem))
                    elif op == "+":
                        current_seq.append(OneOrMore(elem))
                else:
                    # No operator, add as sequence or choice
                    if len(group_elements) == 1:
                        current_seq.append(group_elements[0])
                    elif group_elements:
                        current_seq.append(Sequence(*group_elements))
            elif token in ("?", "*", "+"):
                # Postfix operator - apply to previous element
                if current_seq:
                    prev_elem = current_seq.pop()
                    if token == "?":
                        current_seq.append(Optional(prev_elem))
                    elif token == "*":
                        current_seq.append(ZeroOrMore(prev_elem))
                    elif token == "+":
                        current_seq.append(OneOrMore(prev_elem))
                i += 1
            elif token:
                # Regular element
                current_seq.append(
                    self._parse_single_element(token, rules, expand_refs=expand_refs)
                )
                i += 1
            else:
                i += 1

        # Add last sequence
        if current_seq:
            alternatives.append(current_seq)

        # If we have alternatives, create a Choice
        if len(alternatives) > 1:
            choice_items = []
            for alt in alternatives:
                if len(alt) == 1:
                    choice_items.append(alt[0])
                else:
                    choice_items.append(Sequence(*alt))
            return [Choice(0, *choice_items)], i
        elif len(alternatives) == 1:
            return alternatives[0], i
        else:
            return [], i

    def _parse_single_element(
        self, token: str, rules: dict[str, Any], *, expand_refs: bool = False
    ) -> Any:
        """Parse a single token into a railroad element."""
        # Check if token has suffix operator (only for non-quoted tokens)
        # Quoted strings are handled separately
        if not token.startswith(("'", '"')):
            if token.endswith("?"):
                base_token = token[:-1]
                return Optional(
                    self._create_element(base_token, rules, expand_refs=expand_refs)
                )
            elif token.endswith("*"):
                base_token = token[:-1]
                return ZeroOrMore(
                    self._create_element(base_token, rules, expand_refs=expand_refs)
                )
            elif token.endswith("+"):
                base_token = token[:-1]
                return OneOrMore(
                    self._create_element(base_token, rules, expand_refs=expand_refs)
                )

        return self._create_element(token, rules, expand_refs=expand_refs)

    def _create_element(
        self, token: str, rules: dict[str, Any], *, expand_refs: bool = False
    ) -> Any:
        """Create a railroad element from a token."""
        if token.startswith("'") and token.endswith("'"):
            # Terminal (literal string)
            return Terminal(token.strip("'"))
        elif token.startswith('"') and token.endswith('"'):
            # Terminal (literal string)
            return Terminal(token.strip('"'))
        elif token in rules and expand_refs:
            # Expand the referenced rule inline (recursively expand nested refs too)
            sub_elements = self._parse_rule_elements(
                rules[token], rules, expand_refs=True
            )
            if len(sub_elements) == 1:
                return sub_elements[0]
            else:
                return Sequence(*sub_elements)
        elif token in rules:
            # Non-terminal (reference to another rule) - don't expand
            return NonTerminal(token)
        else:
            # Treat as non-terminal
            return NonTerminal(token)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizing that respects quotes and parentheses."""
        tokens = []
        current = []
        in_quote = None

        for char in text:
            if char in ("'", '"'):
                if in_quote == char:
                    current.append(char)
                    tokens.append("".join(current))
                    current = []
                    in_quote = None
                elif in_quote is None:
                    if current:
                        tokens.append("".join(current))
                        current = []
                    in_quote = char
                    current.append(char)
                else:
                    current.append(char)

            elif char in ("(", ")") and in_quote is None:
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)

            elif char.isspace() and in_quote is None:
                if current:
                    tokens.append("".join(current))
                    current = []
            else:
                current.append(char)

        if current:
            tokens.append("".join(current))

        return tokens


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the railroad diagrams extension with Sphinx."""
    app.add_directive("railroad", RailroadDirective)
    app.add_css_file("css/railroad.css")
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

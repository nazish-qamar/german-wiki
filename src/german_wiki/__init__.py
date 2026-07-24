"""German Wiki — personal German-learning node graph.

Slice 1: the node layer. Nodes are Markdown + YAML frontmatter in /nodes (source
of truth); SQLite + sqlite-vec is a derived, rebuildable index. See docs/SPEC.md.
"""

__version__ = "0.1.0"

"""K/C conceptual-box records: structural validation and dependency queries.

This is the minimal Gate 2 substrate: schema validation, reference integrity,
acyclicity, and the descendant set that must be reviewed when a commitment is
suspended. Graph operations for agents are built on top of this later.
"""

import json
from pathlib import Path

from .archive.store import RecordInvalid, validate


def load_space(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def edges(space: dict):
    """Yield (prerequisite, dependent) pairs."""
    for node in space["nodes"]:
        for group in node["supports"]:
            for prereq in group:
                yield prereq, node["id"]


def structural_errors(space: dict) -> list:
    errors = []
    ids = [n["id"] for n in space["nodes"]]
    known = set(ids)
    if len(known) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        errors.append(f"duplicate node ids: {dupes}")
    for prereq, dependent in edges(space):
        if prereq not in known:
            errors.append(f"{dependent} is supported by unknown node {prereq}")
        if prereq == dependent:
            errors.append(f"{dependent} supports itself")

    # cycle detection (iterative DFS, colours: 0 new, 1 on stack, 2 done)
    children = {i: [] for i in known}
    for prereq, dependent in edges(space):
        if prereq in children and prereq != dependent:
            children[prereq].append(dependent)
    colour = dict.fromkeys(known, 0)
    for start in sorted(known):
        if colour[start]:
            continue
        stack = [(start, iter(children[start]))]
        colour[start] = 1
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                colour[node] = 2
                stack.pop()
            elif colour[nxt] == 1:
                path = [n for n, _ in stack]
                errors.append("dependency cycle: " + " -> ".join(path[path.index(nxt):] + [nxt]))
            elif colour[nxt] == 0:
                colour[nxt] = 1
                stack.append((nxt, iter(children[nxt])))
    return errors


def validate_space(space: dict):
    """Raise RecordInvalid if the space violates the schema or graph constraints."""
    validate("space", space)
    errors = structural_errors(space)
    if errors:
        raise RecordInvalid("; ".join(errors))


def descendants(space: dict, node_id: str) -> set:
    """Nodes whose recorded justification transitively uses node_id."""
    children = {}
    for prereq, dependent in edges(space):
        children.setdefault(prereq, set()).add(dependent)
    seen, frontier = set(), [node_id]
    while frontier:
        for child in children.get(frontier.pop(), ()):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return seen


def render_markdown(space: dict) -> str:
    """Human-readable dossier of a space, grouped by node role."""
    users = {}
    for prereq, dependent in edges(space):
        users.setdefault(prereq, []).append(dependent)
    ctx = space["context"]
    out = [f"# Box `{space['id']}` v{space['version']}", "",
           f"{len(space['nodes'])} nodes; an edge A -> B means B's recorded justification or construction uses A.", ""]
    for title, key in (("Fixed requirements", "fixed_requirements"), ("Editable priors", "editable_priors"),
                       ("Permitted constructions", "permitted_constructions")):
        out += [f"## {title}", ""] + [f"- {x}" for x in ctx[key]] + [""]
    groups = [
        ("Active commitments of the starting formulation", lambda n: n["active"] and n["type"] in ("assumption", "method")),
        ("Consequences and theoretical results", lambda n: n["type"] == "theorem"),
        ("Definitions", lambda n: n["type"] == "definition"),
        ("Empirical claims", lambda n: n["type"] == "empirical_claim"),
        ("Known alternatives (background)", lambda n: not n["active"] and n["type"] == "method"),
    ]
    placed = set()
    for title, pred in groups:
        members = [n for n in space["nodes"] if n["id"] not in placed and pred(n)]
        if not members:
            continue
        out += [f"## {title}", ""]
        for n in members:
            placed.add(n["id"])
            sup = " or ".join("(" + " + ".join(f"`{p}`" for p in g) + ")" for g in n["supports"]) or "root"
            out.append(f"**`{n['id']}`** ({n['type']}, {n['status']}{', active' if n['active'] else ''}): "
                       f"{n['statement']}")
            out.append(f"  - scope: {n['scope']}")
            out.append(f"  - supported by: {sup}; used by: "
                       + (", ".join(f"`{u}`" for u in sorted(users.get(n['id'], []))) or "-"))
            if n.get("notes"):
                out.append(f"  - note: {n['notes']}")
            for s in n["sources"]:
                out.append(f"  - source: {s}")
            out.append("")
    sources = sorted({s for n in space["nodes"] for s in n["sources"]})
    out += ["## Sources", ""] + [f"- {s}" for s in sources] + [""]
    return "\n".join(out)


def unsupported_after_suspension(space: dict, suspended: set) -> set:
    """Nodes left with no fully available support group once `suspended` is set aside.

    Nodes with no supports (roots) stay available. Suspension means reasoning
    without relying on a commitment; it does not assert its negation.
    """
    by_id = {n["id"]: n for n in space["nodes"]}
    available = {}

    def is_available(nid, stack=()):
        if nid in available:
            return available[nid]
        if nid in suspended or nid not in by_id or nid in stack:
            return False
        groups = by_id[nid]["supports"]
        ok = not groups or any(all(is_available(p, stack + (nid,)) for p in g) for g in groups)
        available[nid] = ok
        return ok

    return {nid for nid in by_id if nid not in suspended and not is_available(nid)}

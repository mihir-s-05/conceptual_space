"""Branch formulations: applying space patches and rendering the two conditions.

DAG condition: typed nodes, supports, dependency queries, and patch operations.
Flat condition: the same substantive content rendered as a prose assumption
register; revisions are free text and nothing is computed from structure.
"""

import copy
import json

from ..archive.store import RecordInvalid
from ..box import descendants, unsupported_after_suspension, validate_space
from ..hashing import canonical_json, sha256_bytes

PATCH_OPS = ("suspend", "restore", "add_node", "set_supports", "edit_node", "set_active")
EDITABLE_FIELDS = ("statement", "scope", "type", "status", "sources", "notes")

PATCH_SCHEMA = {
    "type": "array",
    "description": (
        "Ordered operations on the branch formulation. "
        "suspend{node_id}: reason without relying on a commitment (does not assert its negation). "
        "restore{node_id}. add_node{node}: node has id, type, statement, scope, sources, status, active, supports. "
        "set_supports{node_id, supports}: supports is a list of alternative groups, each a list of node ids "
        "used jointly. edit_node{node_id, fields}. set_active{node_id, active}."
    ),
    "items": {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": list(PATCH_OPS)},
            "node_id": {"type": "string"},
            "node": {"type": "object"},
            "supports": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            "fields": {"type": "object"},
            "active": {"type": "boolean"},
        },
        "required": ["op"],
    },
}


class PatchError(ValueError):
    pass


def space_id(space: dict) -> str:
    return "s_" + sha256_bytes(canonical_json(space))[:16]


def _node(space, nid):
    for n in space["nodes"]:
        if n["id"] == nid:
            return n
    raise PatchError(f"unknown node {nid!r}")


def apply_patch(space: dict, patch: list) -> tuple:
    """Return (new_space, report). Raises PatchError on any invalid operation."""
    if not isinstance(patch, list):
        raise PatchError("space_patch must be a list of operations")
    new = copy.deepcopy(space)
    new["parent_version"] = space["version"]
    new["version"] = space["version"] + 1
    suspended_now = set()
    for i, op in enumerate(patch):
        kind = op.get("op") if isinstance(op, dict) else None
        if kind not in PATCH_OPS:
            raise PatchError(f"operation {i}: unknown op {kind!r}; expected one of {PATCH_OPS}")
        try:
            if kind == "suspend":
                n = _node(new, op["node_id"])
                if n["status"] != "suspended":
                    n["prior_status"] = n["status"]
                    n["status"] = "suspended"
                suspended_now.add(n["id"])
            elif kind == "restore":
                n = _node(new, op["node_id"])
                n["status"] = n.pop("prior_status", "unresolved")
            elif kind == "add_node":
                node = dict(op["node"])
                node.setdefault("sources", [])
                node.setdefault("supports", [])
                node.setdefault("status", "unresolved")
                node.setdefault("active", True)
                if any(n["id"] == node.get("id") for n in new["nodes"]):
                    raise PatchError(f"node {node.get('id')!r} already exists")
                new["nodes"].append(node)
            elif kind == "set_supports":
                _node(new, op["node_id"])["supports"] = [list(g) for g in op["supports"]]
            elif kind == "edit_node":
                n = _node(new, op["node_id"])
                bad = set(op["fields"]) - set(EDITABLE_FIELDS)
                if bad:
                    raise PatchError(f"cannot edit fields {sorted(bad)}")
                n.update(op["fields"])
            elif kind == "set_active":
                _node(new, op["node_id"])["active"] = bool(op["active"])
        except KeyError as e:
            raise PatchError(f"operation {i} ({kind}) is missing {e}") from None
    try:
        validate_space(new)
    except RecordInvalid as e:
        raise PatchError(str(e)) from None
    suspended = {n["id"] for n in new["nodes"] if n["status"] == "suspended"}
    report = {
        "suspended": sorted(suspended),
        "needs_review": sorted(set().union(*(descendants(new, s) for s in suspended_now)) if suspended_now else set()),
        "unsupported": sorted(unsupported_after_suspension(new, suspended)),
    }
    return new, report


def query(space: dict, node_id: str) -> dict:
    n = _node(space, node_id)
    children = sorted(d for d in descendants(space, node_id))
    others = {x["id"] for x in space["nodes"] if x["status"] == "suspended"}
    return {
        "node": n,
        "direct_dependents": sorted({x["id"] for x in space["nodes"] if any(node_id in g for g in x["supports"])}),
        "all_descendants": children,
        "unsupported_if_suspended": sorted(unsupported_after_suspension(space, others | {node_id})),
    }


def render_dag(space: dict) -> str:
    deps = {}
    for n in space["nodes"]:
        for g in n["supports"]:
            for p in g:
                deps.setdefault(p, []).append(n["id"])
    ctx = space["context"]
    lines = [f"FORMULATION {space['id']} v{space['version']}",
             "Fixed requirements: " + "; ".join(ctx["fixed_requirements"]),
             "Editable priors: " + "; ".join(ctx["editable_priors"]),
             "Permitted constructions: " + "; ".join(ctx["permitted_constructions"]),
             "",
             "Nodes (edge A -> B means B's justification or construction uses A):"]
    for n in space["nodes"]:
        sup = " | ".join("(" + " & ".join(g) + ")" for g in n["supports"]) or "none (root)"
        role = "ACTIVE" if n["active"] else "background"
        lines.append(f"[{n['id']}] {n['type']}, {n['status']}, {role}: {n['statement']}")
        lines.append(f"    scope: {n['scope']}; supported by: {sup}; used by: {', '.join(sorted(deps.get(n['id'], []))) or '-'}"
                     + (f"; sources: {', '.join(n['sources'])}" if n["sources"] else ""))
    return "\n".join(lines)


def render_flat(space: dict) -> str:
    """Same claims, sources, scopes, and dependency statements, as plain prose."""
    names = {n["id"]: n["id"] for n in space["nodes"]}
    ctx = space["context"]
    out = ["ASSUMPTION REGISTER AND DOSSIER",
           "Fixed requirements: " + "; ".join(ctx["fixed_requirements"]) + ".",
           "Editable priors: " + "; ".join(ctx["editable_priors"]) + ".",
           "Permitted constructions: " + "; ".join(ctx["permitted_constructions"]) + ".", ""]
    for n in space["nodes"]:
        kind = n["type"].replace("_", " ")
        role = "currently relied on" if n["active"] else "background knowledge / known alternative"
        text = f"{names[n['id']]}. ({kind}; {n['status']}; {role}) {n['statement']} Scope: {n['scope']}."
        if n["supports"]:
            alts = [" together with ".join(g) for g in n["supports"]]
            text += " This relies on " + ", or alternatively on ".join(alts) + "."
        if n["sources"]:
            text += " Sources: " + ", ".join(n["sources"]) + "."
        out.append(text)
    return "\n".join(out)


def dumps(space: dict) -> str:
    return json.dumps(space, indent=1)

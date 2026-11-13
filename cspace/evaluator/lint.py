"""Static guardrails for candidate code.

These catch accidental or naive misuse (file and network access, reseeding,
monkeypatching torch or the evaluator). They are not a security sandbox:
Python cannot be fully confined this way, so audits still inspect code.
"""

import ast
from pathlib import Path

ALLOWED_IMPORTS = {
    "torch", "math", "functools", "typing", "dataclasses", "itertools",
    "collections", "operator", "numbers", "enum", "abc",
}
FORBIDDEN_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "memoryview",
}
FORBIDDEN_ATTRS = {
    # serialization / downloads / RNG control / determinism flags
    "load", "save", "hub", "manual_seed", "seed", "set_rng_state", "manual_seed_all",
    "use_deterministic_algorithms",
    # introspection used to escape or patch the runtime
    "__dict__", "__globals__", "__builtins__", "__subclasses__", "__code__",
    "__closure__", "__getattribute__", "__import__", "__loader__", "__spec__", "__mro__",
}


def _root_name(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def lint_source(source: str) -> list:
    """Return a list of violation strings ("line N: message")."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"line {e.lineno}: syntax error: {e.msg}"]

    violations = []
    module_names = set()  # names bound to imported modules or objects

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    violations.append(f"line {node.lineno}: import of {alias.name!r} is not allowed")
                module_names.add(alias.asname or top)
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if node.level or top not in ALLOWED_IMPORTS:
                violations.append(f"line {node.lineno}: import from {node.module!r} is not allowed")
            for alias in node.names:
                module_names.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                violations.append(f"line {node.lineno}: call to {node.func.id}() is not allowed")
            if node.func.id in {"setattr", "delattr"} and node.args:
                if _root_name(node.args[0]) in module_names:
                    violations.append(f"line {node.lineno}: {node.func.id}() on an imported module is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            violations.append(f"line {node.lineno}: attribute {node.attr!r} is not allowed")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Attribute) and _root_name(sub) in module_names:
                        violations.append(
                            f"line {node.lineno}: assigning attributes of imported modules is not allowed")
                        break
        elif isinstance(node, ast.Global):
            violations.append(f"line {node.lineno}: 'global' statements are not allowed")
    return sorted(set(violations), key=lambda v: int(v.split(":")[0].split()[1]))


def lint_file(path) -> list:
    return lint_source(Path(path).read_text(encoding="utf-8"))

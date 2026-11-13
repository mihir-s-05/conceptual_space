import pytest

from cspace.archive import RecordInvalid
from cspace.box import descendants, unsupported_after_suspension, validate_space


def node(nid, supports=(), active=True, ntype="assumption"):
    return dict(id=nid, type=ntype, statement=nid, scope="small causal LMs", sources=[],
                status="established", active=active, supports=[list(g) for g in supports])


def space(*nodes):
    return dict(id="test", version=0, parent_version=None,
                context=dict(fixed_requirements=[], editable_priors=[], permitted_constructions=[]),
                nodes=list(nodes))


def test_valid_space_and_queries():
    s = space(node("A"), node("B"), node("D"), node("C", supports=[["A", "B"], ["D"]]), node("E", supports=[["C"]]))
    validate_space(s)
    assert descendants(s, "A") == {"C", "E"}
    # C keeps the alternative support D when A is suspended
    assert unsupported_after_suspension(s, {"A"}) == set()
    assert unsupported_after_suspension(s, {"A", "D"}) == {"C", "E"}


def test_cycle_and_unknown_reference_rejected():
    with pytest.raises(RecordInvalid, match="cycle"):
        validate_space(space(node("A", supports=[["B"]]), node("B", supports=[["A"]])))
    with pytest.raises(RecordInvalid, match="unknown"):
        validate_space(space(node("A", supports=[["Z"]])))
    with pytest.raises(RecordInvalid):
        validate_space(space(dict(node("A"), type="vibe")))

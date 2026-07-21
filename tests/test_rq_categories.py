from rip.agents.intelligence.rq_categories import _fold_assignments
from rip.core.models import RQCategory


def _cats():
    return [RQCategory(rq_id="RQ1", index=i, name=f"c{i}") for i in (1, 2)]


def test_fold_valid_and_garbage():
    cats = _cats()
    ids = {"a", "b", "c", "d"}
    parsed = {"assignments": [
        {"id": "a", "category": 1},
        {"id": "b", "category": 2},
        {"id": "c", "category": 0},        # explicit none — dropped
        {"id": "d", "category": 9},        # out of range — dropped
        {"id": "zzz", "category": 1},      # hallucinated id — dropped
        {"id": "a", "category": "1"},      # non-int — dropped
    ]}
    assert _fold_assignments(cats, ids, parsed) == 2
    assert cats[0].member_ids == ["a"]
    assert cats[1].member_ids == ["b"]


def test_fold_errored_chunk():
    cats = _cats()
    assert _fold_assignments(cats, {"a"}, None) == 0
    assert _fold_assignments(cats, {"a"}, {}) == 0
    assert all(not c.member_ids for c in cats)

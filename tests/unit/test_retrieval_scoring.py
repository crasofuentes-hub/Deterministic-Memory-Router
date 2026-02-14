from memory_router.core.index_store import IndexStore
from memory_router.core.retrieval import retrieve_topk, ScoreWeights

def test_retrieval_order_stable_tiebreak():
    s = IndexStore()
    now = 1_700_000_000
    a = s.agent("conversation")
    a.add("hello world", ts_unix=now-10, meta={})
    a.add("hello world", ts_unix=now-10, meta={})
    a.add("hello world", ts_unix=now-100, meta={})

    out = retrieve_topk("hello world", a.items, now_unix=now, topk=3)
    assert [x.stable_id for x in out] == [1, 2, 3]

def test_scoring_ranges_and_priority_present():
    s = IndexStore()
    now = 1_700_000_000
    p = s.agent("preferences")
    c = s.agent("conversation")

    p.add("I prefer Linux", ts_unix=now-1000, meta={})
    c.add("I prefer Linux", ts_unix=now-1, meta={})

    items = p.items + c.items
    out = retrieve_topk("prefer Linux", items, now_unix=now, topk=2, weights=ScoreWeights())
    assert all(0.0 <= x.priority <= 1.0 for x in out)
    assert all(0.0 <= x.score_global <= 1.0 for x in out)
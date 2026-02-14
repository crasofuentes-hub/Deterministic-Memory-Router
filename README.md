 # Deterministic Memory Router (5-Agent FAISS + Deterministic Fusion)
>>
>> A **deterministic multi-agent memory system** built for **bounded context**, **auditable recall**, and **reproducible behavior**.
>>
>> It shards conversation history across **five parallel memory agents**, each backed by a **local FAISS** store, then executes **parallel retrieval**, **threshold gating**, and a **deterministic fusion layer** to produce a compact **fused context** (budgeted to ~200 tokens).
>>
>> ---
>>
>> ## What this repo implements (your idea)
>>
>> ### 1) Five parallel agents, sharded by turn windows
>> - Turns **1–10 → agent1**
>> - Turns **11–20 → agent2**
>> - Turns **21–30 → agent3**
>> - Turns **31–40 → agent4**
>> - Turns **41–50 → agent5**
>> - Rolls every 50 turns.
>>
>> Each agent owns:
>> - `IndexFlatIP` FAISS index (inner product over L2-normalized vectors ≈ cosine similarity)
>> - deterministic JSONL metadata aligned to insertion order
>>
>> ### 2) Parallel query → gate → summarize
>> On a query:
>> - all 5 agents search top-K
>> - only agents with `best_similarity >= gate` produce a short summary (≤ 30 tokens)
>> - summaries are fused deterministically
>>
>> ### 3) Deterministic fusion (no voting)
>> - deduplicates lines deterministically
>> - resolves conflicts by **newest shard** (turn window end), then similarity
>>
>> ### 4) Strict budgets
>> - `topk_per_agent`
>> - `max_agent_summary_tokens`
>> - `max_recall_tokens` (default 200)
>>
>> ---
>>
>> ## Repo layout
>>
>> - `src/memory_router/core/faiss_store.py` — per-agent FAISS shard + JSONL meta
>> - `src/memory_router/core/multi_agent.py` — 5-agent router + gating + fusion + budgets
>> - `src/memory_router/cli_multi_agent.py` — `mem5` CLI
>> - `tests/unit/test_multi_agent_faiss_deterministic.py` — determinism tests
>>
>> ---
>>
>> ## Quickstart (Windows PowerShell)
>>
>> ```powershell
>> python -m venv .venv
>> .\.venv\Scripts\Activate.ps1
>> python -m pip install -U pip
>> pip install -e .
>> pytest -q

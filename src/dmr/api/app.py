from __future__ import annotations
import os, time, uuid
from typing import List
import redis
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from dmr.vectorize import DeterministicVectorizer
from dmr.index import FaissHNSWHotIndex
from dmr.storage import RedisHotStorage, SQLiteColdStore, ColdRow
from dmr.core.retrieval import DeterministicRetriever, RetrievalPolicy, EvidenceItem
from dmr.core.signatures import pack_signature, sha256_hex
from dmr.metrics import LAT, mark

class PreRequest(BaseModel):
    tenant_id: str
    user_id: str
    query: str

class EvidenceOut(BaseModel):
    turn_id: str
    signature: str
    score: float
    source: str
    text: str

class PreResponse(BaseModel):
    reliable: bool
    pack_signature: str
    evidence: List[EvidenceOut]
    evidence_block: str

class PostRequest(BaseModel):
    tenant_id: str
    user_id: str
    user_message: str
    assistant_message: str

class ForgetRequest(BaseModel):
    tenant_id: str
    user_id: str
    turn_id: str

def build_components():
    policy = RetrievalPolicy(
        threshold=float(os.environ.get("DMR_THRESHOLD","0.60")),
        k_final=int(os.environ.get("DMR_K_FINAL","5")),
        max_chars=int(os.environ.get("DMR_MAX_CHARS","1200")),
        k_hot_candidates=int(os.environ.get("DMR_K_HOT_CANDIDATES","20")),
        k_cold_candidates=int(os.environ.get("DMR_K_COLD_CANDIDATES","20")),
        budget_ms_hot=float(os.environ.get("DMR_BUDGET_MS_HOT","10")),
        budget_ms_cold=float(os.environ.get("DMR_BUDGET_MS_COLD","30")),
    )
    vectorizer = DeterministicVectorizer()
    dim = int(os.environ.get("DMR_VECTOR_DIM", str(vectorizer.dim)))

    redis_url = os.environ.get("DMR_REDIS_URL","redis://localhost:6379/0")
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    hot_storage = RedisHotStorage(r)

    faiss_dir = os.environ.get("DMR_FAISS_DIR","./dmr_faiss_hot")
    hot_index = FaissHNSWHotIndex(dim=dim, index_dir=faiss_dir, omp_threads=1)

    cold_path = os.environ.get("DMR_COLD_SQLITE","./dmr_cold.sqlite3")
    cold_store = SQLiteColdStore(path=cold_path)

    retriever = DeterministicRetriever(vectorizer, hot_index, hot_storage, cold_store, policy)
    return retriever, policy, vectorizer, hot_index, hot_storage, cold_store

retriever, policy, vectorizer, hot_index, hot_storage, cold_store = build_components()

app = FastAPI(
    title="Deterministic Memory Router (DMR)",
    description="Deterministic, offline memory layer. No cloud. No randomness. No prompt saturation.",
    version="2026.0.2",
)

def _format_block(ev: List[EvidenceItem]) -> str:
    if not ev:
        return ""
    parts = []
    for e in ev:
        parts.append(f"[{e.source.upper()}|{e.turn_id}|{e.signature}|score={e.score:.6f}]\\n{e.text}")
    return "\\n\\n---\\n\\n".join(parts)

@app.post("/pre", response_model=PreResponse)
def pre(req: PreRequest):
    mark("pre")
    t0 = time.perf_counter()
    try:
        ev = retriever.retrieve(req.tenant_id, req.user_id, req.query)
        policy_dict = {
            "threshold": policy.threshold,
            "k_final": policy.k_final,
            "max_chars": policy.max_chars,
            "budget_ms_hot": policy.budget_ms_hot,
            "budget_ms_cold": policy.budget_ms_cold,
        }
        sig = pack_signature(
            req.tenant_id, req.user_id, req.query, policy_dict,
            [(e.turn_id, e.signature, e.score, e.source) for e in ev]
        )
        return PreResponse(
            reliable=(len(ev) > 0),
            pack_signature=sig,
            evidence=[EvidenceOut(**e.__dict__) for e in ev],
            evidence_block=_format_block(ev),
        )
    finally:
        LAT.labels(endpoint="pre").observe((time.perf_counter() - t0) * 1000.0)

@app.post("/post")
def post(req: PostRequest):
    mark("post")
    t0 = time.perf_counter()
    try:
        user_key = f"{req.tenant_id}:{req.user_id}"
        turn_id = uuid.uuid4().hex[:16]
        text = f"Human: {req.user_message}\\nAI: {req.assistant_message}"
        signature = sha256_hex(f"{user_key}|{turn_id}|{text}")[:16]
        ts = time.time()

        v = vectorizer.text_to_vector(text).astype(np.float32)
        hot_index.add(user_key, v, persist=False)
        hot_storage.put_turn(user_key, turn_id, text, signature, ts)
        cold_store.put_many([ColdRow(req.tenant_id, req.user_id, turn_id, signature, ts, text)])

        return {"status":"ok","turn_id":turn_id,"signature":signature}
    finally:
        LAT.labels(endpoint="post").observe((time.perf_counter() - t0) * 1000.0)

@app.post("/forget")
def forget(req: ForgetRequest):
    mark("forget")
    user_key = f"{req.tenant_id}:{req.user_id}"
    ok = hot_storage.tombstone(user_key, req.turn_id)
    return {"status":"ok" if ok else "not_found","turn_id":req.turn_id}

@app.get("/health")
def health():
    return {"status":"ok","version":"2026.0.2"}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return app.response_class(content=data, media_type=CONTENT_TYPE_LATEST)
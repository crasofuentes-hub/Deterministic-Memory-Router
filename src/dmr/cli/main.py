from __future__ import annotations
import argparse, sys
from dmr.cli.doctor import run_doctor

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dmr", description="DMR vFinal 2026 - Enterprise CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="Run determinism/offline/no-saturation checks and emit signed report + cert")
    d.add_argument("--redis-url", default="redis://localhost:6379/0")
    d.add_argument("--cold-sqlite", default="./dmr_cold.sqlite3")
    d.add_argument("--faiss-dir", default="./dmr_faiss_hot")
    d.add_argument("--tenant-id", default="T")
    d.add_argument("--user-id", default="U")
    d.add_argument("--vector-dim", type=int, default=20)
    d.add_argument("--report-out", default="./dmr_doctor_report.json")
    d.add_argument("--report-md", default="./dmr_doctor_report.md")
    d.add_argument("--cert-md", default="./DMR_COMPLIANCE_CERT.md")
    d.add_argument("--runs", type=int, default=50)
    d.add_argument("--strict", action="store_true")
    return p

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    if args.cmd == "doctor":
        return run_doctor(
            redis_url=args.redis_url,
            cold_sqlite=args.cold_sqlite,
            faiss_dir=args.faiss_dir,
            tenant_id=args.tenant_id,
            user_id=args.user_id,
            vector_dim=args.vector_dim,
            report_out=args.report_out,
            report_md=args.report_md,
            cert_md=args.cert_md,
            runs=args.runs,
            strict=args.strict,
        )
    return 2
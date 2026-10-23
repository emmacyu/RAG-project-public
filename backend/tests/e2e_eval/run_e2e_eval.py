"""End-to-end evaluation of the full agent: dispatcher -> rewrite -> retrieval -> generation -> auditor.

Runs every question of tests/retrieval_eval/e23_golden.json plus the follow-up conversations in
conversations.json through the compiled LangGraph agent (in-memory checkpointer and Chroma, so no
Postgres, Redis or Docker services are needed) and scores the answers.

  --dry-run   replaces the LLM with canned replies. Retrieval, reranking and prompts are real, so it
              checks the harness and estimates the token volume of a real run without spending money.
  (default)   calls the configured LLM through OpenRouter and, unless --no-judge, an LLM judge.

Deterministic metrics: refusals of unanswerable questions, false refusals, citation pages, evidence
in the retrieved context, LLM calls, latency. Judge metrics: correctness against the evidence and
faithfulness to the retrieved context.

Run from backend/:
    python tests/e2e_eval/run_e2e_eval.py --dry-run
    OPENROUTER_API_KEY=... python tests/e2e_eval/run_e2e_eval.py --judge-model openai/gpt-4o --out results.jsonl
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parents[1]
sys.path.insert(0, str(BACKEND_DIR))
GOLDEN = BACKEND_DIR / "tests" / "retrieval_eval" / "e23_golden.json"
CONVERSATIONS = HERE / "conversations.json"

NOT_FOUND = "cannot find related content"
CITATION = re.compile(r"\[Source:([^\]]*)\]", re.I)


def normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def cited_pages(answer: str) -> set[int]:
    pages = set()
    for group in CITATION.findall(answer):
        for a, b in re.findall(r"(\d+)(?:\s*-\s*(\d+))?", group):
            pages.update(range(int(a), int(b or a) + 1))
    return pages


def node_kind(messages) -> str:
    text = " ".join(str(m.content) for m in messages)
    if "intent recognizer" in text:
        return "dispatch"
    if "search query optimizer" in text:
        return "rewrite"
    if "audit judge criticized" in text:
        return "expand"
    if "structured risk control system" in text:
        return "audit"
    return "generate"


class RecordingLLM:
    """Wraps the agent's LLM: records every call (node, characters in/out, tokens, latency)."""

    def __init__(self, inner, log, dry_run):
        self.inner, self.log, self.dry_run = inner, log, dry_run

    def bind(self, **kwargs):
        return RecordingLLM(self.inner if self.dry_run else self.inner.bind(**kwargs), self.log, self.dry_run)

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        kind = node_kind(messages)
        started = time.perf_counter()
        if self.dry_run:
            reply = AIMessage(content={
                "dispatch": "AUDIT",
                "rewrite": messages[-1].content.split("Current question:")[-1].strip(),
                "expand": "expanded query",
                "audit": json.dumps({"is_hallucinating": False, "feedback": ""}),
                "generate": "Dry-run answer with a typical length of about sixty words. " * 6 + "[Source: Page 1]",
            }[kind])
        else:
            reply = self.inner.invoke(messages)
        usage = getattr(reply, "usage_metadata", None) or {}
        self.log.append({
            "node": kind,
            "chars_in": sum(len(str(m.content)) for m in messages),
            "chars_out": len(str(reply.content)),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "ms": round((time.perf_counter() - started) * 1000),
        })
        return reply


JUDGE_PROMPT = """You are grading an answer produced by a document question-answering system.

Question: {question}

Reference passage from the source document (the answer should convey this information):
{evidence}

Context the system retrieved and was allowed to use:
{context}

System answer:
{answer}

Return a JSON object with:
"correct": "yes" if the answer conveys the reference information, "partial" if only part of it, "no" otherwise.
"faithful": true if every factual claim in the answer is supported by the retrieved context, else false.
"unsupported": a short list of claims that are not supported by the retrieved context (empty if none)."""

MULTI_JUDGE_PROMPT = """You are grading an answer produced by a document question-answering system.

Question: {question}

Reference points from the source document, numbered:
{points}

Context the system retrieved and was allowed to use:
{context}

System answer:
{answer}

Return a JSON object with:
"points_covered": the list of reference point numbers the answer conveys.
"faithful": true if every factual claim in the answer is supported by the retrieved context, else false.
"unsupported": a short list of claims that are not supported by the retrieved context (empty if none)."""


def judge(judge_llm, item, record):
    from langchain_core.messages import HumanMessage

    if item["kind"] == "multi":
        prompt = MULTI_JUDGE_PROMPT.format(
            question=item["question"], context=record["context"] or "(nothing)", answer=record["answer"],
            points="\n".join(f"{i + 1}. {e}" for i, e in enumerate(item["evidence"])))
    else:
        prompt = JUDGE_PROMPT.format(question=item["question"], evidence=item["evidence"],
                                     context=record["context"] or "(nothing)", answer=record["answer"])
    reply = judge_llm.invoke([HumanMessage(content=prompt)])
    try:
        return json.loads(reply.content)
    except Exception:
        return {"error": "judge returned invalid JSON", "raw": reply.content[:300]}


def load_items():
    golden = json.loads(GOLDEN.read_text())
    items = [{"kind": "single", "id": q["id"], "turns": [q["question"]], "question": q["question"],
              "evidence": q["evidence"], "page": q["page"]} for q in golden["questions"]]
    items += [{"kind": "multi", "id": q["id"], "turns": [q["question"]], "question": q["question"],
               "evidence": q["evidence"]} for q in golden["multi"]]
    items += [{"kind": "negative", "id": q["id"], "turns": [q["question"]], "question": q["question"]}
              for q in golden["negatives"]]
    for c in json.loads(CONVERSATIONS.read_text())["conversations"]:
        items.append({"kind": "followup", "id": c["id"], "turns": c["turns"], "question": c["turns"][-1],
                      "evidence": c["evidence"], "page": c["page"]})
    return golden, items


def setup_agent(pdf: str, dry_run: bool, log: list):
    os.environ.setdefault("DATABASE_URL", "postgresql://unused")
    os.environ.setdefault("REDIS_URL", "redis://unused")
    if dry_run:
        os.environ.setdefault("OPENROUTER_API_KEY", "dry-run")
    elif not os.environ.get("OPENROUTER_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv(BACKEND_DIR.parent / ".env")
        if not os.environ.get("OPENROUTER_API_KEY"):
            sys.exit("OPENROUTER_API_KEY is not set (environment or .env). Use --dry-run to test the harness.")

    import chromadb
    from langgraph.checkpoint.memory import MemorySaver
    import app.services.agent as agent
    import app.services.vector_store as vs

    client = chromadb.EphemeralClient()
    vs.get_chroma_client = lambda: client
    vs.COLLECTION_NAME = f"e2e-{uuid.uuid4().hex[:8]}"
    vs.vector_store_instance = None
    vs.retriever_instance = None
    vs.process_pdf(pdf)

    agent.llm = RecordingLLM(agent.llm, log, dry_run)
    return agent.agent_builder.compile(checkpointer=MemorySaver())


def run_item(app, item, log):
    from langchain_core.messages import HumanMessage

    thread = {"configurable": {"thread_id": f"{item['id']}-{uuid.uuid4().hex[:6]}"}}
    for turn in item["turns"]:
        start = len(log)
        started = time.perf_counter()
        state = app.invoke({"messages": [HumanMessage(content=turn)]}, config=thread)
        elapsed = round((time.perf_counter() - started) * 1000)
        calls = log[start:]
    answer = state["messages"][-1].content
    return {
        "id": item["id"], "kind": item["kind"], "question": item["question"], "answer": answer,
        "context": state.get("current_context", ""), "intent": state.get("intent"),
        "retry_count": state.get("retry_count", 0), "flagged_at_end": bool(state.get("is_hallucinating")),
        "refused": NOT_FOUND in normalise(answer), "cited_pages": sorted(cited_pages(answer)),
        "llm_calls": [c["node"] for c in calls], "last_turn_ms": elapsed,
        "chars_in": sum(c["chars_in"] for c in calls), "chars_out": sum(c["chars_out"] for c in calls),
        "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
        "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
    }


def summarise(items, records, dry_run):
    by_id = {i["id"]: i for i in items}
    rows = []
    for kind in ("single", "multi", "followup", "negative"):
        recs = [r for r in records if r["kind"] == kind]
        if not recs:
            continue
        row = {"set": kind, "n": len(recs)}
        if kind == "negative":
            row["refused"] = f"{sum(r['refused'] for r in recs)}/{len(recs)}"
            row["refused_without_llm_answer"] = sum(r["refused"] and "generate" not in r["llm_calls"] for r in recs)
        else:
            row["false_refusals"] = sum(r["refused"] for r in recs)
            if kind == "multi":
                row["evidence_in_context"] = round(statistics.mean(
                    statistics.mean(normalise(e) in normalise(r["context"]) for e in by_id[r["id"]]["evidence"])
                    for r in recs), 3)
            else:
                row["evidence_in_context"] = round(statistics.mean(
                    normalise(by_id[r["id"]]["evidence"]) in normalise(r["context"]) for r in recs), 3)
                if not dry_run:
                    cited = [r for r in recs if r["cited_pages"]]
                    row["answers_with_citation"] = f"{len(cited)}/{len(recs)}"
                    row["citation_page_correct"] = f"{sum(by_id[r['id']]['page'] in r['cited_pages'] for r in cited)}/{len(cited)}"
            judged = [r["judge"] for r in recs if "judge" in r and "error" not in r["judge"]]
            if judged and kind == "multi":
                row["judge_points_covered"] = round(statistics.mean(
                    len(j.get("points_covered", [])) / len(by_id[r["id"]]["evidence"])
                    for r, j in zip([r for r in recs if "judge" in r and "error" not in r["judge"]], judged)), 3)
            elif judged:
                row["judge_correct"] = f"{sum(j.get('correct') == 'yes' for j in judged)}/{len(judged)}"
                row["judge_partial"] = sum(j.get("correct") == "partial" for j in judged)
            if judged:
                row["judge_faithful"] = f"{sum(bool(j.get('faithful')) for j in judged)}/{len(judged)}"
        row["llm_calls_avg"] = round(statistics.mean(len(r["llm_calls"]) for r in recs), 2)
        row["retries"] = sum(r["retry_count"] for r in recs)
        row["ms_median"] = round(statistics.median(r["last_turn_ms"] for r in recs))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-model", default=None, help="OpenRouter model id; defaults to LLM_MODEL")
    parser.add_argument("--only", help="comma-separated item ids, e.g. q01,n03,c02")
    parser.add_argument("--out", help="write one JSON record per item to this file")
    args = parser.parse_args()

    golden, items = load_items()
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i["id"] in wanted]

    log: list = []
    app = setup_agent(str(BACKEND_DIR / golden["document"]), args.dry_run, log)

    judge_llm = None
    if not args.dry_run and not args.no_judge:
        from langchain_openai import ChatOpenAI
        from app.config import settings
        judge_llm = ChatOpenAI(model=args.judge_model or settings.LLM_MODEL, temperature=0,
                               api_key=settings.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"
                               ).bind(response_format={"type": "json_object"})

    records = []
    for n, item in enumerate(items, 1):
        record = run_item(app, item, log)
        if judge_llm and item["kind"] != "negative" and not record["refused"]:
            record["judge"] = judge(judge_llm, item, record)
        records.append(record)
        print(f"[{n}/{len(items)}] {item['id']} calls={record['llm_calls']} refused={record['refused']}", file=sys.stderr)

    rows = summarise(items, records, args.dry_run)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")

    calls = len(log)
    chars_in, chars_out = sum(c["chars_in"] for c in log), sum(c["chars_out"] for c in log)
    print(f"\nAgent LLM calls: {calls}  (by node: "
          + ", ".join(f"{k}={sum(c['node'] == k for c in log)}" for k in ("dispatch", "rewrite", "expand", "generate", "audit"))
          + ")")
    print(f"Prompt size: {chars_in:,} characters in, {chars_out:,} out (about {chars_in // 4:,} / {chars_out // 4:,} tokens)")
    if args.dry_run:
        answered = sum(1 for r in records if r["kind"] != "negative" and not r["refused"])
        print(f"A real run adds about {answered} judge calls. Retries are not simulated: each rejected draft "
              f"adds one expand, one generate and one audit call (at most two retries per question).")

    if args.out:
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

"""Multi-turn tests of the compiled LangGraph agent, with a scripted fake LLM (no API calls, no database).

They exercise the real graph wiring and node code; only the LLM and the retriever are replaced.
"""
import json
import os
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

backend_dir = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(backend_dir))


class FakeLLM:
    """Answers each node's prompt with a canned reply and records which node called it."""

    def __init__(self):
        self.calls: list[str] = []
        self.verdicts: list[bool] = []   # auditor answers, consumed in order; default is "no hallucination"

    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        text = " ".join(str(m.content) for m in messages)
        if "intent recognizer" in text:
            kind, out = "dispatch", "AUDIT"
        elif "search query optimizer" in text:
            kind, out = "rewrite", "standalone rewritten query"
        elif "audit judge criticized" in text:
            kind, out = "expand", "expanded query"
        elif "structured risk control system" in text:
            verdict = self.verdicts.pop(0) if self.verdicts else False
            kind, out = "audit", json.dumps({"is_hallucinating": verdict, "feedback": "unsupported claim"})
        else:
            kind, out = "generate", "Model owners coordinate development [Source: Page 5]."
        self.calls.append(kind)
        return AIMessage(content=out)


class FakeRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, k):
        self.queries.append(query)
        return [Document(page_content="Model owners coordinate development.", metadata={"page": 5, "page_end": 5})]


@pytest.fixture()
def agent_app(monkeypatch):
    for key, value in {"OPENROUTER_API_KEY": "test", "DATABASE_URL": "postgresql://x", "REDIS_URL": "redis://x"}.items():
        monkeypatch.setenv(key, os.environ.get(key, value))
    import app.services.agent as agent
    from langgraph.checkpoint.memory import MemorySaver

    llm, retriever = FakeLLM(), FakeRetriever()
    monkeypatch.setattr(agent, "llm", llm)
    monkeypatch.setattr(agent, "get_retriever", lambda: retriever)
    app = agent.agent_builder.compile(checkpointer=MemorySaver())
    return app, llm, retriever


def ask(app, text, thread="t1"):
    return app.invoke({"messages": [HumanMessage(content=text)]}, config={"configurable": {"thread_id": thread}})


def test_circuit_breaker_stops_after_three_attempts(agent_app):
    app, llm, _ = agent_app
    llm.verdicts = [True, True, True, True]
    state = ask(app, "Who coordinates model development?")
    assert llm.calls.count("generate") == 3
    assert state["retry_count"] == 3


def test_a_new_question_starts_with_a_clean_retry_state(agent_app):
    app, llm, retriever = agent_app

    # Turn 1: every draft is rejected, so the circuit breaker trips
    llm.verdicts = [True, True, True]
    ask(app, "Who coordinates model development?")

    # Turn 2: a follow-up with a pronoun; the first draft is rejected once
    llm.calls.clear()
    retriever.queries.clear()
    llm.verdicts = [True, False]
    state = ask(app, "What else are they responsible for?")

    assert llm.calls[:2] == ["dispatch", "rewrite"], "the follow-up must be rewritten into a standalone query"
    assert retriever.queries[0] == "standalone rewritten query"
    assert llm.calls.count("generate") == 2, "the rejected draft must get a retry in the new turn"
    assert state["retry_count"] == 1
    assert state["audit_trail"] == ["unsupported claim"], "feedback from the previous question must not leak in"

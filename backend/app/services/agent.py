import os
import json
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.postgres import PostgresSaver

from app.services.vector_store import get_retriever
from app.services.database import get_db_pool
from app.config import settings
from app.prompts import (
    DISPATCHER_PROMPT,
    REWRITE_PROMPT,
    EXPAND_PROMPT,
    CHAT_PROMPT,
    AUDIT_PROMPT,
    HALLUCINATION_PROMPT,
    HALLUCINATION_SYS_MSG
)

# Define universal LLM engines with built-in failover capabilities
llm_primary = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=0,  # Force generation to be highly robust and deterministic for risk auditing
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

llm_fallback = ChatOpenAI(
    model=settings.FALLBACK_LLM_MODEL,
    temperature=0,
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# LangChain will automatically intercept 429/500/Timeout errors and seamlessly route to the fallback model
llm = llm_primary.with_fallbacks([llm_fallback])

# =========================================================================
# 1. Core Agent State - handling storage, memory, and serving as system's diagnostic black box
# =========================================================================
class AgentState(TypedDict):
    # Leverage LangGraph's built-in Reducer to automatically append new messages
    messages: Annotated[list[AnyMessage], add_messages]
    
    # Traceability fields for industrial-grade auditing:
    intent: str                  # User intent classified by dispatcher: 'chat' or 'audit'
    current_context: str         # Raw document context relied on for answers (used by logic auditor to catch hallucinations)
    audit_trail: list[str]       # A history log of rejection reasons from the logic auditor
    retry_count: int             # Loop restart attempts (used for infinite loop circuit-breaking)
    is_hallucinating: bool       # Flags if the current output hallucinates

# =========================================================================
# 2. Four Core Processing Nodes
# =========================================================================

def dispatcher_node(state: AgentState):
    """Dispatcher Node: Combine conversation history to identify the true user intent."""
    messages = state["messages"]
    last_msg = messages[-1].content
    
    # Extract recent context to assist intention recognition
    history = "\n".join([f"{m.type}: {m.content}" for m in messages[-3:-1]]) if len(messages) > 1 else "None"
    
    sys_msg = SystemMessage(content=DISPATCHER_PROMPT.format(history=history))
    
    resp = llm.invoke([sys_msg, HumanMessage(content=last_msg)])
    
    intent = "chat" if "CHAT" in resp.content.upper() else "audit"
    
    # Every user message starts a new question, so its retry state starts fresh. Carrying the previous
    # turn's values over would skip the follow-up rewrite (it only runs when retry_count is 0), reuse
    # stale auditor feedback, and leave the circuit breaker tripped for the rest of the conversation.
    # Earlier values are not lost: the checkpointer keeps every step of the thread in Postgres.
    return {
        "intent": intent,
        "retry_count": 0,
        "audit_trail": [],
        "current_context": "",
    }

# Same wording the audit prompt asks for, so answers look the same whichever path produced them
NO_RELEVANT_CONTENT = "Cannot find related content in the database."

def _page_ref(metadata: dict) -> str:
    """'4', or '12-13' for a chunk that runs across a page break."""
    start = metadata.get("page", "Unknown")
    end = metadata.get("page_end", start)
    return f"{start}-{end}" if end != start else f"{start}"

def retrieve_node(state: AgentState):
    """Retriever Node: Equipped with Query Expansion & Contextual Rewrite capabilities."""
    messages = state["messages"]
    last_msg = messages[-1].content
    retriever = get_retriever()
    
    if not retriever:
        return {"current_context": "The system hasn't mounted the vector database. Please prompt the user to upload documents."}

    query = last_msg
    
    # 1st Advanced Mechanism: Contextual Query Rewrite (Resolving Pronouns)
    if len(messages) > 1 and state.get("retry_count", 0) == 0:
        history_text = "\n".join([f"{m.type}: {m.content}" for m in messages[-4:-1]])
        rewrite_prompt = REWRITE_PROMPT.format(history_text=history_text, last_msg=last_msg)
        rewrite_resp = llm.invoke([HumanMessage(content=rewrite_prompt)])
        query = rewrite_resp.content.strip()
        print(f"🔄 [Context Rewrite] '{last_msg}' -> '{query}'")

    # 2nd Advanced Mechanism: Query Expansion based on Auditor Rejections
    if state.get("retry_count", 0) > 0 and state.get("audit_trail"):
        feedback = state["audit_trail"][-1]
        expand_prompt = EXPAND_PROMPT.format(query=query, feedback=feedback)
        expand_resp = llm.invoke([HumanMessage(content=expand_prompt)])
        query = expand_resp.content.strip()
        print(f"🔄 [Correction Rewrite] Fusing penalizing feedback -> '{query}'")
    
    # Dense + BM25 candidates, fused and reranked by a cross-encoder; only the best few reach the LLM.
    # (BM25 covers exact identifiers such as "B-10" that dense embeddings tend to miss.)
    # The reranker scores also decide how many chunks to pass, and return none when nothing is relevant.
    docs = retriever.search(query, k=settings.RETRIEVAL_TOP_K)
    print(f"📚 [Retrieval] {len(docs)} chunk(s) passed the relevance cutoff")
    
    # Assemble the found documents and their metadata into pure text context
    context_str = "\n\n".join([
        f"[Source File Page: {_page_ref(d.metadata)} ] Content Snippet:\n{d.page_content}"
        for d in docs
    ])
    
    return {"current_context": context_str}

def generate_node(state: AgentState):
    """Generator Node: Strictly anchor to source documents while relaxing constraints for casual chat/explanations."""
    context = state.get("current_context", "")
    intent = state.get("intent", "chat")
    
    if intent == "chat":
        sys_prompt = CHAT_PROMPT
    elif not context:
        # Retrieval found nothing relevant: answer directly instead of asking the LLM to work from nothing
        return {"messages": [AIMessage(content=NO_RELEVANT_CONTENT)]}
    else:
        sys_prompt = AUDIT_PROMPT.format(context=context if context else 'No reference materials')
    
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response = llm.invoke(messages)
    
    # Return a list here to trigger the state's add_messages Reducer to append them
    return {"messages": [response]} 

def auditor_node(state: AgentState):
    """Logic Auditor Node: Responsible for catching LLM hallucinations."""
    # Let casual chats pass through
    if state.get("intent") == "chat":
        return {"is_hallucinating": False}
    # Nothing was retrieved, so the answer is the fixed "not found" reply: nothing to check
    if not state.get("current_context"):
        return {"is_hallucinating": False}
        
    last_aimessage = state["messages"][-1].content
    context = state.get("current_context", "")
    
    prompt = HALLUCINATION_PROMPT.format(
        context=context,
        last_aimessage=last_aimessage
    )
    
    # Force the LLM to return a strict JSON Object
    eval_llm = llm.bind(response_format={"type": "json_object"})
    resp = eval_llm.invoke([
        SystemMessage(content=HALLUCINATION_SYS_MSG), 
        HumanMessage(content=prompt)
    ])
    
    try:
        data = json.loads(resp.content)
        is_hallucinating = data.get("is_hallucinating", False)
        feedback = data.get("feedback", "")
    except Exception:
        # Safety fallback: If JSON parsing fails, approve to prevent application deadlocks
        is_hallucinating = False
        feedback = "Parsing anomaly encountered, skipping validation"
        
    if is_hallucinating:
        trail = list(state.get("audit_trail", []))
        trail.append(feedback)
        rc = state.get("retry_count", 0) + 1
        return {"is_hallucinating": True, "audit_trail": trail, "retry_count": rc}
    else:
        return {"is_hallucinating": False}


# =========================================================================
# 3. Pipeline Definitions: Wiring dynamic routing and recursive loops (Edges and Compilation)
# =========================================================================

def route_after_dispatcher(state: AgentState):
    """Dynamic pipeline dispatching based on intent."""
    if state.get("intent") == "audit":
        return "retrieve_node"
    return "generate_node"

def route_after_auditor(state: AgentState):
    """Core self-reflection and rejection loop."""
    # If hallucinations exist and retries < 3, route back to the retriever (uses audit trace to search better)
    if state.get("is_hallucinating") and state.get("retry_count", 0) < 3:
        return "retrieve_node"
    # If passes, or circuit breaker hits 3 retries, route to the END phase
    return END

def get_agent_app():
    builder = StateGraph(AgentState)
    
    # Register the four core computational nodes
    builder.add_node("dispatcher", dispatcher_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("generate_node", generate_node)
    builder.add_node("auditor_node", auditor_node)
    
    # Wiring: START -> Dispatcher
    builder.add_edge(START, "dispatcher")
    
    # Dispatcher -> Fork (Retrieve vs Chat Generate)
    builder.add_conditional_edges(
        "dispatcher", 
        route_after_dispatcher,
        {
            "retrieve_node": "retrieve_node",
            "generate_node": "generate_node"
        }
    )
    
    # Retrieve -> Generate
    builder.add_edge("retrieve_node", "generate_node")
    
    # Generate -> Logic Auditor
    builder.add_edge("generate_node", "auditor_node")
    
    # Auditor -> Loop back to Retrieve or END
    builder.add_conditional_edges(
        "auditor_node",
        route_after_auditor,
        {
            "retrieve_node": "retrieve_node",
            END: END
        }
    )
    
    # Avoid forcing database hookups here; instead, expose the raw architecture blueprint (builder)
    # Allowing FastAPI or test scripts to hook up real databases inside their own Async Event Loops
    return builder

# Expose the architectural blueprint
agent_builder = get_agent_app()

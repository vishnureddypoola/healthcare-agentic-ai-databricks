"""Clinical Data Dictionary — conversational UI (Streamlit on Databricks Apps).

LangChain/LangGraph tool-calling agent whose ONLY tools come from the custom MCP server.
The agent never sees raw SQL access — it can only call the curated, governed dictionary tools.
"""
import asyncio
import os
import time

import pandas as pd

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, HumanMessage, convert_to_messages, convert_to_openai_messages

from eval_suite import EVAL_SET, score

MCP_URL = os.environ["MCP_SERVER_URL"]
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-5-5")
USE_OBO = os.getenv("USE_OBO", "false").lower() == "true"

SYSTEM_PROMPT = """You are the data dictionary assistant for a healthcare lakehouse (FHIR clinical data, X12 claims,
clinical trials, prior authorization). Answer ONLY from tool results — never guess table or column names.
- To find where something lives, call search_data_dictionary first, then describe_table.
- Always mention whether a column is PHI and which code system it uses (ICD-10, LOINC, RxNorm, CPT, CARC...).
- When asked for example data, use get_sample_rows (identifiers are masked) and remind the user it is synthetic.
- If the user asks for something outside the catalog, say so plainly.
Format: short answer first, then a compact table of the relevant columns."""


def extract_text(content) -> str:
    """Return only the user-visible text from a chat message (skips reasoning blocks)."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") in ("text", "output_text"):
            parts.append(b.get("text", ""))
    return "".join(parts)


def workspace_client() -> WorkspaceClient:
    if USE_OBO:
        token = st.context.headers.get("x-forwarded-access-token")
        if token:
            return WorkspaceClient(host=WorkspaceClient().config.host, token=token, auth_type="pat")
    return WorkspaceClient()  # app service principal


@wrap_model_call
async def convert_mcp_content(request, handler):
    # Normalise MCP tool-result content blocks into plain chat messages for the Databricks chat model.
    messages = convert_to_messages(convert_to_openai_messages(request.messages))
    return await handler(request.override(messages=messages))


async def run_agent(history: list[dict]):
    client = DatabricksMultiServerMCPClient([
        DatabricksMCPServer(name="dictionary", url=MCP_URL, workspace_client=workspace_client()),
    ])
    tools = await client.get_tools()
    agent = create_agent(
        model=ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0, max_tokens=4096),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[convert_mcp_content],
    )
    msgs = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in history]
    result = await agent.ainvoke({"messages": msgs})
    if not extract_text(result["messages"][-1].content).strip():
        # GPT-OSS occasionally ends a tool loop with an empty message; nudge it once to write the answer.
        nudge = result["messages"] + [HumanMessage("Using the tool results above, write the final answer now.")]
        result = await agent.ainvoke({"messages": nudge})
    return result


st.set_page_config(page_title="Clinical Data Dictionary", page_icon="📘", layout="wide")
st.title("📘 Clinical Data Dictionary")
st.caption("Ask about any table or column in the hc_agentic catalog. Synthetic data only.")

with st.sidebar:
    st.markdown("**Try asking**")
    for q in ["Where is HbA1c stored and what code identifies it?",
              "Which columns in the clinical schema are PHI?",
              "Explain the claim_line table to a new analyst",
              "Which table tells me why a claim was denied?"]:
        if st.button(q, use_container_width=True):
            st.session_state.setdefault("history", []).append({"role": "user", "content": q})

    st.divider()
    st.markdown("**🧪 Evaluation (Part 7)**")
    eval_runs = st.slider("Runs per question", 1, 5, 3)
    run_eval = st.button(f"Run eval suite ({len(EVAL_SET)} questions)", use_container_width=True)


def run_eval_suite(runs: int) -> pd.DataFrame:
    rows, total = [], len(EVAL_SET) * runs
    bar = st.progress(0.0, text="Running eval…")
    for case in EVAL_SET:
        for r in range(1, runs + 1):
            t0 = time.time()
            try:
                res = asyncio.run(run_agent([{"role": "user", "content": case["question"]}]))
                answer = extract_text(res["messages"][-1].content)
                calls = [tc["name"] for m in res["messages"] if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
                passed, reason = score(case, answer)
            except Exception as e:  # a crash is a failed run, not a crashed eval
                answer, calls, passed, reason = "", [], False, f"error: {type(e).__name__}"
            rows.append({"case": case["id"], "category": case["category"], "run": r, "passed": passed,
                         "reason": reason, "tool_calls": len(calls), "tools": " > ".join(calls),
                         "latency_s": round(time.time() - t0, 1), "answer_preview": answer[:200]})
            bar.progress(len(rows) / total, text=f"{len(rows)}/{total} runs")
    bar.empty()
    return pd.DataFrame(rows)


if run_eval:
    st.subheader("🧪 Eval results")
    df = run_eval_suite(eval_runs)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{df.passed.mean():.0%}", f"{int(df.passed.sum())}/{len(df)} runs")
    c2.metric("Avg tool calls", f"{df.tool_calls.mean():.1f}", f"max {df.tool_calls.max()}")
    c3.metric("Avg latency", f"{df.latency_s.mean():.1f}s", f"p95 {df.latency_s.quantile(0.95):.1f}s")
    c4.metric("Empty/errors", int(df.reason.str.startswith(("empty", "error")).sum()))
    per_case = (df.groupby(["case", "category"])
                  .agg(pass_rate=("passed", "mean"), avg_tools=("tool_calls", "mean"),
                       max_tools=("tool_calls", "max"), avg_latency_s=("latency_s", "mean"))
                  .reset_index())
    per_case["pass_rate"] = (per_case.pass_rate * 100).round().astype(int).astype(str) + "%"
    st.dataframe(per_case.round(1), use_container_width=True)
    with st.expander("All runs (with failure reasons)"):
        st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download results CSV", df.to_csv(index=False), "p1_eval_results.csv", "text/csv")
    st.stop()

st.session_state.setdefault("history", [])
if prompt := st.chat_input("Ask about the data…"):
    st.session_state.history.append({"role": "user", "content": prompt})

for m in st.session_state.history:
    st.chat_message(m["role"]).markdown(m["content"])

if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Looking it up in Unity Catalog…"):
            result = asyncio.run(run_agent(st.session_state.history))
        last = result["messages"][-1]
        answer = extract_text(last.content)
        if answer.strip():
            st.markdown(answer)
        else:
            # Reasoning models (e.g. GPT-OSS) can spend the whole budget "thinking" and return no visible text.
            st.warning("The model returned no answer text. Raw output below — try asking again or rephrasing.")
            with st.expander("Debug: full agent trace"):
                for i, msg in enumerate(result["messages"]):
                    usage = (getattr(msg, "usage_metadata", None) or {})
                    st.code(
                        f"[{i}] {type(msg).__name__}"
                        f" | tool_calls={[tc['name'] for tc in (getattr(msg, 'tool_calls', None) or [])]}"
                        f" | tokens in/out={usage.get('input_tokens')}/{usage.get('output_tokens')}\n"
                        f"{repr(msg.content)[:600]}"
                    )
            answer = "_(no answer text returned)_"
        calls = [tc for msg in result["messages"] if isinstance(msg, AIMessage) for tc in (msg.tool_calls or [])]
        if calls:
            with st.expander(f"🔧 {len(calls)} tool call(s)"):
                for tc in calls:
                    st.code(f"{tc['name']}({tc['args']})")
    st.session_state.history.append({"role": "assistant", "content": answer})
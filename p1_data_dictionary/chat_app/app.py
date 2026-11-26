"""Clinical Data Dictionary — conversational UI (Streamlit on Databricks Apps).

LangChain/LangGraph tool-calling agent whose ONLY tools come from the custom MCP server.
The agent never sees raw SQL access — it can only call the curated, governed dictionary tools.
"""
import asyncio
import os

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, HumanMessage, convert_to_messages, convert_to_openai_messages

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
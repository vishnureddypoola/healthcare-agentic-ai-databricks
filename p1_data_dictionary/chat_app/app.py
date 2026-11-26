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
Tool use (be efficient, usually 2-3 calls):
- To find where something lives: call search_data_dictionary ONCE, then describe_table ONCE for the best match.
- Do not repeat a tool call with the same arguments. Stop calling tools once you can answer.
- For example data, use get_sample_rows (identifiers are masked) and say the data is synthetic.
Governance — keep these two ideas separate, exactly as the steward recorded them:
- "HIPAA identifier" = the column's phi_identifier value (e.g. NAME, DATE, OTHER_UNIQUE_ID) or "None".
- "Sensitivity" = the column's sensitivity value (PUBLIC, INTERNAL, CONFIDENTIAL, PHI).
  A lab value can be sensitive health information without being a Safe Harbor identifier.
If the user asks for something outside the catalog, say so plainly.
Format: short answer first, then a compact table with columns: Column | HIPAA identifier | Sensitivity | Code system | Description."""


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
        model=ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[convert_mcp_content],
    )
    msgs = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in history]
    return await agent.ainvoke({"messages": msgs})


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
        answer = result["messages"][-1].content
        if isinstance(answer, list):  # some models return content blocks
            answer = "".join(b.get("text", "") for b in answer if isinstance(b, dict))
        st.markdown(answer)
        calls = [tc for msg in result["messages"] if isinstance(msg, AIMessage) for tc in (msg.tool_calls or [])]
        if calls:
            with st.expander(f"🔧 {len(calls)} tool call(s)"):
                for tc in calls:
                    st.code(f"{tc['name']}({tc['args']})")
    st.session_state.history.append({"role": "assistant", "content": answer})
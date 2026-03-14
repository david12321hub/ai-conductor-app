import streamlit as st
import anthropic
import os
from google import genai
from google.genai import types

st.set_page_config(page_title="AI Conductor", page_icon="🎼", layout="centered")
st.title("🎼 Your Personal AI Conductor")
st.caption("One question → Multiple AIs → Best plan + execution")


def check_env_vars():
    missing = []
    for var in [
        "AI_INTEGRATIONS_ANTHROPIC_BASE_URL",
        "AI_INTEGRATIONS_ANTHROPIC_API_KEY",
        "AI_INTEGRATIONS_GEMINI_BASE_URL",
        "AI_INTEGRATIONS_GEMINI_API_KEY",
    ]:
        if not os.environ.get(var):
            missing.append(var)
    return missing


@st.cache_resource
def get_anthropic_client():
    return anthropic.Anthropic(
        base_url=os.environ["AI_INTEGRATIONS_ANTHROPIC_BASE_URL"],
        api_key=os.environ["AI_INTEGRATIONS_ANTHROPIC_API_KEY"],
    )


@st.cache_resource
def get_gemini_client():
    return genai.Client(
        api_key=os.environ["AI_INTEGRATIONS_GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            base_url=os.environ["AI_INTEGRATIONS_GEMINI_BASE_URL"],
            api_version="",
        ),
    )


missing_vars = check_env_vars()
if missing_vars:
    st.error(
        f"Missing environment variables: {', '.join(missing_vars)}. "
        "Please make sure the AI integrations are configured."
    )
    st.stop()

CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-flash"


def ask_claude(client: anthropic.Anthropic, prompt: str) -> str:
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    block = message.content[0]
    return block.text if block.type == "text" else ""


def ask_gemini(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text or ""


def synthesize_plan(client: anthropic.Anthropic, question: str, claude_resp: str, gemini_resp: str) -> str:
    system = "You are a master synthesizer. Given a question and two AI answers, create one clear, actionable combined plan that takes the best parts from each response."
    prompt = f"""Question: {question}

Claude's response:
{claude_resp}

Gemini's response:
{gemini_resp}

Create one strong, actionable plan combining the best insights from both responses."""
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    block = message.content[0]
    return block.text if block.type == "text" else ""


def run_code_agent(client: anthropic.Anthropic, plan: str) -> str:
    prompt = f"Write clean, well-commented Python code that implements this plan:\n\n{plan}"
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    block = message.content[0]
    return block.text if block.type == "text" else ""


def run_planning_agent(client: anthropic.Anthropic, plan: str) -> str:
    prompt = f"Break this plan into detailed, numbered action steps a developer can follow immediately:\n\n{plan}"
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    block = message.content[0]
    return block.text if block.type == "text" else ""


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What would you like to build or solve?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        anthropic_client = get_anthropic_client()
        gemini_client = get_gemini_client()

        status = st.status("Conductor is thinking...", expanded=True)

        with status:
            st.write("🤖 Asking Claude...")
            raw_claude = ask_claude(anthropic_client, prompt)

            st.write("✨ Asking Gemini...")
            raw_gemini = ask_gemini(gemini_client, prompt)

            st.write("🎼 Synthesizing the best plan...")
            plan = synthesize_plan(anthropic_client, prompt, raw_claude, raw_gemini)

            st.write("💻 Running Code Agent...")
            code_output = run_code_agent(anthropic_client, plan)

            st.write("📋 Running Planning Agent...")
            planning_output = run_planning_agent(anthropic_client, plan)

        status.update(label="Done!", state="complete")

        st.markdown("### 🎯 Synthesized Plan")
        st.markdown(plan)

        with st.expander("💻 Code Agent Output"):
            st.markdown(code_output)

        with st.expander("📋 Planning Agent Output"):
            st.markdown(planning_output)

        final_result = f"""## Final Result from AI Conductor

### Synthesized Plan
{plan}

### Code Agent Output
{code_output}

### Planning Agent Output
{planning_output}

**Recommended Next Step:** Save the code above and run it!
"""

        with open("conductor_result.md", "w", encoding="utf-8") as f:
            f.write(f"# Question\n{prompt}\n\n{final_result}")

        st.success("✅ Result saved as conductor_result.md")
        st.caption("💰 Estimated cost per run: ≈ $0.08–$0.25 depending on response length")

        st.session_state.messages.append({"role": "assistant", "content": final_result})

with st.sidebar:
    st.header("How it works")
    st.write("1. Multiple AIs answer your question")
    st.write("2. Conductor creates one strong plan")
    st.write("3. Agents compete (Code + Planning)")
    st.write("4. You get the best combined result + downloadable file")
    st.markdown("---")
    st.caption("Built with Claude (Anthropic) + Gemini via Replit AI Integrations")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from google import genai
from google.genai import types
import os
from pathlib import Path

# Resolve config.yaml relative to this file, regardless of working directory
CONFIG_PATH = Path(__file__).parent / "config.yaml"

st.set_page_config(page_title="AI Conductor", page_icon="🎼", layout="centered")

# ==================== Authentication Setup ====================
with open(CONFIG_PATH) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

authenticator.login()

name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

if authentication_status is False:
    st.error('Username/password is incorrect')

elif authentication_status is None:
    st.warning('Please enter your username and password')

elif authentication_status:
    st.success(f"Welcome, {name}!")

    # ==================== Usage Counter (per user) ====================
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Limit: 10 free queries per session
    MAX_FREE_QUERIES = 10

    # ==================== Main App ====================
    st.title("🎼 AI Conductor")
    st.caption("One task → Multiple AIs → Best plan + execution")

    @st.cache_resource
    def get_claude(temperature: float = 0.3):
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            anthropic_api_url=os.environ["AI_INTEGRATIONS_ANTHROPIC_BASE_URL"],
            anthropic_api_key=os.environ["AI_INTEGRATIONS_ANTHROPIC_API_KEY"],
            temperature=temperature,
            max_tokens=8192,
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

    def ask_gemini(prompt: str) -> str:
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text or ""

    # Sidebar
    with st.sidebar:
        st.header("How it works")
        st.write("1. Multiple AIs answer your question")
        st.write("2. Conductor creates one strong plan")
        st.write("3. Agents compete (Code + Planning)")
        st.write("4. You get the best combined result + downloadable file")
        st.divider()
        st.metric("Your Usage Today", f"{st.session_state.usage_count} / {MAX_FREE_QUERIES} queries")
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        authenticator.logout('Logout', 'sidebar')

    if st.session_state.usage_count >= MAX_FREE_QUERIES:
        st.warning(f"You've reached the free limit ({MAX_FREE_QUERIES} queries). Upgrade coming soon!")
        st.stop()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("What would you like to build or solve?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.session_state.usage_count += 1

            try:
                claude = get_claude(temperature=0.3)
                synthesizer = get_claude(temperature=0.2)

                # --- Step 1: Gather raw answers ---
                status = st.status("🤖 Asking Claude...", expanded=True)
                with status:
                    raw_claude = claude.invoke(prompt).content
                    status.update(label="✅ Claude answered")

                status2 = st.status("✨ Asking Gemini...", expanded=True)
                with status2:
                    raw_gemini = ask_gemini(prompt)
                    status2.update(label="✅ Gemini answered")

                # --- Step 2: Synthesize plan (show immediately) ---
                status3 = st.status("🎼 Synthesizing plan...", expanded=True)
                with status3:
                    plan = synthesizer.invoke([
                        SystemMessage(content="Create one clear, actionable plan by combining the best parts."),
                        HumanMessage(content=f"Task: {prompt}\n\nClaude: {raw_claude}\n\nGemini: {raw_gemini}"),
                    ]).content
                    status3.update(label="✅ Plan ready")

                st.markdown("**📋 Synthesized Plan:**")
                st.write(plan)

                # --- Step 3: Run agents ---
                status4 = st.status("💻 Running Code Agent...", expanded=True)
                with status4:
                    code_agent = synthesizer.invoke([
                        HumanMessage(content=f"Write clean Python code for this plan:\n{plan}"),
                    ]).content
                    status4.update(label="✅ Code Agent done")

                status5 = st.status("📋 Running Planning Agent...", expanded=True)
                with status5:
                    planning_agent = synthesizer.invoke([
                        HumanMessage(content=f"Break this into detailed numbered steps:\n{plan}"),
                    ]).content
                    status5.update(label="✅ Planning Agent done")

                with st.expander("💻 Code Agent Output"):
                    st.markdown(code_agent)

                with st.expander("📋 Planning Agent Output"):
                    st.markdown(planning_agent)

                final = f"""**Final Result**

**Plan Summary:**
{plan}

**Code Agent Output:**
```python
{code_agent}
```

**Planning Agent Output:**
{planning_agent}

**Recommended Next Step:** Save the code above and run it!
"""

                with open(Path(__file__).parent / "conductor_result.md", "w", encoding="utf-8") as f:
                    f.write(f"# Question\n{prompt}\n\n{final}")

                st.success("✅ Result saved as conductor_result.md")
                st.caption("💰 Estimated cost per run: ≈ $0.08–$0.25 depending on length")
                st.session_state.messages.append({"role": "assistant", "content": final})

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
                st.session_state.usage_count = max(0, st.session_state.usage_count - 1)

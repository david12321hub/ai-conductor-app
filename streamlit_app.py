import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from google import genai
from pathlib import Path
import os

st.set_page_config(page_title="AI Conductor", page_icon="🎼", layout="centered")

# ==================== Authentication Setup ====================
# Supports both Streamlit Cloud (st.secrets) and local config.yaml

def load_auth_config():
    if "credentials" in st.secrets:
        return {
            "credentials": st.secrets["credentials"].to_dict(),
            "cookie": st.secrets["cookie"].to_dict(),
        }
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.load(f, Loader=SafeLoader)
    st.error("No authentication config found. Add credentials to Streamlit secrets or provide config.yaml.")
    st.stop()

config = load_auth_config()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

authenticator.login()

name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")

if authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")
elif authentication_status:
    st.success(f"Welcome, {name}!")

    # ==================== Usage Counter ====================
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    MAX_FREE_QUERIES = 10

    # ==================== AI Client Setup ====================
    # Reads from Streamlit secrets (Streamlit Cloud) or env vars (Replit)
    def get_anthropic_key():
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"], None
        return (
            os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY"),
            os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL"),
        )

    def get_gemini_key():
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"], None
        return (
            os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY"),
            os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL"),
        )

    @st.cache_resource
    def get_claude(temperature: float = 0.3):
        api_key, base_url = get_anthropic_key()
        kwargs = dict(
            model="claude-sonnet-4-5",
            anthropic_api_key=api_key,
            temperature=temperature,
            max_tokens=8192,
        )
        if base_url:
            kwargs["anthropic_api_url"] = base_url
        return ChatAnthropic(**kwargs)

    @st.cache_resource
    def get_gemini_client():
        api_key, base_url = get_gemini_key()
        if base_url:
            from google.genai import types
            return genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(base_url=base_url, api_version=""),
            )
        return genai.Client(api_key=api_key)

    def ask_gemini(prompt: str) -> str:
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text or ""

    # ==================== Main App ====================
    st.title("🎼 AI Conductor")
    st.caption("One task → Multiple AIs → Best plan + execution")

    with st.sidebar:
        st.header("How it works")
        st.write("1. Multiple AIs answer your question")
        st.write("2. Conductor creates one strong plan")
        st.write("3. Agents compete (Code + Planning)")
        st.write("4. You get the best combined result")
        st.divider()
        st.metric("Your Usage", f"{st.session_state.usage_count} / {MAX_FREE_QUERIES} queries")
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        authenticator.logout('Logout', 'sidebar')

    if st.session_state.usage_count >= MAX_FREE_QUERIES:
        st.warning(f"You've reached the free limit ({MAX_FREE_QUERIES} queries). Upgrade coming soon!")
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("What would you like to build or solve?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.session_state.usage_count += 1

            try:
                claude = get_claude(temperature=0.3)
                synthesizer = get_claude(temperature=0.2)

                status = st.status("🤖 Asking Claude...", expanded=True)
                with status:
                    raw_claude = claude.invoke(prompt).content
                    status.update(label="✅ Claude answered")

                status2 = st.status("✨ Asking Gemini...", expanded=True)
                with status2:
                    try:
                        raw_gemini = ask_gemini(prompt)
                        status2.update(label="✅ Gemini answered")
                        gemini_available = True
                    except Exception as gemini_err:
                        raw_gemini = ""
                        gemini_available = False
                        if "429" in str(gemini_err) or "RESOURCE_EXHAUSTED" in str(gemini_err):
                            status2.update(label="⚠️ Gemini quota exceeded — continuing with Claude only")
                        else:
                            status2.update(label=f"⚠️ Gemini unavailable — continuing with Claude only")

                status3 = st.status("🎼 Synthesizing plan...", expanded=True)
                with status3:
                    if gemini_available:
                        synth_context = f"Task: {prompt}\n\nClaude: {raw_claude}\n\nGemini: {raw_gemini}"
                        synth_instruction = "Create one clear, actionable plan by combining the best parts of both responses."
                    else:
                        synth_context = f"Task: {prompt}\n\nClaude: {raw_claude}"
                        synth_instruction = "Create one clear, actionable plan based on this response."
                    plan = synthesizer.invoke([
                        SystemMessage(content=synth_instruction),
                        HumanMessage(content=synth_context),
                    ]).content
                    status3.update(label="✅ Plan ready")

                st.markdown("**📋 Synthesized Plan:**")
                st.write(plan)

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
                # Download instead of file write (Streamlit Cloud is read-only)
                st.download_button(
                    label="⬇️ Download result as Markdown",
                    data=f"# Question\n{prompt}\n\n{final}",
                    file_name="conductor_result.md",
                    mime="text/markdown",
                )

                st.session_state.messages.append({"role": "assistant", "content": final})

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
                st.session_state.usage_count = max(0, st.session_state.usage_count - 1)

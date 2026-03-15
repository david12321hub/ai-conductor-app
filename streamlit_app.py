import streamlit as st
from supabase import create_client, Client
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from google import genai
from pathlib import Path
import os

st.set_page_config(page_title="AI Conductor", page_icon="🎼", layout="centered")

# ==================== Supabase Setup ====================
supabase_url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
supabase_key = st.secrets.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    st.error("Supabase credentials not found. Add SUPABASE_URL and SUPABASE_ANON_KEY to Streamlit secrets.")
    st.stop()

supabase: Client = create_client(supabase_url, supabase_key)

# ==================== Session State ====================
if "user" not in st.session_state:
    st.session_state.user = None
if "usage_count" not in st.session_state:
    st.session_state.usage_count = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== Auth UI ====================
def show_auth():
    st.title("🎼 AI Conductor")
    st.caption("One task → Multiple AIs → Best plan + execution")
    st.divider()

    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Log In", use_container_width=True):
            try:
                response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                if response.user:
                    st.session_state.user = response.user
                    st.rerun()
                else:
                    st.error("Login failed.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab_signup:
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_pw")
        confirm_pw = st.text_input("Confirm Password", type="password", key="signup_confirm")
        if st.button("Create Account", use_container_width=True):
            if new_password != confirm_pw:
                st.error("Passwords don't match.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                try:
                    response = supabase.auth.sign_up({"email": new_email, "password": new_password})
                    if response.user:
                        st.success("Account created! Check your email to confirm, then log in.")
                    else:
                        st.error("Sign up failed — email may already be in use.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ==================== AI Clients ====================
def get_anthropic_key():
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
    base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
    return api_key, base_url

def get_gemini_key():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
    base_url = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
    return api_key, base_url

@st.cache_resource
def get_claude(temperature: float = 0.3):
    api_key, base_url = get_anthropic_key()
    kwargs = dict(model="claude-sonnet-4-5", anthropic_api_key=api_key,
                  temperature=temperature, max_tokens=8192)
    if base_url:
        kwargs["anthropic_api_url"] = base_url
    return ChatAnthropic(**kwargs)

@st.cache_resource
def get_gemini_client():
    api_key, base_url = get_gemini_key()
    if base_url:
        from google.genai import types
        return genai.Client(api_key=api_key,
                            http_options=types.HttpOptions(base_url=base_url, api_version=""))
    return genai.Client(api_key=api_key)

def ask_gemini(prompt: str) -> str:
    client = get_gemini_client()
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text or ""

# ==================== Main App ====================
if st.session_state.user:
    MAX_FREE_QUERIES = 10
    user_email = st.session_state.user.email

    st.title("🎼 AI Conductor")
    st.caption("One task → Multiple AIs → Best plan + execution")

    with st.sidebar:
        st.header("How it works")
        st.write("1. Multiple AIs answer your question")
        st.write("2. Conductor creates one strong plan")
        st.write("3. Agents compete (Code + Planning)")
        st.write("4. You get the best combined result")
        st.divider()
        st.caption(f"Signed in as **{user_email}**")
        st.metric("Usage Today", f"{st.session_state.usage_count} / {MAX_FREE_QUERIES} queries")
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        if st.button("🚪 Log Out"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.usage_count = 0
            st.rerun()

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
                            status2.update(label="⚠️ Gemini unavailable — continuing with Claude only")

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

else:
    show_auth()

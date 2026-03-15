import streamlit as st
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from google import genai
import os

st.set_page_config(page_title="AI Conductor", page_icon="🎼", layout="centered")

# ==================== Supabase Setup (direct REST — no package needed) ====================
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Add SUPABASE_URL and SUPABASE_ANON_KEY to your Streamlit secrets.")
    st.stop()

AUTH_HEADERS = {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}

def supabase_login(email: str, password: str):
    r = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        headers=AUTH_HEADERS,
        json={"email": email, "password": password},
        timeout=10,
    )
    if r.status_code == 200:
        return r.json().get("user"), None
    return None, r.json().get("error_description", r.json().get("msg", "Login failed."))

def supabase_signup(email: str, password: str):
    r = requests.post(
        f"{SUPABASE_URL}/auth/v1/signup",
        headers=AUTH_HEADERS,
        json={"email": email, "password": password},
        timeout=10,
    )
    if r.status_code == 200:
        return r.json().get("user"), None
    return None, r.json().get("error_description", r.json().get("msg", "Sign up failed."))

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
            if email and password:
                user, err = supabase_login(email, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error(f"Login failed: {err}")
            else:
                st.warning("Please enter your email and password.")

    with tab_signup:
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_pw")
        confirm_pw = st.text_input("Confirm Password", type="password", key="signup_confirm")
        if st.button("Create Account", use_container_width=True):
            if new_password != confirm_pw:
                st.error("Passwords don't match.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters.")
            elif not new_email:
                st.warning("Please enter your email.")
            else:
                user, err = supabase_signup(new_email, new_password)
                if user:
                    st.success("Account created! Check your email to confirm, then log in.")
                else:
                    st.error(f"Sign up failed: {err}")

# ==================== AI Helpers ====================
def _secret(key: str) -> str:
    return st.secrets.get(key) or os.environ.get(key, "")

@st.cache_resource
def get_claude(temperature: float = 0.3):
    api_key = _secret("ANTHROPIC_API_KEY") or _secret("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
    base_url = _secret("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
    kwargs = dict(model="claude-sonnet-4-5", anthropic_api_key=api_key,
                  temperature=temperature, max_tokens=8192)
    if base_url:
        kwargs["anthropic_api_url"] = base_url
    return ChatAnthropic(**kwargs)

@st.cache_resource
def get_gemini_client():
    api_key = _secret("GEMINI_API_KEY") or _secret("AI_INTEGRATIONS_GEMINI_API_KEY")
    base_url = _secret("AI_INTEGRATIONS_GEMINI_BASE_URL")
    if base_url:
        from google.genai import types
        return genai.Client(api_key=api_key,
                            http_options=types.HttpOptions(base_url=base_url, api_version=""))
    return genai.Client(api_key=api_key)

def ask_gemini(prompt: str) -> str:
    client = get_gemini_client()
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text or ""

def ask_cohere(prompt: str) -> str:
    api_key = _secret("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not configured in Streamlit secrets.")
    r = requests.post(
        "https://api.cohere.com/v2/chat",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "command-r-plus-08-2024", "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["message"]["content"][0]["text"]

def ask_mistral(prompt: str) -> str:
    api_key = _secret("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not configured in Streamlit secrets.")
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "mistral-large-latest", "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_ai_safely(name: str, fn, prompt: str, status_widget):
    try:
        result = fn(prompt)
        status_widget.update(label=f"✅ {name} answered")
        return result, True
    except Exception:
        status_widget.update(label=f"⚠️ {name} unavailable — skipped")
        return "", False

# ==================== Main App ====================
if st.session_state.user:
    MAX_FREE_QUERIES = 10
    user_email = st.session_state.user.get("email", "")

    st.title("🎼 AI Conductor")
    st.caption("One task → Claude · Gemini · Cohere · Mistral → Best combined plan")

    with st.sidebar:
        st.header("How it works")
        st.write("1. **4 AIs** answer your question simultaneously")
        st.write("2. Conductor synthesizes the best combined plan")
        st.write("3. Code Agent writes the implementation")
        st.write("4. Planning Agent breaks it into steps")
        st.divider()
        st.markdown("**Active AI Models:**")
        st.write("🤖 Claude (Anthropic)")
        st.write("✨ Gemini (Google)")
        st.write("🟣 Command R+ (Cohere)")
        st.write("🟠 Mistral Large")
        st.divider()
        st.caption(f"Signed in as **{user_email}**")
        st.metric("Usage Today", f"{st.session_state.usage_count} / {MAX_FREE_QUERIES} queries")
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        if st.button("🚪 Log Out"):
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

                s1 = st.status("🤖 Asking Claude...", expanded=False)
                with s1:
                    raw_claude = claude.invoke(prompt).content
                    s1.update(label="✅ Claude answered")

                s2 = st.status("✨ Asking Gemini...", expanded=False)
                with s2:
                    raw_gemini, gemini_ok = call_ai_safely("Gemini", ask_gemini, prompt, s2)

                s3 = st.status("🟣 Asking Cohere...", expanded=False)
                with s3:
                    raw_cohere, cohere_ok = call_ai_safely("Cohere", ask_cohere, prompt, s3)

                s4 = st.status("🟠 Asking Mistral...", expanded=False)
                with s4:
                    raw_mistral, mistral_ok = call_ai_safely("Mistral", ask_mistral, prompt, s4)

                s5 = st.status("🎼 Synthesizing plan from all AIs...", expanded=False)
                with s5:
                    responses = [f"Claude:\n{raw_claude}"]
                    if gemini_ok:
                        responses.append(f"Gemini:\n{raw_gemini}")
                    if cohere_ok:
                        responses.append(f"Cohere:\n{raw_cohere}")
                    if mistral_ok:
                        responses.append(f"Mistral:\n{raw_mistral}")

                    ai_count = len(responses)
                    synth_context = f"Task: {prompt}\n\n" + "\n\n".join(responses)
                    synth_instruction = (
                        f"You received {ai_count} AI responses to the same task. "
                        "Synthesize the best ideas from all of them into one clear, actionable, well-structured plan. "
                        "Do not simply pick one — combine the strongest insights from each."
                    )
                    plan = synthesizer.invoke([
                        SystemMessage(content=synth_instruction),
                        HumanMessage(content=synth_context),
                    ]).content
                    s5.update(label=f"✅ Plan synthesized from {ai_count} AI response(s)")

                st.markdown("**📋 Synthesized Plan:**")
                st.write(plan)

                s6 = st.status("💻 Running Code Agent...", expanded=False)
                with s6:
                    code_agent = synthesizer.invoke([
                        HumanMessage(content=f"Write clean, well-commented Python code to implement this plan:\n{plan}"),
                    ]).content
                    s6.update(label="✅ Code Agent done")

                s7 = st.status("📋 Running Planning Agent...", expanded=False)
                with s7:
                    planning_agent = synthesizer.invoke([
                        HumanMessage(content=f"Break this plan into detailed, numbered action steps a developer can follow:\n{plan}"),
                    ]).content
                    s7.update(label="✅ Planning Agent done")

                with st.expander("💻 Code Agent Output"):
                    st.markdown(code_agent)
                with st.expander("📋 Planning Agent Output"):
                    st.markdown(planning_agent)

                active_ais = (["Claude"] + (["Gemini"] if gemini_ok else []) +
                              (["Cohere"] if cohere_ok else []) + (["Mistral"] if mistral_ok else []))

                final = f"""**Final Result** *(synthesized from {", ".join(active_ais)})*

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
                st.success(f"✅ Done! Used {len(active_ais)} AI(s): {', '.join(active_ais)}")
                st.session_state.messages.append({"role": "assistant", "content": final})

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
                st.session_state.usage_count = max(0, st.session_state.usage_count - 1)

else:
    show_auth()

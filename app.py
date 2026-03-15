import streamlit as st
from supabase import create_client, Client
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from google import genai
from google.genai import types
import requests
import os
from pathlib import Path

st.set_page_config(page_title="AI Conductor", page_icon="🎼", layout="centered")

# ==================== Supabase Setup ====================
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    st.error("Supabase credentials not found. Add SUPABASE_URL and SUPABASE_ANON_KEY to secrets.")
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
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text or ""

def ask_cohere(prompt: str) -> str:
    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        raise ValueError("COHERE_API_KEY not set")
    r = requests.post(
        "https://api.cohere.com/v2/chat",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "command-r-plus-08-2024", "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["message"]["content"][0]["text"]

def ask_mistral(prompt: str) -> str:
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")
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
    except Exception as err:
        status_widget.update(label=f"⚠️ {name} unavailable — skipped")
        return "", False

# ==================== Main App ====================
if st.session_state.user:
    MAX_FREE_QUERIES = 10
    user_email = st.session_state.user.email

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

                active_ais = ["Claude"] + (["Gemini"] if gemini_ok else []) + (["Cohere"] if cohere_ok else []) + (["Mistral"] if mistral_ok else [])
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
                with open(Path(__file__).parent / "conductor_result.md", "w", encoding="utf-8") as f:
                    f.write(f"# Question\n{prompt}\n\n{final}")

                st.success(f"✅ Done! Used {len(active_ais)} AI(s): {', '.join(active_ais)}")
                st.caption("💰 Estimated cost per run: ≈ $0.08–$0.30 depending on length")
                st.session_state.messages.append({"role": "assistant", "content": final})

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
                st.session_state.usage_count = max(0, st.session_state.usage_count - 1)

else:
    show_auth()

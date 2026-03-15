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

st.set_page_config(page_title="AI Conductor", page_icon="🎼", layout="centered")

# ==================== Authentication Setup ====================
COOKIE_NAME = os.environ.get("COOKIE_NAME", "ai_conductor_auth")
COOKIE_KEY = os.environ.get("COOKIE_KEY", "ai_conductor_secret_key_change_me")
COOKIE_EXPIRY = int(os.environ.get("COOKIE_EXPIRY_DAYS", "30"))

@st.cache_data(ttl=60)
def load_credentials():
    try:
        from supabase_auth import load_credentials_from_supabase
        return load_credentials_from_supabase()
    except Exception:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.load(f, SafeLoader)
            return cfg.get("credentials", {"usernames": {}})
        return {"usernames": {}}

credentials = load_credentials()

authenticator = stauth.Authenticate(
    credentials,
    COOKIE_NAME,
    COOKIE_KEY,
    COOKIE_EXPIRY,
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
        st.divider()

        # ── Admin Panel (only shown to 'admin' user) ──
        if username == "admin":
            with st.expander("👤 User Management"):
                try:
                    from supabase_auth import list_users, add_user, delete_user
                    tab_list, tab_add, tab_del = st.tabs(["Users", "Add", "Remove"])

                    with tab_list:
                        users = list_users()
                        if users:
                            for u in users:
                                st.write(f"**{u['username']}** — {u['name']} ({u['email']})")
                        else:
                            st.info("No users found.")

                    with tab_add:
                        new_uname = st.text_input("Username", key="add_uname")
                        new_name = st.text_input("Display name", key="add_name")
                        new_email = st.text_input("Email", key="add_email")
                        new_pass = st.text_input("Password", type="password", key="add_pass")
                        if st.button("Add user"):
                            if new_uname and new_name and new_email and new_pass:
                                add_user(new_uname, new_name, new_email, new_pass)
                                st.success(f"User '{new_uname}' added.")
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.warning("Please fill in all fields.")

                    with tab_del:
                        del_uname = st.text_input("Username to remove", key="del_uname")
                        if st.button("Remove user", type="primary"):
                            if del_uname and del_uname != "admin":
                                delete_user(del_uname)
                                st.success(f"User '{del_uname}' removed.")
                                st.cache_data.clear()
                                st.rerun()
                            elif del_uname == "admin":
                                st.error("Cannot remove the admin user.")
                            else:
                                st.warning("Enter a username to remove.")
                except Exception as e:
                    st.error(f"User management unavailable: {e}")

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

                # --- Step 2: Synthesize plan (show immediately) ---
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

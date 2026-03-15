import streamlit as st
import requests
import stripe as stripe_lib
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
DB_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

stripe_lib.api_key = st.secrets.get("STRIPE_SECRET_KEY") or os.environ.get("STRIPE_SECRET_KEY", "")

# ==================== Credit Packages ====================
CREDIT_PACKAGES = [
    {
        "key": "starter",
        "name": "Starter",
        "emoji": "🥉",
        "price_usd": "$5",
        "price_cents": 500,
        "credits": 50,
        "features": ["50 credits", "All 4 AI models", "Download results"],
    },
    {
        "key": "pro",
        "name": "Pro",
        "emoji": "🥈",
        "price_usd": "$15",
        "price_cents": 1500,
        "credits": 200,
        "features": ["200 credits", "All 4 AI models", "Download results", "Priority support"],
    },
    {
        "key": "unlimited",
        "name": "Unlimited",
        "emoji": "🥇",
        "price_usd": "$49",
        "price_cents": 4900,
        "credits": 9999,
        "features": ["Unlimited credits", "All 4 AI models", "Download results", "Priority support", "Early features"],
    },
]

# ==================== Auth Helpers ====================
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

# ==================== Credits Helpers ====================
def get_credits(user_id: str) -> int:
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/credits?user_id=eq.{user_id}&select=balance",
            headers=DB_HEADERS, timeout=10,
        )
        data = r.json()
        if data:
            return data[0]["balance"]
        requests.post(
            f"{SUPABASE_URL}/rest/v1/credits",
            headers=DB_HEADERS,
            json={"user_id": user_id, "balance": 10},
            timeout=10,
        )
        return 10
    except Exception:
        return 0

def deduct_credit(user_id: str, balance: int) -> int:
    new_balance = max(0, balance - 1)
    try:
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/credits?user_id=eq.{user_id}",
            headers=DB_HEADERS,
            json={"balance": new_balance},
            timeout=10,
        )
    except Exception:
        pass
    return new_balance

def add_credits(user_id: str, current_balance: int, amount: int) -> int:
    if amount >= 9999:
        new_balance = 9999
    else:
        new_balance = min(9999, current_balance + amount)
    try:
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/credits?user_id=eq.{user_id}",
            headers=DB_HEADERS,
            json={"balance": new_balance},
            timeout=10,
        )
    except Exception:
        pass
    return new_balance

# ==================== Stripe Helpers ====================
def get_base_url() -> str:
    try:
        host = st.context.headers.get("host", "")
        if host:
            return f"https://{host}"
    except Exception:
        pass
    return os.environ.get("APP_URL", "http://localhost:8501")

def create_checkout_session(package: dict, user_email: str, user_id: str) -> tuple[str, str]:
    base_url = get_base_url()
    credits = package["credits"]
    label = "Unlimited" if credits >= 9999 else str(credits)
    session = stripe_lib.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "unit_amount": package["price_cents"],
                "product_data": {
                    "name": f"AI Conductor — {package['name']} Pack",
                    "description": f"{label} credits for AI Conductor",
                },
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=f"{base_url}?payment=success&session_id={{CHECKOUT_SESSION_ID}}&credits={credits}",
        cancel_url=f"{base_url}?payment=cancelled",
        customer_email=user_email,
        metadata={"user_id": user_id, "credits": str(credits)},
    )
    return session.id, session.url

def verify_stripe_session(session_id: str) -> bool:
    try:
        session = stripe_lib.checkout.Session.retrieve(session_id)
        return session.payment_status == "paid"
    except Exception:
        return False

# ==================== Session State ====================
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "chat"
if "checkout_url" not in st.session_state:
    st.session_state.checkout_url = None

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

# ==================== Upgrade Page ====================
def show_upgrade(user_email: str, user_id: str, balance: int):
    st.title("💳 Buy Credits")
    st.caption(f"Current balance: **{balance} credit(s)**  •  Each query costs 1 credit")
    st.divider()

    if st.session_state.checkout_url:
        st.success("Your payment session is ready!")
        st.link_button(
            "🔒 Complete Payment on Stripe",
            st.session_state.checkout_url,
            use_container_width=True,
        )
        if st.button("← Choose a different plan"):
            st.session_state.checkout_url = None
            st.rerun()
        st.caption("You'll be taken to Stripe's secure checkout. Return here after payment.")
        st.stop()

    cols = st.columns(3)
    for col, pkg in zip(cols, CREDIT_PACKAGES):
        with col:
            st.markdown(f"### {pkg['emoji']} {pkg['name']}")
            st.markdown(f"**{pkg['price_usd']}**")
            for feature in pkg["features"]:
                st.markdown(f"- {feature}")
            if st.button(f"Buy {pkg['name']}", use_container_width=True, key=f"buy_{pkg['key']}"):
                with st.spinner("Creating secure checkout..."):
                    try:
                        _, checkout_url = create_checkout_session(pkg, user_email, user_id)
                        st.session_state.checkout_url = checkout_url
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not create payment session: {e}")

    st.divider()
    st.caption("Payments are processed securely by Stripe. All major cards accepted.")
    if st.button("← Back to Chat"):
        st.session_state.page = "chat"
        st.rerun()

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
        raise ValueError("COHERE_API_KEY not configured.")
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
        raise ValueError("MISTRAL_API_KEY not configured.")
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
    user_id = st.session_state.user.get("id", "")
    user_email = st.session_state.user.get("email", "")
    balance = get_credits(user_id)

    # ---- Handle Stripe redirect back ----
    params = st.query_params
    if params.get("payment") == "success":
        session_id = params.get("session_id", "")
        credits_to_add = int(params.get("credits", "0"))
        if session_id and credits_to_add > 0:
            if verify_stripe_session(session_id):
                balance = add_credits(user_id, balance, credits_to_add)
                st.session_state.checkout_url = None
                st.query_params.clear()
                label = "Unlimited" if credits_to_add >= 9999 else str(credits_to_add)
                st.success(f"🎉 Payment successful! {label} credits added. New balance: **{balance}**")
            else:
                st.query_params.clear()
                st.warning("Payment could not be verified. Contact support if you were charged.")
    elif params.get("payment") == "cancelled":
        st.query_params.clear()
        st.info("Payment cancelled — no charge was made.")

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("**🎼 AI Conductor**")
        st.caption(f"Signed in as **{user_email}**")
        st.divider()

        credit_color = "🟢" if balance > 5 else ("🟡" if balance > 0 else "🔴")
        st.metric("Credits", f"{credit_color} {balance}")

        if st.button("💳 Buy More Credits", use_container_width=True):
            st.session_state.page = "upgrade"
            st.rerun()

        st.divider()
        st.markdown("**Active AI Models:**")
        st.write("🤖 Claude · ✨ Gemini")
        st.write("🟣 Cohere · 🟠 Mistral")
        st.divider()

        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        if st.button("🚪 Log Out"):
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.page = "chat"
            st.rerun()

    if st.session_state.page == "upgrade":
        show_upgrade(user_email, user_id, balance)
        st.stop()

    st.title("🎼 AI Conductor")
    st.caption("One task → Claude · Gemini · Cohere · Mistral → Best combined plan")

    if balance <= 0:
        st.error("🔴 You're out of credits.")
        st.markdown("Upgrade your plan to keep using AI Conductor.")
        if st.button("💳 View Plans & Buy Credits", use_container_width=True):
            st.session_state.page = "upgrade"
            st.rerun()
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("What would you like to build or solve?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
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

                balance = deduct_credit(user_id, balance)

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
                st.success(f"✅ Done! Used {len(active_ais)} AI(s): {', '.join(active_ais)} · Credits remaining: {balance}")
                st.session_state.messages.append({"role": "assistant", "content": final})

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")

else:
    show_auth()

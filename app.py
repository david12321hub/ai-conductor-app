import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from google import genai
from google.genai import types
import requests
import os
from pathlib import Path

st.set_page_config(page_title="AI Conductor", page_icon="", layout="centered")

st.markdown("""
    <style>
    /* ── Base ── */
    .stApp, .stApp > div { background-color: #0e1117 !important; color: #ffffff !important; }
    header[data-testid="stHeader"] { background-color: #0e1117 !important; }
    footer { display: none !important; }

    /* ── Text ── */
    h1, h2, h3, h4, h5, h6 { color: #00d4ff !important; font-family: 'Helvetica Neue', sans-serif; }
    p, div, span, label, small, li, td, th { color: #ffffff !important; }
    .stCaption, [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] { color: #bbbbbb !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #ffffff !important; }
    [data-testid="stMarkdownContainer"] p { color: #ffffff !important; }

    /* ── Inputs — dark background, no white boxes ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background-color: #1e2130 !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
        border-radius: 6px !important;
    }
    [data-testid="stChatInput"] {
        background-color: #90EE90 !important;
        padding: 8px !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInputContainer"] {
        background-color: #90EE90 !important;
    }
    [data-testid="stChatInput"] textarea,
    .stChatInput textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #aaaaaa !important;
        border-radius: 6px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background-color: #0e1117 !important; }
    .stTabs [data-baseweb="tab"] { color: #aaaaaa !important; background-color: #0e1117 !important; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #2e2e2e !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        border: 1px solid #444444 !important;
    }
    .stButton > button:hover { background-color: #3d3d3d !important; border: 1px solid #555555 !important; }
    .stDownloadButton > button {
        background-color: #2e2e2e !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
        border-radius: 8px !important;
    }
    .stLinkButton a, [data-testid="stLinkButton"] a {
        background-color: #2e2e2e !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        border: 1px solid #444444 !important;
        text-decoration: none !important;
        display: inline-block !important;
    }
    .stLinkButton a:hover, [data-testid="stLinkButton"] a:hover {
        background-color: #3d3d3d !important; color: #ffffff !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }

    /* ── Expanders ── */
    .stExpander { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
    .stExpander summary, details summary span, .stExpander summary p {
        color: #111111 !important;
        background-color: #d0d0d0 !important;
        border-radius: 6px !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }

    /* ── Status / alerts ── */
    [data-testid="stStatus"] { background-color: #1e2130 !important; border: 1px solid #30363d !important; }
    [data-testid="stAlert"] { background-color: #1e2130 !important; }

    /* ── Selectbox / dropdown ── */
    [data-testid="stSelectbox"] > div > div { background-color: #1e2130 !important; color: #ffffff !important; border: 1px solid #444444 !important; }

    /* ── Divider ── */
    hr { border-color: #30363d !important; }
    </style>
""", unsafe_allow_html=True)

LOGO_PATH = str(Path(__file__).parent / "conductor_logo.png")

# ==================== Supabase Setup (direct REST) ====================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials not found. Add SUPABASE_URL and SUPABASE_ANON_KEY to secrets.")
    st.stop()

AUTH_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

# ==================== Stripe Price IDs ====================
STRIPE_BASIC_PRICE_ID        = "price_1TBMRNFC68YihsMHbAOq7jGj"  # $9/mo  — Basic
STRIPE_PRO_PRICE_ID          = "price_1TBMROFC68YihsMHDom1cias"   # $49/mo — Pro Unlimited
STRIPE_ENTERPRISE_PRICE_ID   = "price_1TBMRKFC68YihsMHgZofMvjO"   # $149/mo — Enterprise Unlimited

# Pay-as-you-go credit packs (one-time, priced via price_data)
PAYG_PACKS = {5: 100, 10: 250, 15: 400, 20: 600}  # dollars → credits

# ==================== Session State ====================
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "chat"
if "free_trial_used" not in st.session_state:
    st.session_state.free_trial_used = False
if "checkout_url" not in st.session_state:
    st.session_state.checkout_url = None

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
    data = r.json()
    if r.status_code == 200:
        user = data if data.get("id") else data.get("user")
        if user:
            return user, None
        return None, "This email is already registered. Please log in instead."
    return None, data.get("error_description", data.get("msg", data.get("error", "Sign up failed.")))

# ==================== Credits Helpers ====================
def get_credits(user_id: str) -> int:
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/credits?user_id=eq.{user_id}&select=balance",
            headers=AUTH_HEADERS, timeout=10,
        )
        data = r.json()
        if data:
            return data[0]["balance"]
        requests.post(
            f"{SUPABASE_URL}/rest/v1/credits",
            headers=AUTH_HEADERS,
            json={"user_id": user_id, "balance": 10},
            timeout=10,
        )
        return 10
    except Exception:
        return 0

def set_credits(user_id: str, new_balance: int):
    try:
        r = requests.patch(
            f"{SUPABASE_URL}/rest/v1/credits?user_id=eq.{user_id}",
            headers={**AUTH_HEADERS, "Prefer": "return=representation"},
            json={"balance": new_balance},
            timeout=10,
        )
        updated = r.json()
        if not updated:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/credits",
                headers=AUTH_HEADERS,
                json={"user_id": user_id, "balance": new_balance},
                timeout=10,
            )
    except Exception:
        pass

def deduct_credit(user_id: str, balance: int) -> int:
    new_balance = max(0, balance - 1)
    set_credits(user_id, new_balance)
    return new_balance

def add_credits(user_id: str, current_balance: int, amount: int) -> int:
    new_balance = min(9999, current_balance + amount) if amount < 9999 else 9999
    set_credits(user_id, new_balance)
    return new_balance

# ==================== Stripe Helpers ====================
def get_base_url() -> str:
    try:
        host = st.context.headers.get("host", "")
        if host:
            return f"https://{host}"
    except Exception:
        pass
    domain = os.environ.get("REPLIT_DEV_DOMAIN", "")
    return f"https://{domain}" if domain else "http://localhost:8501"

def _stripe_key() -> str:
    return os.environ.get("STRIPE_SECRET_KEY", "")

def create_stripe_session(price_id: str, mode: str, user_email: str, user_id: str, credits_grant: int) -> str:
    base_url = get_base_url()
    key = _stripe_key()
    success_url = f"{base_url}?payment=success&session_id={{CHECKOUT_SESSION_ID}}&credits={credits_grant}&mode={mode}"
    fields = [
        ("line_items[0][price]", price_id),
        ("line_items[0][quantity]", "1"),
        ("mode", mode),
        ("success_url", success_url),
        ("cancel_url", f"{base_url}?payment=cancelled"),
        ("client_reference_id", user_id),
    ]
    if user_email:
        fields.append(("customer_email", user_email))
    r = requests.post(
        "https://api.stripe.com/v1/checkout/sessions",
        auth=(key, ""), data=fields, timeout=15,
    )
    if not r.ok:
        err = r.json().get("error", {}).get("message", r.text)
        raise ValueError(f"Stripe error: {err}")
    return r.json()["url"]

def create_stripe_payg_session(amount_dollars: int, user_email: str, user_id: str) -> str:
    credits_grant = PAYG_PACKS.get(amount_dollars, amount_dollars * 20)
    base_url = get_base_url()
    key = _stripe_key()
    success_url = (
        f"{base_url}?payment=success&session_id={{CHECKOUT_SESSION_ID}}"
        f"&credits={credits_grant}&mode=payment"
    )
    fields = [
        ("line_items[0][price_data][currency]", "usd"),
        ("line_items[0][price_data][unit_amount]", str(amount_dollars * 100)),
        ("line_items[0][price_data][product_data][name]", f"AI Conductor Credits (${amount_dollars})"),
        ("line_items[0][quantity]", "1"),
        ("mode", "payment"),
        ("success_url", success_url),
        ("cancel_url", f"{base_url}?payment=cancelled"),
        ("client_reference_id", user_id),
    ]
    if user_email:
        fields.append(("customer_email", user_email))
    r = requests.post(
        "https://api.stripe.com/v1/checkout/sessions",
        auth=(key, ""), data=fields, timeout=15,
    )
    if not r.ok:
        err = r.json().get("error", {}).get("message", r.text)
        raise ValueError(f"Stripe error: {err}")
    return r.json()["url"]

def verify_stripe_session(session_id: str, mode: str) -> bool:
    try:
        r = requests.get(
            f"https://api.stripe.com/v1/checkout/sessions/{session_id}",
            auth=(_stripe_key(), ""), timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("status") == "complete" if mode == "subscription" else data.get("payment_status") == "paid"
    except Exception:
        return False

# ==================== Auth UI ====================
def show_auth():
    st.image(LOGO_PATH, width=180)
    st.title("AI Conductor")
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
                user, err = supabase_signup(new_email, new_password)
                if user:
                    st.success("Account created! Check your email to confirm, then log in.")
                else:
                    st.error(f"Sign up failed: {err}")

# ==================== Upgrade Page ====================
def show_upgrade(user_email: str, user_id: str, balance: int):
    st.title(" Upgrade for More")
    st.caption(f"Current balance: {balance} credit(s) • Free tier: 10 queries. Unlock unlimited + priority models.")
    st.divider()

    if st.session_state.checkout_url:
        st.success("Your secure checkout is ready!")
        st.link_button(" Pay securely with Stripe →", st.session_state.checkout_url, use_container_width=True)
        if st.button("← Choose a different plan"):
            st.session_state.checkout_url = None
            st.rerun()
        st.caption("You'll be taken to Stripe's hosted checkout. Return here after payment.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### Basic")
        st.markdown("$9 / month")
        st.markdown("- 500 queries / month\n- All 4 AI models\n- Basic agents\n- Download results\n- Cancel anytime")
        if st.button("Choose Basic — $9/mo", use_container_width=True, key="buy_basic"):
            with st.spinner("Creating checkout..."):
                try:
                    url = create_stripe_session(STRIPE_BASIC_PRICE_ID, "subscription", user_email, user_id, 500)
                    st.session_state.checkout_url = url
                    st.rerun()
                except Exception as e:
                    st.error(f"Payment setup error: {str(e)}")

    with col2:
        st.markdown("### Pro Unlimited")
        st.markdown("$49 / month")
        st.markdown("- Unlimited light use\n- 10K tokens/day\n- All 4 AI models\n- Custom agents\n- Priority processing\n- Cancel anytime")
        if st.button("Choose Pro Unlimited — $49/mo", use_container_width=True, key="buy_pro"):
            with st.spinner("Creating checkout..."):
                try:
                    url = create_stripe_session(STRIPE_PRO_PRICE_ID, "subscription", user_email, user_id, 9999)
                    st.session_state.checkout_url = url
                    st.rerun()
                except Exception as e:
                    st.error(f"Payment setup error: {str(e)}")

    with col3:
        st.markdown("### Enterprise Unlimited")
        st.markdown("$149 / month")
        st.markdown("- 50K tokens/day\n- Priority processing\n- API access\n- Dedicated support\n- Custom integrations\n- SLA guarantee")
        if st.button("Choose Enterprise — $149/mo", use_container_width=True, key="buy_enterprise"):
            with st.spinner("Creating checkout..."):
                try:
                    url = create_stripe_session(STRIPE_ENTERPRISE_PRICE_ID, "subscription", user_email, user_id, 9999)
                    st.session_state.checkout_url = url
                    st.rerun()
                except Exception as e:
                    st.error(f"Payment setup error: {str(e)}")

    with col4:
        st.markdown("### Pay-as-you-go")
        st.markdown("One-time credit pack")
        st.markdown("- $5 → 100 credits\n- $10 → 250 credits\n- $15 → 400 credits\n- $20 → 600 credits")
        payg_amt = st.selectbox("Amount ($)", [5, 10, 15, 20], key="payg_amt_upg")
        if st.button(f"Buy ${payg_amt} Credits", use_container_width=True, key="buy_payg"):
            with st.spinner("Creating checkout..."):
                try:
                    url = create_stripe_payg_session(payg_amt, user_email, user_id)
                    st.session_state.checkout_url = url
                    st.rerun()
                except Exception as e:
                    st.error(f"Payment setup error: {str(e)}")

    st.divider()
    st.caption("Payments processed securely by Stripe. Test card: `4242 4242 4242 4242` · any future date · any CVC.")
    if st.button("← Back to Chat"):
        st.session_state.page = "chat"
        st.rerun()

# ==================== Plans (main area) ====================
def show_plans(user_email: str, user_id: str):
    st.subheader("Choose Your Plan")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown("Free Trial")
        st.caption("One-time free query")
        if st.button("Start Free Trial", key="plan_trial", use_container_width=True):
            if not st.session_state.get("free_trial_used"):
                st.session_state.free_trial_used = True
                st.success("Free trial activated! Ask your question below.")
            else:
                st.warning("Already used — upgrade to continue.")

    with c2:
        st.markdown("Basic")
        st.caption("$9/mo · 500 queries")
        if st.button("Choose Basic", key="plan_basic", use_container_width=True):
            try:
                url = create_stripe_session(STRIPE_BASIC_PRICE_ID, "subscription", user_email, user_id, 500)
                st.session_state.checkout_url = url
                st.session_state.page = "upgrade"
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with c3:
        st.markdown("Pro Unlimited")
        st.caption("$49/mo · 10K tokens/day")
        if st.button("Choose Pro", key="plan_pro", use_container_width=True):
            try:
                url = create_stripe_session(STRIPE_PRO_PRICE_ID, "subscription", user_email, user_id, 9999)
                st.session_state.checkout_url = url
                st.session_state.page = "upgrade"
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with c4:
        st.markdown("Enterprise Unlimited")
        st.caption("$149/mo · 50K tokens/day")
        if st.button("Choose Enterprise", key="plan_ent", use_container_width=True):
            try:
                url = create_stripe_session(STRIPE_ENTERPRISE_PRICE_ID, "subscription", user_email, user_id, 9999)
                st.session_state.checkout_url = url
                st.session_state.page = "upgrade"
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with c5:
        st.markdown("Pay-as-you-go")
        st.caption("One-time credit pack")
        payg_amt = st.selectbox("Amount ($)", [5, 10, 15, 20], key="payg_amt_plans")
        if st.button(f"Buy ${payg_amt}", key="plan_payg", use_container_width=True):
            try:
                url = create_stripe_payg_session(payg_amt, user_email, user_id)
                st.session_state.checkout_url = url
                st.session_state.page = "upgrade"
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()

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
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def ask_gemini(prompt: str) -> str:
    client = get_gemini_client()
    return client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text or ""

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
    except Exception as e:
        status_widget.update(label=f"⚠️ {name} skipped — {str(e)[:80]}")
        return "", False

# ==================== Main App ====================
if st.session_state.user:
    user_id = st.session_state.user.id
    user_email = st.session_state.user.email
    balance = get_credits(user_id)

    # ---- Handle Stripe redirect ----
    params = st.query_params
    if params.get("payment") == "success":
        session_id = params.get("session_id", "")
        credits_to_add = int(params.get("credits", "0"))
        mode = params.get("mode", "payment")
        if session_id and verify_stripe_session(session_id, mode):
            balance = add_credits(user_id, balance, credits_to_add)
            st.session_state.checkout_url = None
            st.query_params.clear()
            if credits_to_add >= 9999:
                st.success("🎉 Pro subscription activated! Unlimited queries unlocked.")
            else:
                st.success(f"🎉 Payment successful! {credits_to_add} credits added. Balance: {balance}")
        elif session_id:
            st.query_params.clear()
            st.warning("Payment could not be verified. Contact support if you were charged.")
    elif params.get("payment") == "cancelled":
        st.query_params.clear()
        st.info("Payment cancelled — no charge was made.")

    # ---- Sidebar ----
    with st.sidebar:
        st.image(LOGO_PATH, width=100)
        st.caption(f"Signed in as {user_email}")
        st.divider()
        st.header("Choose Your Plan")
        st.caption("Start free or upgrade for unlimited access.")

        with st.expander(" Free Trial — One-Time", expanded=True):
            st.write("Try AI Conductor once, completely free.")
            st.write("• Full access to all models & agents")
            st.write("• One complete query (plan + code)")
            st.write("No card needed — just click below.")
            if st.button("Start Free Trial", key="free_trial", use_container_width=True):
                if not st.session_state.get("free_trial_used"):
                    st.session_state.free_trial_used = True
                    st.success("Free trial activated! Ask your question below.")
                else:
                    st.warning("You've already used your free trial. Upgrade to continue!")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("Basic")
            st.caption("$9/mo")
            if st.button("Choose", key="sb_basic", use_container_width=True):
                try:
                    url = create_stripe_session(STRIPE_BASIC_PRICE_ID, "subscription", user_email, user_id, 500)
                    st.session_state.checkout_url = url
                    st.session_state.page = "upgrade"
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with sc2:
            st.markdown("Pro Unlimited")
            st.caption("$49/mo")
            if st.button("Choose", key="sb_pro", use_container_width=True):
                try:
                    url = create_stripe_session(STRIPE_PRO_PRICE_ID, "subscription", user_email, user_id, 9999)
                    st.session_state.checkout_url = url
                    st.session_state.page = "upgrade"
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with sc3:
            st.markdown("Enterprise")
            st.caption("$149/mo")
            if st.button("Choose", key="sb_ent", use_container_width=True):
                try:
                    url = create_stripe_session(STRIPE_ENTERPRISE_PRICE_ID, "subscription", user_email, user_id, 9999)
                    st.session_state.checkout_url = url
                    st.session_state.page = "upgrade"
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        st.markdown("Pay-as-you-go")
        sb_payg = st.selectbox("Top-up ($)", [5, 10, 15, 20], key="sb_payg_amt")
        if st.button(f"Buy ${sb_payg} Credits", key="sb_payg_buy", use_container_width=True):
            try:
                url = create_stripe_payg_session(sb_payg, user_email, user_id)
                st.session_state.checkout_url = url
                st.session_state.page = "upgrade"
                st.rerun()
            except Exception as e:
                st.error(str(e))

        st.divider()
        credit_color = "" if balance > 5 else ("" if balance > 0 else "")
        st.metric("Credits", f"{credit_color} {balance}")
        st.divider()
        st.markdown("Active AI Models:")
        st.write(" Claude · Gemini")
        st.write(" Cohere · Mistral")
        st.divider()
        if st.button(" Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        if st.button(" Log Out"):
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.page = "chat"
            st.rerun()

    if st.session_state.page == "upgrade":
        show_upgrade(user_email, user_id, balance)
        st.stop()

    st.image(LOGO_PATH, width=180)
    st.title("AI Conductor")
    st.caption("One task → Claude · Gemini · Cohere · Mistral → Best combined plan")
    st.markdown("Welcome! Type any task or question below — AI Conductor will handle the rest.")

    show_plans(user_email, user_id)

    if balance <= 0:
        st.error("You're out of credits. Choose a plan below to continue.")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Basic — $9/mo", use_container_width=True):
                try:
                    url = create_stripe_session(STRIPE_BASIC_PRICE_ID, "subscription", user_email, user_id, 500)
                    st.session_state.checkout_url = url
                    st.session_state.page = "upgrade"
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with col2:
            if st.button("Pro Unlimited — $49/mo", use_container_width=True):
                try:
                    url = create_stripe_session(STRIPE_PRO_PRICE_ID, "subscription", user_email, user_id, 9999)
                    st.session_state.checkout_url = url
                    st.session_state.page = "upgrade"
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with col3:
            if st.button("Enterprise — $149/mo", use_container_width=True):
                try:
                    url = create_stripe_session(STRIPE_ENTERPRISE_PRICE_ID, "subscription", user_email, user_id, 9999)
                    st.session_state.checkout_url = url
                    st.session_state.page = "upgrade"
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        st.stop()
    elif balance < 10:
        st.warning(f"Low credits ({balance} remaining) — consider upgrading your plan.")

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

                s1 = st.status(" Asking Claude...", expanded=False)
                with s1:
                    raw_claude = claude.invoke(prompt).content
                    s1.update(label="✅ Claude answered")

                s2 = st.status(" Asking Gemini...", expanded=False)
                with s2:
                    raw_gemini, gemini_ok = call_ai_safely("Gemini", ask_gemini, prompt, s2)

                s3 = st.status(" Asking Cohere...", expanded=False)
                with s3:
                    raw_cohere, cohere_ok = call_ai_safely("Cohere", ask_cohere, prompt, s3)

                s4 = st.status(" Asking Mistral...", expanded=False)
                with s4:
                    raw_mistral, mistral_ok = call_ai_safely("Mistral", ask_mistral, prompt, s4)

                s5 = st.status(" Synthesizing plan from all AIs...", expanded=False)
                with s5:
                    responses = [f"Claude:\n{raw_claude}"]
                    if gemini_ok: responses.append(f"Gemini:\n{raw_gemini}")
                    if cohere_ok: responses.append(f"Cohere:\n{raw_cohere}")
                    if mistral_ok: responses.append(f"Mistral:\n{raw_mistral}")
                    ai_count = len(responses)
                    plan = synthesizer.invoke([
                        SystemMessage(content=(
                            f"You received {ai_count} AI responses to the same task. "
                            "Synthesize the best ideas from all of them into one clear, actionable, well-structured plan. "
                            "Do not simply pick one — combine the strongest insights from each."
                        )),
                        HumanMessage(content=f"Task: {prompt}\n\n" + "\n\n".join(responses)),
                    ]).content
                    s5.update(label=f"✅ Plan synthesized from {ai_count} AI response(s)")

                balance = deduct_credit(user_id, balance)
                st.markdown(" Synthesized Plan:")
                st.write(plan)

                s6 = st.status("💻 Running Code Agent...", expanded=False)
                with s6:
                    code_agent = synthesizer.invoke([
                        HumanMessage(content=f"Write clean, well-commented Python code to implement this plan:\n{plan}"),
                    ]).content
                    s6.update(label="✅ Code Agent done")

                s7 = st.status(" Running Planning Agent...", expanded=False)
                with s7:
                    planning_agent = synthesizer.invoke([
                        HumanMessage(content=f"Break this plan into detailed, numbered action steps a developer can follow:\n{plan}"),
                    ]).content
                    s7.update(label="✅ Planning Agent done")

                with st.expander("💻 Code Agent Output"):
                    st.markdown(code_agent)
                with st.expander(" Planning Agent Output"):
                    st.markdown(planning_agent)

                active_ais = (["Claude"] + (["Gemini"] if gemini_ok else []) +
                              (["Cohere"] if cohere_ok else []) + (["Mistral"] if mistral_ok else []))

                final = (
                    f"Final Result *(synthesized from {', '.join(active_ais)})*\n\n"
                    f"Plan Summary:\n{plan}\n\n"
                    f"Code Agent Output:\n```python\n{code_agent}\n```\n\n"
                    f"Planning Agent Output:\n{planning_agent}\n\n"
                    f"Recommended Next Step: Save the code above and run it!\n"
                )
                with open(Path(__file__).parent / "conductor_result.md", "w", encoding="utf-8") as f:
                    f.write(f"# Question\n{prompt}\n\n{final}")

                st.session_state.last_code = code_agent
                st.session_state.last_plan = f"# Question\n{prompt}\n\n{final}"
                st.success(f"✅ Done! {', '.join(active_ais)} · Credits remaining: {balance}")
                st.session_state.messages.append({"role": "assistant", "content": final})

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")

    if st.session_state.get("last_code") or st.session_state.get("last_plan"):
        st.divider()
        st.markdown("### Plan Actions")
        col_exec, col_save = st.columns(2)

        with col_exec:
            if st.button("Run Code", use_container_width=True):
                import subprocess, tempfile, sys
                code = st.session_state.get("last_code", "")
                # Strip markdown code fences if present
                import re
                code = re.sub(r"^```[a-z]*\n?", "", code, flags=re.MULTILINE)
                code = re.sub(r"```$", "", code, flags=re.MULTILINE).strip()
                with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name
                with st.spinner("Running code..."):
                    result = subprocess.run(
                        [sys.executable, tmp_path],
                        capture_output=True, text=True, timeout=30
                    )
                if result.stdout:
                    with st.expander("Output", expanded=True):
                        st.code(result.stdout)
                if result.stderr:
                    with st.expander("Errors", expanded=True):
                        st.code(result.stderr)
                if not result.stdout and not result.stderr:
                    st.success("Code ran successfully with no output.")

        with col_save:
            plan_text = st.session_state.get("last_plan", "")
            st.download_button(
                label="Save Plan",
                data=plan_text,
                file_name="conductor_plan.md",
                mime="text/markdown",
                use_container_width=True,
            )

else:
    show_auth()

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from google import genai
from google.genai import types
import requests
import os
import concurrent.futures
import datetime
from pathlib import Path

st.set_page_config(page_title="AI Conductor", page_icon="", layout="centered")

st.markdown("""
    <style>
    /* ── Base ── */
    .stApp, .stApp > div { background-color: #0e1117 !important; color: #e6edf3 !important; }
    /* login page — keep main content dark */
    .block-container, .stMainBlockContainer { background-color: #0e1117 !important; }
    header[data-testid="stHeader"] { background-color: #0e1117 !important; }
    footer { display: none !important; }

    /* ── Text ── */
    h1, h2, h3, h4, h5, h6 { color: #00d4ff !important; font-family: 'Helvetica Neue', sans-serif; }
    /* override span/div inside headings so the text isn't overridden by the global span rule */
    h1 span, h1 div, h1 p,
    h2 span, h2 div, h2 p,
    h4 span, h4 div, h4 p { color: #00d4ff !important; }
    /* Choose Your Plan — amber for h3 and all children */
    h3, h3 span, h3 div, h3 p { color: #ffc107 !important; }
    /* sidebar headings stay cyan */
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h1 span,
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h2 span,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h3 span,
    section[data-testid="stSidebar"] h3 p, section[data-testid="stSidebar"] h3 div { color: #00d4ff !important; }
    p, div, span, label, small, li, td, th { color: #e6edf3 !important; }
    section[data-testid="stSidebar"] span.sidebar-email { color: #00d4ff !important; }
    .stCaption, [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] { color: #8b949e !important; }
    [data-testid="stMetricValue"] { color: #00d4ff !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li { color: #e6edf3 !important; }

    /* ── Inputs — dark background ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background-color: #1e2130 !important;
        color: #e6edf3 !important;
        border: 1px solid #444c56 !important;
        border-radius: 6px !important;
    }

    /* ── Chat input bottom area — grey surround ── */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background-color: #6e7681 !important;
        padding: 12px !important;
    }

    /* ── Chat input box ── */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInputContainer"] {
        background-color: #00d4ff !important;
        padding: 8px !important;
        border-radius: 10px !important;
        border: none !important;
    }
    [data-testid="stChatInput"] textarea,
    .stChatInput textarea,
    [data-testid="stChatInputContainer"] textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 6px !important;
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] textarea:focus-visible,
    [data-testid="stChatInputContainer"] textarea:focus {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInputContainer"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    [data-testid="stChatInput"] button,
    [data-testid="stChatInputContainer"] button {
        background-color: transparent !important;
        border: none !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background-color: #0e1117 !important; }
    .stTabs [data-baseweb="tab"] { color: #8b949e !important; background-color: #0e1117 !important; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        border: 1px solid #444c56 !important;
    }
    .stButton > button:hover { background-color: #30363d !important; border-color: #58a6ff !important; color: #ffffff !important; }
    .stDownloadButton > button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #444c56 !important;
        border-radius: 8px !important;
    }
    .stLinkButton a, [data-testid="stLinkButton"] a {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        border: 1px solid #444c56 !important;
        text-decoration: none !important;
        display: inline-block !important;
    }
    .stLinkButton a:hover, [data-testid="stLinkButton"] a:hover {
        background-color: #30363d !important; color: #ffffff !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label { color: #e6edf3 !important; }

    /* ── Expanders — header ── */
    .stExpander { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; overflow: hidden !important; }
    .stExpander summary, details summary span, .stExpander summary p {
        color: #e6edf3 !important;
        background-color: #21262d !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    /* ── Expanders — scrollable body ── */
    [data-testid="stExpanderDetails"] {
        background-color: #0d1117 !important;
        max-height: 520px !important;
        overflow-y: auto !important;
        padding: 14px 18px !important;
        border-top: 1px solid #30363d !important;
        scrollbar-width: thin !important;
        scrollbar-color: #444c56 #0d1117 !important;
    }
    [data-testid="stExpanderDetails"]::-webkit-scrollbar { width: 8px; }
    [data-testid="stExpanderDetails"]::-webkit-scrollbar-track { background: #0d1117; border-radius: 4px; }
    [data-testid="stExpanderDetails"]::-webkit-scrollbar-thumb { background-color: #444c56; border-radius: 4px; }
    [data-testid="stExpanderDetails"] p,
    [data-testid="stExpanderDetails"] li,
    [data-testid="stExpanderDetails"] span,
    [data-testid="stExpanderDetails"] div { color: #e6edf3 !important; line-height: 1.65 !important; }
    [data-testid="stExpanderDetails"] h1,
    [data-testid="stExpanderDetails"] h2,
    [data-testid="stExpanderDetails"] h3,
    [data-testid="stExpanderDetails"] h4 { color: #79c0ff !important; }
    [data-testid="stExpanderDetails"] code {
        background-color: #1f2937 !important;
        color: #a5f3fc !important;
        border-radius: 4px !important;
        padding: 2px 5px !important;
        font-size: 0.88em !important;
    }
    [data-testid="stExpanderDetails"] strong { color: #f0f6fc !important; }

    /* ── Code blocks (st.code) — white box, black text ── */
    .stCodeBlock, [data-testid="stCode"],
    .stCodeBlock > div, [data-testid="stCode"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 6px !important;
    }
    .stCodeBlock pre, [data-testid="stCode"] pre {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    .stCodeBlock code, [data-testid="stCode"] code,
    .stCodeBlock span, [data-testid="stCode"] span {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }

    /* ── Chat messages — scrollable ── */
    [data-testid="stChatMessage"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        max-height: 600px !important;
        overflow-y: auto !important;
        scrollbar-width: thin !important;
        scrollbar-color: #444c56 #161b22 !important;
    }
    [data-testid="stChatMessage"]::-webkit-scrollbar { width: 8px; }
    [data-testid="stChatMessage"]::-webkit-scrollbar-track { background: #161b22; border-radius: 4px; }
    [data-testid="stChatMessage"]::-webkit-scrollbar-thumb { background-color: #444c56; border-radius: 4px; }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] div { color: #e6edf3 !important; }

    /* ── Status boxes ── */
    [data-testid="stStatus"] { background-color: #1e2130 !important; border: 1px solid #30363d !important; border-radius: 6px !important; }
    [data-testid="stStatus"] p, [data-testid="stStatus"] span,
    [data-testid="stStatus"] div { color: #e6edf3 !important; }

    /* ── Alert / success / error / warning / info ── */
    [data-testid="stAlert"] { border-radius: 6px !important; }
    [data-testid="stAlert"] p, [data-testid="stAlert"] span,
    [data-testid="stAlert"] div { color: #e6edf3 !important; }
    div[data-baseweb="notification"][kind="positive"],
    .stSuccess > div { background-color: #0a2a18 !important; border-left: 4px solid #22c55e !important; }
    div[data-baseweb="notification"][kind="negative"],
    .stError > div { background-color: #2a0d0d !important; border-left: 4px solid #ef4444 !important; }
    div[data-baseweb="notification"][kind="warning"],
    .stWarning > div { background-color: #2a1c00 !important; border-left: 4px solid #f59e0b !important; }
    div[data-baseweb="notification"][kind="info"],
    .stInfo > div { background-color: #0a1a2a !important; border-left: 4px solid #3b82f6 !important; }

    /* ── Selectbox label — white text, no box ── */
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] label p,
    [data-testid="stSelectbox"] label span {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    /* ── Selectbox closed — transparent background, white selected value ── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-baseweb="select"],
    [data-baseweb="select"] > div {
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px solid #444c56 !important;
    }
    [data-testid="stSelectbox"] span,
    [data-testid="stSelectbox"] div,
    [data-testid="stSelectbox"] p,
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    [data-testid="stSelectbox"] svg,
    [data-baseweb="select"] svg { fill: #ffffff !important; }
    /* ── Selectbox open dropdown — white background, black text ── */
    [data-baseweb="popover"],
    [data-baseweb="popover"] ul,
    [data-baseweb="popover"] [data-baseweb="menu"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #ffffff !important;
    }
    [data-baseweb="option"],
    li[role="option"],
    ul[role="listbox"] li {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    [data-baseweb="option"] *,
    li[role="option"] *,
    ul[role="listbox"] li * {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    [data-baseweb="option"]:hover,
    li[role="option"]:hover {
        background-color: #e8e8e8 !important;
    }
    [data-baseweb="option"]:hover *,
    li[role="option"]:hover * {
        background-color: #e8e8e8 !important;
        color: #000000 !important;
    }

    /* ── Plan Action Buttons ── */
    .plan-exec-marker { display: none; }
    div:has(.plan-exec-marker) + div [data-testid="stColumn"]:nth-child(1) .stButton > button {
        background-color: #dc2626 !important;
        border-color: #b91c1c !important;
        color: #ffffff !important;
    }
    div:has(.plan-exec-marker) + div [data-testid="stColumn"]:nth-child(1) .stButton > button:hover {
        background-color: #b91c1c !important;
    }
    div:has(.plan-exec-marker) + div [data-testid="stColumn"]:nth-child(2) .stButton > button,
    div:has(.plan-exec-marker) + div [data-testid="stColumn"]:nth-child(2) .stDownloadButton > button {
        background-color: #16a34a !important;
        border-color: #15803d !important;
        color: #ffffff !important;
    }
    div:has(.plan-exec-marker) + div [data-testid="stColumn"]:nth-child(2) .stButton > button:hover,
    div:has(.plan-exec-marker) + div [data-testid="stColumn"]:nth-child(2) .stDownloadButton > button:hover {
        background-color: #15803d !important;
    }

    /* ── Divider ── */
    hr { border-color: #30363d !important; }
    </style>
""", unsafe_allow_html=True)

LOGO_PATH         = str(Path(__file__).parent / "conductor_logo.jpg")
SIDEBAR_LOGO_PATH = str(Path(__file__).parent / "sidebar_logo.png")

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
STRIPE_PRO_PRICE_ID          = "price_1TCRYTFC68YihsMHeQmcuabi"   # $10/mo — Pro Unlimited
STRIPE_ENTERPRISE_PRICE_ID   = "price_1TCRVKFC68YihsMHX5dYRXcJ"   # $25/mo — Enterprise Unlimited

OWNER_EMAIL = "david_darling12321@yahoo.co.uk"

PAYG_CREDITS = {5: 100, 10: 250, 15: 400, 20: 600}  # dollars → credits granted

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

# ==================== Daily Token Cap ====================
# Requires a Supabase table:
#   CREATE TABLE user_usage (
#     user_id text NOT NULL,
#     date    text NOT NULL,
#     tokens_used int NOT NULL DEFAULT 0,
#     PRIMARY KEY (user_id, date)
#   );
DAILY_TOKEN_CAP = {
    "free_trial":             5_000,
    "payg":                  10_000,
    "pro_unlimited":         100_000,
    "enterprise_unlimited":  500_000,
}

def get_user_tier(balance: int) -> str:
    if balance >= 9999:
        return "pro_unlimited"
    if balance >= 11:
        return "payg"
    return "free_trial"

def get_daily_tokens_used(user_id: str) -> int:
    try:
        today = datetime.date.today().isoformat()
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/user_usage?user_id=eq.{user_id}&date=eq.{today}&select=tokens_used",
            headers=AUTH_HEADERS, timeout=5,
        )
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("tokens_used", 0)
        return 0
    except Exception:
        return 0

def add_daily_tokens(user_id: str, tokens: int):
    try:
        today = datetime.date.today().isoformat()
        current = get_daily_tokens_used(user_id)
        new_total = current + tokens
        r = requests.patch(
            f"{SUPABASE_URL}/rest/v1/user_usage?user_id=eq.{user_id}&date=eq.{today}",
            headers={**AUTH_HEADERS, "Prefer": "return=representation"},
            json={"tokens_used": new_total},
            timeout=5,
        )
        if not r.ok or not r.json():
            requests.post(
                f"{SUPABASE_URL}/rest/v1/user_usage",
                headers=AUTH_HEADERS,
                json={"user_id": user_id, "date": today, "tokens_used": tokens},
                timeout=5,
            )
    except Exception:
        pass

def check_token_cap(user_id: str, balance: int) -> bool:
    tier = get_user_tier(balance)
    cap = DAILY_TOKEN_CAP.get(tier, 5_000)
    used = get_daily_tokens_used(user_id)
    if used >= cap:
        tier_label = tier.replace("_", " ").title()
        st.warning(
            f"Daily token limit reached ({used:,} / {cap:,} tokens) for your **{tier_label}** tier. "
            "Upgrade your plan or wait until tomorrow to continue."
        )
        st.stop()
    return True

# ==================== Transaction Tracking ====================
# Requires a Supabase table:
#   CREATE TABLE transactions (
#     id            bigserial PRIMARY KEY,
#     user_id       text      NOT NULL,
#     user_email    text,
#     tier          text,
#     amount_paid   numeric   NOT NULL DEFAULT 0,
#     amount_profit numeric   NOT NULL DEFAULT 0,
#     created_at    timestamptz DEFAULT now()
#   );

def record_transaction(user_id: str, user_email: str, tier: str,
                       amount_paid: float, amount_profit: float):
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/transactions",
            headers=AUTH_HEADERS,
            json={
                "user_id": user_id,
                "user_email": user_email,
                "tier": tier,
                "amount_paid": round(amount_paid, 2),
                "amount_profit": round(amount_profit, 2),
            },
            timeout=10,
        )
    except Exception:
        pass

def fetch_transactions() -> list:
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/transactions?select=*&order=created_at.desc",
            headers=AUTH_HEADERS, timeout=10,
        )
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []

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
    credits_grant = PAYG_CREDITS.get(amount_dollars, amount_dollars * 20)
    base_url = get_base_url()
    key = _stripe_key()
    success_url = (
        f"{base_url}?payment=success&session_id={{CHECKOUT_SESSION_ID}}"
        f"&credits={credits_grant}&mode=payment&amount={amount_dollars}"
    )
    fields = [
        ("line_items[0][price_data][currency]", "usd"),
        ("line_items[0][price_data][unit_amount]", str(amount_dollars * 100)),
        ("line_items[0][price_data][product_data][name]",
         f"AI Conductor Credits — ${amount_dollars} ({credits_grant} credits)"),
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
    st.image(LOGO_PATH, use_container_width=True)
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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Pro Unlimited")
        st.markdown("$10 / month")
        st.markdown("- Unlimited light use\n- 10K tokens/day\n- All 4 AI models\n- Custom agents\n- Priority processing\n- Cancel anytime")
        if st.button("Choose Pro Unlimited — $10/mo", use_container_width=True, key="buy_pro"):
            with st.spinner("Creating checkout..."):
                try:
                    url = create_stripe_session(STRIPE_PRO_PRICE_ID, "subscription", user_email, user_id, 9999)
                    st.session_state.checkout_url = url
                    st.rerun()
                except Exception as e:
                    st.error(f"Payment setup error: {str(e)}")

    with col2:
        st.markdown("### Enterprise Unlimited")
        st.markdown("$25 / month")
        st.markdown("- 50K tokens/day\n- Priority processing\n- API access\n- Dedicated support\n- Custom integrations\n- SLA guarantee")
        if st.button("Choose Enterprise — $25/mo", use_container_width=True, key="buy_enterprise"):
            with st.spinner("Creating checkout..."):
                try:
                    url = create_stripe_session(STRIPE_ENTERPRISE_PRICE_ID, "subscription", user_email, user_id, 9999)
                    st.session_state.checkout_url = url
                    st.rerun()
                except Exception as e:
                    st.error(f"Payment setup error: {str(e)}")

    with col3:
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
    c1, c2, c3, c4 = st.columns(4)

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
        st.markdown("Pro Unlimited")
        st.caption("$10/mo · 10K tokens/day")
        if st.button("Choose Pro", key="plan_pro", use_container_width=True):
            try:
                url = create_stripe_session(STRIPE_PRO_PRICE_ID, "subscription", user_email, user_id, 9999)
                st.session_state.checkout_url = url
                st.session_state.page = "upgrade"
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with c3:
        st.markdown("Enterprise Unlimited")
        st.caption("$25/mo · 50K tokens/day")
        if st.button("Choose Enterprise", key="plan_ent", use_container_width=True):
            try:
                url = create_stripe_session(STRIPE_ENTERPRISE_PRICE_ID, "subscription", user_email, user_id, 9999)
                st.session_state.checkout_url = url
                st.session_state.page = "upgrade"
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with c4:
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
        timeout=15,
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
        timeout=15,
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
        paid_amount = float(params.get("amount", "0") or "0")
        if session_id and verify_stripe_session(session_id, mode):
            balance = add_credits(user_id, balance, credits_to_add)
            st.session_state.checkout_url = None
            st.query_params.clear()
            if credits_to_add >= 9999:
                tier = "subscription"
                record_transaction(user_id, user_email, tier, 0, 0)
                st.success("Payment successful! Your subscription is now active — unlimited queries unlocked.")
            elif paid_amount > 0:
                credit_value = paid_amount * 0.5
                profit = paid_amount * 0.5
                tier = "payg"
                record_transaction(user_id, user_email, tier, paid_amount, profit)
                st.success(
                    f"Payment successful! ${credit_value:.2f} in credits added to your account "
                    f"({credits_to_add} queries). Profit retained: ${profit:.2f}. "
                    f"New balance: {balance} credits."
                )
            else:
                record_transaction(user_id, user_email, "other", 0, 0)
                st.success(f"Payment successful! {credits_to_add} credits added. Balance: {balance}")
        elif session_id:
            st.query_params.clear()
            st.warning("Payment could not be verified. Contact support if you were charged.")
    elif params.get("payment") == "cancelled":
        st.query_params.clear()
        st.info("Payment cancelled — no charge was made.")

    # ---- Sidebar ----
    with st.sidebar:
        st.image(SIDEBAR_LOGO_PATH, width=90)
        st.markdown(
            f'<p style="color:#8b949e;font-size:0.875rem;margin:0">Signed in as '
            f'<span class="sidebar-email">{user_email}</span></p>',
            unsafe_allow_html=True,
        )
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

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("Pro Unlimited")
            st.caption("$10/mo")
            if st.button("Choose", key="sb_pro", use_container_width=True):
                try:
                    url = create_stripe_session(STRIPE_PRO_PRICE_ID, "subscription", user_email, user_id, 9999)
                    st.session_state.checkout_url = url
                    st.session_state.page = "upgrade"
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with sc2:
            st.markdown("Enterprise")
            st.caption("$25/mo")
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

        # ---- Owner Dashboard ----
        if user_email == OWNER_EMAIL:
            st.divider()
            with st.expander("Owner Dashboard — Profit Tracking", expanded=False):
                st.subheader("Revenue & Profit per User")
                txns = fetch_transactions()
                if txns:
                    total_revenue = sum(t.get("amount_paid", 0) for t in txns)
                    total_profit  = sum(t.get("amount_profit", 0) for t in txns)
                    c1, c2 = st.columns(2)
                    c1.metric("Total Revenue", f"${total_revenue:,.2f}")
                    c2.metric("Total Profit (50%)", f"${total_profit:,.2f}")
                    st.divider()
                    st.markdown("**Per-user breakdown**")
                    user_map: dict = {}
                    for t in txns:
                        uid   = t.get("user_id", "unknown")
                        email = t.get("user_email") or uid[:8] + "…"
                        if uid not in user_map:
                            user_map[uid] = {"email": email, "revenue": 0.0, "profit": 0.0, "txns": 0}
                        user_map[uid]["revenue"] += t.get("amount_paid", 0)
                        user_map[uid]["profit"]  += t.get("amount_profit", 0)
                        user_map[uid]["txns"]    += 1
                    for uid, row in sorted(user_map.items(),
                                           key=lambda x: x[1]["profit"], reverse=True):
                        st.markdown(
                            f"**{row['email']}** — "
                            f"Revenue: ${row['revenue']:.2f} · "
                            f"Profit: **${row['profit']:.2f}** · "
                            f"{row['txns']} transaction(s)"
                        )
                else:
                    st.info("No transactions recorded yet.")

    if st.session_state.page == "upgrade":
        show_upgrade(user_email, user_id, balance)
        st.stop()

    st.image(LOGO_PATH, use_container_width=True)
    st.title("AI Conductor")
    st.caption("One task → Claude · Gemini · Cohere · Mistral → Best combined plan")
    st.markdown("Welcome! Type any task or question below — AI Conductor will handle the rest.")

    show_plans(user_email, user_id)

    if balance <= 0:
        st.error("You're out of credits. Choose a plan below to continue.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Pro Unlimited — $10/mo", use_container_width=True):
                try:
                    url = create_stripe_session(STRIPE_PRO_PRICE_ID, "subscription", user_email, user_id, 9999)
                    st.session_state.checkout_url = url
                    st.session_state.page = "upgrade"
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with col2:
            if st.button("Enterprise — $25/mo", use_container_width=True):
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
                check_token_cap(user_id, balance)
                synthesizer = get_claude(temperature=0.2)

                # Run all 4 AIs in parallel so total wait = slowest single model
                def _call_claude():
                    return get_claude(temperature=0.3).invoke(prompt).content

                with st.spinner("Asking Claude, Gemini, Cohere & Mistral simultaneously…"):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                        f_claude  = ex.submit(_call_claude)
                        f_gemini  = ex.submit(ask_gemini,  prompt)
                        f_cohere  = ex.submit(ask_cohere,  prompt)
                        f_mistral = ex.submit(ask_mistral, prompt)

                # Collect results and show individual status boxes
                try:
                    raw_claude = f_claude.result()
                    st.success("✅ Claude answered")
                except Exception as e:
                    raw_claude = ""
                    st.warning(f"⚠️ Claude skipped — {str(e)[:80]}")

                try:
                    raw_gemini = f_gemini.result()
                    gemini_ok = True
                    st.success("✅ Gemini answered")
                except Exception as e:
                    raw_gemini = ""
                    gemini_ok = False
                    st.warning(f"⚠️ Gemini skipped — {str(e)[:80]}")

                try:
                    raw_cohere = f_cohere.result()
                    cohere_ok = True
                    st.success("✅ Cohere answered")
                except Exception as e:
                    raw_cohere = ""
                    cohere_ok = False
                    st.warning(f"⚠️ Cohere skipped — {str(e)[:80]}")

                try:
                    raw_mistral = f_mistral.result()
                    mistral_ok = True
                    st.success("✅ Mistral answered")
                except Exception as e:
                    raw_mistral = ""
                    mistral_ok = False
                    st.warning(f"⚠️ Mistral skipped — {str(e)[:80]}")

                s5 = st.status(" Synthesizing plan from all AIs...", expanded=False)
                with s5:
                    responses = []
                    if raw_claude: responses.append(f"Claude:\n{raw_claude}")
                    if gemini_ok:  responses.append(f"Gemini:\n{raw_gemini}")
                    if cohere_ok:  responses.append(f"Cohere:\n{raw_cohere}")
                    if mistral_ok: responses.append(f"Mistral:\n{raw_mistral}")
                    if not responses:
                        s5.update(label="❌ All AIs failed — cannot synthesize")
                        st.error("All AI models failed to respond. Please try again.")
                        st.stop()
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

                # Estimate tokens from all text produced this query (~4 chars/token)
                all_text = " ".join(responses) + plan
                add_daily_tokens(user_id, max(1, len(all_text) // 4))

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

        # Paid = user has actually purchased credits or a subscription.
        # Free trial users start with 10 initial credits and drop to ≤9 after one query.
        is_paid = get_user_tier(balance) in ("payg", "pro_unlimited", "enterprise_unlimited")

        _UPGRADE_MSG = (
            "Executing and saving plans requires credits or an active subscription. "
            "Choose a plan in the sidebar — Pay-as-you-go from **$5**, "
            "or **Pro Unlimited** at **$10/mo**."
        )

        st.markdown('<span class="plan-exec-marker"></span>', unsafe_allow_html=True)
        col_exec, col_save = st.columns(2)

        with col_exec:
            if st.button("Execute Plan", use_container_width=True):
                if not is_paid:
                    st.warning(_UPGRADE_MSG)
                else:
                    import subprocess, tempfile, sys, re
                    code = st.session_state.get("last_code", "")
                    code = re.sub(r"^```[a-z]*\n?", "", code, flags=re.MULTILINE)
                    code = re.sub(r"```$", "", code, flags=re.MULTILINE).strip()
                    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
                        tmp.write(code)
                        tmp_path = tmp.name
                    with st.spinner("Executing..."):
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
                        st.success("Plan executed!")

        with col_save:
            if not is_paid:
                if st.button("Save Plan", use_container_width=True):
                    st.warning(_UPGRADE_MSG)
            else:
                plan_text = st.session_state.get("last_plan", "")
                st.download_button(
                    label="Save Plan",
                    data=plan_text,
                    file_name="conductor_plan.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

    # ==================== AI Helper Chat ====================
    if "helper_messages" not in st.session_state:
        st.session_state.helper_messages = []

    with st.expander("AI Helper", expanded=False):
        st.caption("Ask me anything about using AI Conductor")

        for msg in st.session_state.helper_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        h_col1, h_col2 = st.columns([5, 1])
        with h_col1:
            helper_input = st.text_input(
                "Question", key="helper_text_input",
                label_visibility="collapsed",
                placeholder="Type your question here...",
            )
        with h_col2:
            helper_send = st.button("Send", key="helper_send", use_container_width=True)

        if helper_send and helper_input:
            st.session_state.helper_messages.append({"role": "user", "content": helper_input})
            with st.spinner("Thinking..."):
                _system = (
                    "You are a friendly, patient assistant for the AI Conductor app. "
                    "Answer questions about how to use the app, its features, pricing tiers, "
                    "or troubleshooting. Keep answers short, clear, and encouraging."
                )
                response = get_claude().invoke(f"{_system}\n\nUser question: {helper_input}").content
            st.session_state.helper_messages.append({"role": "assistant", "content": response})
            st.rerun()

else:
    show_auth()

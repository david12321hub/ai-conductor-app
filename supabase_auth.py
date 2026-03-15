import os
import bcrypt
from supabase import create_client, Client


def get_supabase() -> Client:
    url = (os.environ.get("SUPABASE_URL")
           or _get_streamlit_secret("SUPABASE_URL"))
    key = (os.environ.get("SUPABASE_ANON_KEY")
           or os.environ.get("SUPABASE_KEY")
           or _get_streamlit_secret("SUPABASE_ANON_KEY")
           or _get_streamlit_secret("SUPABASE_KEY"))
    if not url or not key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_ANON_KEY. "
            "Add them as environment secrets (Replit) or Streamlit secrets."
        )
    return create_client(url, key)


def _get_streamlit_secret(key: str):
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


def load_credentials_from_supabase() -> dict:
    """Return a credentials dict compatible with streamlit-authenticator."""
    client = get_supabase()
    response = client.table("users").select("username, name, email, password").execute()
    usernames = {}
    for row in response.data:
        usernames[row["username"]] = {
            "name": row["name"],
            "email": row["email"],
            "password": row["password"],
        }
    return {"usernames": usernames}


def add_user(username: str, name: str, email: str, password: str) -> dict:
    """Add a new user with a bcrypt-hashed password."""
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    client = get_supabase()
    response = client.table("users").insert({
        "username": username,
        "name": name,
        "email": email,
        "password": hashed,
    }).execute()
    return response.data[0] if response.data else {}


def delete_user(username: str) -> bool:
    """Delete a user by username."""
    client = get_supabase()
    client.table("users").delete().eq("username", username).execute()
    return True


def update_password(username: str, new_password: str) -> bool:
    """Update a user's password."""
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    client = get_supabase()
    client.table("users").update({"password": hashed}).eq("username", username).execute()
    return True


def list_users() -> list:
    """List all users (without passwords)."""
    client = get_supabase()
    response = client.table("users").select("username, name, email, created_at").execute()
    return response.data or []

"""
watchlist_alerts.py — Watchlist + Telegram price-alert module
==============================================================
A standalone, reusable module with NO Streamlit dependency. Import it into
app.py (or any other project):

    from watchlist_alerts import AlertEngine, WatchlistStore, send_telegram

Features
--------
- send_telegram(): plain-requests Telegram Bot API delivery. Token/chat id
  come from environment variables (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)
  or can be passed explicitly (e.g. from st.secrets).
- AlertEngine: create alerts with DE-DUPLICATION (no identical
  symbol/direction/target twice), trigger evaluation, and a per-alert
  NOTIFICATION COOLDOWN so a trigger that stays true does not spam your
  chat every Streamlit rerun. Alerts RE-ARM after the cooldown expires.
- WatchlistStore: JSON persistence for the watchlist and the alerts so
  they survive app restarts (Streamlit session_state alone is volatile).

Security note
-------------
NEVER hardcode the bot token in source code or documents. Put it in a
.streamlit/secrets.toml file or a .env file, both of which must be
git-ignored.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

__all__ = ["AlertEngine", "WatchlistStore", "send_telegram"]

TELEGRAM_API = "https://api.telegram.org"
DEFAULT_STATE_FILE = "terminal_state.json"
DEFAULT_COOLDOWN_MINUTES = 60


# ===========================================================================
# Telegram delivery
# ===========================================================================
def send_telegram(message: str, token: str = "", chat_id: str = "",
                  timeout: int = 10) -> bool:
    """Send a message through the Telegram Bot API.

    `token` / `chat_id` fall back to the environment variables
    TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID when omitted, so callers can
    either pass st.secrets values or rely on secrets.toml / a .env file.

    Returns True only when Telegram answers HTTP 200. Never raises.
    """
    token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id or not message:
        return False
    try:
        response = requests.post(
            f"{TELEGRAM_API}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=timeout,
        )
        return response.status_code == 200
    except Exception:
        return False


def get_bot_username(token: str = "") -> str:
    """Best-effort getMe() check — useful in a diagnostics page."""
    token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return ""
    try:
        response = requests.get(f"{TELEGRAM_API}/bot{token}/getMe", timeout=10)
        payload = response.json()
        if payload.get("ok"):
            return str(payload["result"].get("username", ""))
    except Exception:
        pass
    return ""


# ===========================================================================
# Alert engine
# ===========================================================================
class AlertEngine:
    """Price-alert rules: de-duplication, trigger evaluation, cooldown."""

    def __init__(self, state_path: str = DEFAULT_STATE_FILE,
                 cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES) -> None:
        self.state_path = Path(state_path)
        self.cooldown_minutes = max(1, int(cooldown_minutes))

    # ---- persistence (watchlist + alerts survive restarts) ----
    def load_state(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            data = {}
        watchlist = [str(s) for s in (data.get("watchlist") or [])]
        alerts = []
        for alert in (data.get("alerts") or []):
            if isinstance(alert, dict) and alert.get("symbol"):
                alert.setdefault("id", uuid.uuid4().hex)
                alert.setdefault("notified", False)
                alert.setdefault("last_sent", None)
                alerts.append(alert)
        return watchlist, alerts

    def save_state(self, watchlist: List[str],
                   alerts: List[Dict[str, Any]]) -> None:
        payload = {"watchlist": list(watchlist), "alerts": alerts,
                   "saved_at": datetime.now().isoformat()}
        try:
            self.state_path.write_text(json.dumps(payload, indent=2),
                                       encoding="utf-8")
        except Exception:
            pass  # persistence is best-effort; never crash the app

    # ---- alert rules ----
    @staticmethod
    def add(alerts: List[Dict[str, Any]], symbol: str, price: float,
            direction: str) -> Tuple[bool, str]:
        """Append a new alert unless an identical one already exists.

        Returns (created, message). De-duplication key:
        (symbol, direction, target price).
        """
        if not symbol or price <= 0:
            return False, "A symbol and a positive target price are required."
        direction = "Above" if str(direction).lower().startswith("a") else "Below"
        for existing in alerts:
            same = (existing.get("symbol") == symbol
                    and existing.get("direction") == direction
                    and abs(float(existing.get("price", 0)) - float(price)) < 1e-9)
            if same:
                return False, (f"Duplicate alert: {symbol} {direction} "
                               f"{price:g} already exists.")
        alerts.append({
            "id": uuid.uuid4().hex,
            "symbol": symbol,
            "price": float(price),
            "direction": direction,
            "notified": False,
            "last_sent": None,
            "created": datetime.now().isoformat(),
        })
        return True, "Alert created."

    @staticmethod
    def is_triggered(alert: Dict[str, Any],
                     current_price: Optional[float]) -> bool:
        """True when the alert condition holds for the current price."""
        if current_price is None:
            return False
        try:
            current = float(current_price)
            target = float(alert["price"])
        except (TypeError, ValueError, KeyError):
            return False
        if alert.get("direction") == "Above":
            return current >= target
        if alert.get("direction") == "Below":
            return current <= target
        return False

    def should_notify(self, alert: Dict[str, Any]) -> bool:
        """Cooldown gate: an alert may notify again only after the
        cooldown has elapsed since its last Telegram message."""
        last_sent = alert.get("last_sent")
        if last_sent:
            try:
                elapsed_min = (datetime.now()
                               - datetime.fromisoformat(str(last_sent))
                               ).total_seconds() / 60.0
                if elapsed_min < self.cooldown_minutes:
                    return False
            except Exception:
                pass
        return True

    @staticmethod
    def mark_notified(alert: Dict[str, Any]) -> None:
        """Stamp the alert so the cooldown window starts now."""
        alert["last_sent"] = datetime.now().isoformat()
        alert["notified"] = True

    def rearm(self, alert: Dict[str, Any]) -> None:
        """Manually re-arm an alert (clears the one-shot flag)."""
        alert["notified"] = False

    def evaluate_many(self, alerts: List[Dict[str, Any]],
                      price_of, notify) -> List[Dict[str, Any]]:
        """Evaluate a list of alerts against `price_of(alert) -> price|None`."""
        delivered: List[Dict[str, Any]] = []
        for alert in alerts:
            current = price_of(alert)
            if not self.is_triggered(alert, current):
                continue
            if not self.should_notify(alert):
                continue
            if notify(alert, current):
                self.mark_notified(alert)
                delivered.append(alert)
        return delivered


# ===========================================================================
# Watchlist store
# ===========================================================================
class WatchlistStore:
    """Thin JSON-backed store shared with AlertEngine's state file."""

    def __init__(self, state_path: str = DEFAULT_STATE_FILE) -> None:
        self.state_path = Path(state_path)

    def load(self, default: Optional[List[str]] = None) -> List[str]:
        engine = AlertEngine(str(self.state_path))
        watchlist, _ = engine.load_state()
        return watchlist or list(default or [])

    def save(self, watchlist: List[str]) -> None:
        engine = AlertEngine(str(self.state_path))
        _, alerts = engine.load_state()
        engine.save_state(list(watchlist), alerts)

"""
watchlist_alerts.py — Alert Engine & Watchlist Persistence
==========================================================
Headless service layer with Telegram Bot delivery,
trigger deduplication, and notification cooldown gates.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

__all__ = ["AlertEngine", "WatchlistStore", "send_telegram"]

TELEGRAM_API = "https://api.telegram.org"
DEFAULT_STATE_FILE = Path(__file__).resolve().parent / ".cache" / "terminal_state.json"
DEFAULT_COOLDOWN_MINUTES = 60


def send_telegram(message: str, token: str = "", chat_id: str = "", timeout: int = 8) -> bool:
    """Send a formatted notification via Telegram Bot API.

    Falls back to environment variables TELEGRAM_BOT_TOKEN and
    TELEGRAM_CHAT_ID when credentials are not passed explicitly.
    """
    tok = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    cid = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
    if not tok or not cid or not message:
        return False
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/bot{tok}/sendMessage",
            json={
                "chat_id": cid,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


class AlertEngine:
    """Manages active price alerts with deduplication and cooldowns."""

    def __init__(
        self,
        state_path: Path | str = DEFAULT_STATE_FILE,
        cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES,
    ) -> None:
        self.state_path = Path(state_path)
        self.cooldown_minutes = max(1, int(cooldown_minutes))

    def load_state(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        if not self.state_path.exists():
            return [], []
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            data = {}
        watchlist = [str(s) for s in (data.get("watchlist") or [])]
        alerts = []
        for a in data.get("alerts") or []:
            if isinstance(a, dict) and a.get("symbol"):
                a.setdefault("id", uuid.uuid4().hex)
                a.setdefault("last_sent", None)
                alerts.append(a)
        return watchlist, alerts

    def save_state(self, watchlist: List[str], alerts: List[Dict[str, Any]]) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "watchlist": list(dict.fromkeys(watchlist)),
                "alerts": alerts,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def add(
        alerts: List[Dict[str, Any]],
        symbol: str,
        target: float,
        condition: str,
    ) -> Tuple[bool, str]:
        if not symbol or target <= 0:
            return False, "Specify a valid symbol and a price > 0."
        cond = "Above" if condition.lower().startswith("a") else "Below"
        for a in alerts:
            if (
                a.get("symbol") == symbol
                and a.get("condition") == cond
                and abs(float(a.get("target", 0)) - target) < 1e-6
            ):
                return False, f"Alert for {symbol} {cond} {target:g} already active."

        alerts.append({
            "id": uuid.uuid4().hex,
            "symbol": symbol.strip().upper(),
            "target": float(target),
            "condition": cond,
            "created": datetime.now().strftime("%d %b %H:%M"),
            "last_sent": None,
        })
        return True, "Alert set successfully."

    def should_notify(self, alert: Dict[str, Any]) -> bool:
        last = alert.get("last_sent")
        if not last:
            return True
        try:
            elapsed = (
                datetime.now(timezone.utc) - datetime.fromisoformat(last)
            ).total_seconds() / 60.0
            return elapsed >= self.cooldown_minutes
        except Exception:
            return True

    def evaluate_all(
        self,
        alerts: List[Dict[str, Any]],
        price_lookup: Callable[[str], Optional[float]],
        notifier: Optional[Callable[[str], bool]] = None,
    ) -> List[str]:
        triggered_msgs: List[str] = []
        now_str = datetime.now(timezone.utc).isoformat()

        for a in alerts:
            cur_price = price_lookup(a["symbol"])
            if cur_price is None or cur_price <= 0:
                continue

            hit = (
                (cur_price >= a["target"])
                if a["condition"] == "Above"
                else (cur_price <= a["target"])
            )
            if hit and self.should_notify(a):
                a["last_sent"] = now_str
                msg = (
                    f"⚡ <b>Price Alert</b>: {a['symbol']} is {a['condition'].lower()} "
                    f"target {a['target']:.2f} (Current: {cur_price:.2f})"
                )
                triggered_msgs.append(msg)
                if notifier:
                    notifier(msg)

        return triggered_msgs


class WatchlistStore:
    def __init__(self, state_path: Path | str = DEFAULT_STATE_FILE) -> None:
        self.engine = AlertEngine(state_path)

    def get(self, defaults: Optional[List[str]] = None) -> List[str]:
        wl, _ = self.engine.load_state()
        return wl if wl else list(defaults or ["RELIANCE.NS", "AAPL", "BTC-USD"])

    def save(self, watchlist: List[str], alerts: List[Dict[str, Any]]) -> None:
        self.engine.save_state(watchlist, alerts)

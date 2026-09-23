"""
AI Trade Terminal - Streamlit Edition (repaired build)
======================================================
Multi-market trading terminal + a REAL conversational AI assistant.

WHAT THIS BUILD FIXES (see the report in chat for the full detail)
------------------------------------------------------------------
1. CRASH FIX (traceback line 945): news payloads are parsed by
   `parse_news_items()` which never assumes `item["content"]` is a dict.
   A string / None / list / missing field can no longer kill the app.
2. CURRENCY FIX: every price is rendered in the instrument's OWN currency
   (INR, USD, GBP, EUR, JPY, HKD, AUD, SGD, KRW, CNY, CAD ...). Indian
   (.NS/.BO) instruments no longer display a dollar sign.
3. CHAT FIX: the assistant is a real multi-turn LLM conversation with
   streaming, memory, follow-ups, tool access and error handling - and it is
   a GENERAL assistant: you can talk about anything (thoughts, ideas, code,
   life). Market data is optional context, never a cage.
4. Multi-market architecture: market -> exchange -> universe -> symbols ->
   quotes -> normalized frame -> heat map / breadth / sectors, behind a
   provider interface, with an honest LIVE / DELAYED / LAST SESSION / CACHED /
   UNAVAILABLE data-status model (LIVE is never faked).
5. Market-closed behaviour: the last completed session is always shown.
6. ONE BAD SYMBOL, ONE BAD NEWS ITEM OR ONE FAILED AI/API CALL CAN NEVER CRASH
   THE TERMINAL.
7. SINGLE-INSTRUMENT QUOTES FIXED (KeyError: 'Close'): yf.download([one_symbol],
   group_by="ticker") returns MultiIndex columns, so frame["Close"] raised KeyError
   for every single-symbol quote (dashboard, watchlist, alerts, AI tools) while the
   multi-symbol heat map worked. The column layout is now normalized in one place.
8. HEAT-MAP TILES NOW SHOW 'SYMBOL / NAME' (plus price and % change), and the hover
   carries the full provider name, currency, sector, data status and last bar.

Framework: Streamlit (unchanged). Data: Yahoo Finance via yfinance.
AI: Groq and Google Gemini (keys from st.secrets / environment).
"""

from __future__ import annotations

import concurrent.futures as futures
import html
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

try:  # optional dependency - only needed for the Groq models
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None

try:  # optional dependency - only needed for the Gemini models
    from google import genai as google_genai
except Exception:  # pragma: no cover
    google_genai = None

try:
    from google.genai import types as genai_types
except Exception:  # pragma: no cover
    genai_types = None

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

APP_VERSION = "3.1.1"
HTTP_UA = "Mozilla/5.0 (compatible; AI-Trade-Terminal/3.0; +https://streamlit.io)"

# ---------------------------------------------------------------------------
# LOGGING  (never logs secrets - values are redacted before they reach a sink)
# ---------------------------------------------------------------------------
LOG = logging.getLogger("ai_trade_terminal")
if not LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)

SECRET_NAMES = (
    "GROQ_KEY",
    "GEMINI_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
)


def secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets first, then the environment."""
    try:
        value = st.secrets.get(name, None)
        if value not in (None, ""):
            return str(value).strip()
    except Exception:
        LOG.debug("st.secrets is not available for %s", name)
    return (os.environ.get(name, default) or default).strip()


def _redact(text: Any) -> str:
    out = str(text)
    for name in SECRET_NAMES:
        value = secret(name, "")
        if value and len(value) >= 6:
            out = out.replace(value, "***")
    return out


def safe_error(exc: BaseException) -> str:
    """A logged/shown error string with any secret material removed."""
    return f"{type(exc).__name__}: {_redact(exc)}"


def log_exception(context: str, exc: BaseException) -> None:
    LOG.warning("%s | %s", context, safe_error(exc))


# ---------------------------------------------------------------------------
# PAGE CONFIG + CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Trade Terminal", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

_CSS = """
<style>
    .stApp { background-color: #020617 !important; color: #f8fafc; }
    #MainMenu, footer, header { visibility: hidden; }

    /* dark card look applied to real Streamlit containers (responsive, no broken divs) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #0F172A;
        border: 1px solid rgba(51,65,85,0.45) !important;
        border-radius: 16px !important;
    }
    div[data-testid="stExpander"] {
        background: #0F172A; border: 1px solid rgba(51,65,85,0.45); border-radius: 14px;
    }

    .card-header { font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.1rem; }
    .card-sub { font-size: 0.75rem; color: #64748b; margin-bottom: 0.6rem; }

    .metric-label { font-size: 0.65rem; font-weight: 500; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.3rem; font-weight: 700; color: #f8fafc; font-variant-numeric: tabular-nums; }
    .up { color: #34d399 !important; }
    .down { color: #f87171 !important; }
    .flat { color: #94a3b8 !important; }
    .amber { color: #fbbf24 !important; }

    .pill {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.03em;
    }
    .pill-open      { background: rgba(16,185,129,0.18);  color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
    .pill-closed    { background: rgba(148,163,184,0.16); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.35); }
    .pill-pre       { background: rgba(251,191,36,0.16);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }
    .pill-delayed   { background: rgba(96,165,250,0.16);  color: #93c5fd; border: 1px solid rgba(96,165,250,0.4); }
    .pill-last      { background: rgba(167,139,250,0.16); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.4); }
    .pill-cached    { background: rgba(251,146,60,0.16);  color: #fdba74; border: 1px solid rgba(251,146,60,0.4); }
    .pill-unavailable { background: rgba(248,113,113,0.16); color: #fca5a5; border: 1px solid rgba(248,113,113,0.4); }
    .pill-neutral   { background: rgba(148,163,184,0.14); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.3); }

    .stButton > button {
        border-radius: 10px !important; font-weight: 600 !important;
        border: 1px solid rgba(51,65,85,0.55) !important;
        background: #1E293B !important; color: #e2e8f0 !important;
    }
    .stButton > button:hover {
        border-color: #fbbf24 !important; color: #fbbf24 !important;
        background: rgba(251,191,36,0.08) !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #94a3b8; padding: 7px 14px; }
    .stTabs [aria-selected="true"] { background: #1E293B !important; color: #f8fafc !important; }

    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid rgba(51,65,85,0.4);
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea textarea {
        background-color: #1E293B !important;
        border: 1px solid rgba(51,65,85,0.55) !important;
        border-radius: 8px !important; color: #f8fafc !important;
    }

    .chat-user {
        background: #334155; border-radius: 12px; padding: 9px 13px;
        margin: 5px 0 5px 16%; font-size: 0.88rem; color: #f1f5f9;
    }
    .chat-assistant {
        background: #1E293B; border: 1px solid rgba(51,65,85,0.4);
        border-radius: 12px; padding: 9px 13px; margin: 5px 10% 5px 0;
        font-size: 0.88rem; line-height: 1.45; color: #e2e8f0;
    }
    .chat-meta { font-size: 0.66rem; color: #64748b; margin: 0 0 6px 2px; }

    .level-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 12px; border-radius: 9px; margin-bottom: 4px; font-size: 0.9rem;
    }
    .level-support { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); }
    .level-resistance { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); }

    .plan-card { border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; }
    .plan-bull { background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.35); }
    .plan-bear { background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.35); }
    .plan-title-bull { color: #34d399; font-weight: 700; margin-bottom: 6px; }
    .plan-title-bear { color: #f87171; font-weight: 700; margin-bottom: 6px; }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

    /* dock the chat input inside the sidebar so it never overlaps the transcript */
    section[data-testid="stSidebar"] { padding-bottom: 5.5rem; }
    section[data-testid="stSidebar"] div[data-testid="stChatInput"] { padding-bottom: 0.6rem; }
    div[data-testid="stChatInput"] textarea { font-size: 0.9rem; }

    /* responsive: keep everything usable on laptop / tablet / phone */
    @media (max-width: 1100px) {
        .metric-value { font-size: 1.1rem; }
        .chat-user { margin-left: 6%; }
        .chat-assistant { margin-right: 3%; }
    }
    @media (max-width: 700px) {
        .card-header { font-size: 0.98rem; }
        .metric-value { font-size: 1.0rem; }
        .chat-user, .chat-assistant { margin: 5px 0; font-size: 0.85rem; }
        .level-row { font-size: 0.8rem; }
    }
</style>
"""


# ===========================================================================
# SECTION 1 - AI PROVIDER LAYER
# ---------------------------------------------------------------------------
# One small abstraction so every assistant in the app talks to a real model
# through the same interface:
#     provider.stream_chat(system, messages)  -> yields text
#     provider.complete_chat(system, messages, tools) -> (text, tool_calls)
# ===========================================================================
class AIUnavailable(RuntimeError):
    """The provider/model cannot be used right now (key, quota, model, network)."""


@dataclass(frozen=True)
class ModelSpec:
    label: str
    provider: str          # "groq" | "gemini"
    model_id: str


# The model ids are the ones configured in this project. If your key does not
# have access to one of them the chat shows a clear "model unavailable" message
# and you can pick another entry (or type your own id under "Custom model id").
MODEL_CATALOG: List[ModelSpec] = [
    ModelSpec("Groq · GPT-OSS 20B (fast)", "groq", "openai/gpt-oss-20b"),
    ModelSpec("Groq · GPT-OSS 120B (strongest)", "groq", "openai/gpt-oss-120b"),
    ModelSpec("Groq · Qwen3 32B", "groq", "qwen/qwen3-32b"),
    ModelSpec("Groq · Kimi K2", "groq", "moonshotai/kimi-k2-instruct"),
    ModelSpec("Gemini · 3.5 Flash", "gemini", "gemini-3.5-flash"),
    ModelSpec("Gemini · 3.1 Pro (preview)", "gemini", "gemini-3.1-pro-preview"),
]
CUSTOM_MODEL_LABEL = "Custom model id…"


def model_labels() -> List[str]:
    return [m.label for m in MODEL_CATALOG] + [CUSTOM_MODEL_LABEL]


def get_model_spec(label: str, custom_provider: str = "groq", custom_id: str = "") -> ModelSpec:
    for spec in MODEL_CATALOG:
        if spec.label == label:
            return spec
    provider = custom_provider if custom_provider in ("groq", "gemini") else "groq"
    return ModelSpec(f"Custom ({provider}: {custom_id or 'unset'})", provider, (custom_id or "").strip())


def friendly_ai_error(exc: BaseException) -> str:
    """Map provider failures onto a short, user-safe message (never a key)."""
    raw = safe_error(exc)
    low = str(exc).lower()
    if any(k in low for k in ("401", "invalid api key", "invalid_api_key", "unauthorized", "api key")):
        return "Invalid or missing API key for this provider - check your Streamlit secrets."
    if any(k in low for k in ("403", "permission", "forbidden")):
        return "The provider refused the request (403 / permission). The key may lack access to this model."
    if any(k in low for k in ("429", "rate limit", "rate_limit", "quota", "too many requests")):
        return "Provider rate limit or quota reached. Wait a moment and try again."
    if any(k in low for k in ("404", "not found", "does not exist", "no such model", "unavailable", "decommission")):
        return "This model id is not available on your key. Choose another model in the chat settings."
    if any(k in low for k in ("timeout", "timed out", "connection", "network", "ssl", "dns")):
        return "Network/timeout error while contacting the AI provider. Check connectivity and retry."
    if "context" in low and any(k in low for k in ("length", "limit", "window")):
        return "The conversation is longer than the model's context window - start a new conversation."
    return f"AI provider error: {raw}"


class AIProvider:
    label = "base"
    supports_tools = False

    def __init__(self, spec: ModelSpec, temperature: float = 0.65, max_tokens: int = 1100) -> None:
        self.spec = spec
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def is_configured(self) -> bool:
        return False

    def stream_chat(self, system: str, messages: List[Dict[str, str]],
                    tools: Optional[List[Dict[str, Any]]] = None) -> Iterable[str]:
        raise AIUnavailable("This provider is not implemented.")

    def complete_chat(self, system: str, messages: List[Dict[str, str]],
                      tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, str]]]:
        raise AIUnavailable("This provider is not implemented.")


def provider_key(provider: str) -> str:
    """v3.1.1: first configured secret among common alias names.

    Accepts GROQ_KEY or GROQ_API_KEY, and GEMINI_KEY / GEMINI_API_KEY /
    GOOGLE_API_KEY - so the chat works no matter which name you add to
    secrets. The value is never logged.
    """
    aliases = ("GROQ_KEY", "GROQ_API_KEY") if provider == "groq" else \
              ("GEMINI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    for name in aliases:
        value = secret(name)
        if value:
            return value
    return ""


class GroqProvider(AIProvider):
    label = "Groq"
    supports_tools = True

    def _client(self):
        if Groq is None:
            raise AIUnavailable("The `groq` package is not installed in this deployment.")
        key = provider_key("groq")
        if not key:
            raise AIUnavailable("No Groq key found. Add GROQ_KEY (or GROQ_API_KEY) to "
                                ".streamlit/secrets.toml or Streamlit → Settings → Secrets. "
                                "Free key: console.groq.com")
        return Groq(api_key=key)

    def is_configured(self) -> bool:
        return bool(provider_key("groq")) and Groq is not None

    @staticmethod
    def _payload(system: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            role = m.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = m.get("content", "") or ""
            if not content:
                continue
            out.append({"role": role, "content": content})
        return out

    def complete_chat(self, system, messages, tools=None):
        client = self._client()
        kwargs: Dict[str, Any] = dict(model=self.model_id,
                                      messages=self._payload(system, messages),
                                      temperature=self.temperature,
                                      max_tokens=self.max_tokens)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise AIUnavailable(friendly_ai_error(exc)) from exc
        choices = getattr(resp, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        text = (getattr(message, "content", "") or "").strip()
        calls: List[Dict[str, str]] = []
        for call in (getattr(message, "tool_calls", None) or []):
            try:
                calls.append({"id": call.id, "name": call.function.name,
                              "arguments": call.function.arguments or "{}"})
            except Exception as exc:
                log_exception("groq tool_call parse", exc)
        return text, calls

    def stream_chat(self, system, messages, tools=None):
        client = self._client()
        try:
            stream = client.chat.completions.create(model=self.model_id,
                                                    messages=self._payload(system, messages),
                                                    stream=True, temperature=self.temperature,
                                                    max_tokens=self.max_tokens)
        except Exception as exc:
            raise AIUnavailable(friendly_ai_error(exc)) from exc
        try:
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    yield piece
        except Exception as exc:
            raise AIUnavailable(friendly_ai_error(exc)) from exc


class GeminiProvider(AIProvider):
    label = "Gemini"

    def _client(self):
        if google_genai is None:
            raise AIUnavailable("The `google-genai` package is not installed in this deployment.")
        key = provider_key("gemini")
        if not key:
            raise AIUnavailable("No Gemini key found. Add GEMINI_KEY (or GEMINI_API_KEY / "
                                "GOOGLE_API_KEY) to .streamlit/secrets.toml or Streamlit → "
                                "Settings → Secrets. Free key: aistudio.google.com")
        return google_genai.Client(api_key=key)

    def is_configured(self) -> bool:
        return bool(provider_key("gemini")) and google_genai is not None

    @staticmethod
    def _contents(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = "model" if m.get("role") == "assistant" else "user"
            text = (m.get("content") or "").strip()
            if not text:
                continue
            if out and out[-1]["role"] == role:
                out[-1]["parts"][0]["text"] += "\n\n" + text
            else:
                out.append({"role": role, "parts": [{"text": text}]})
        if out and out[0]["role"] != "user":
            out.insert(0, {"role": "user", "parts": [{"text": "(conversation start)"}]})
        return out

    def _config(self, system: str) -> Any:
        if genai_types is not None:
            return genai_types.GenerateContentConfig(system_instruction=system or None,
                                                     temperature=self.temperature,
                                                     max_output_tokens=self.max_tokens)
        return {"system_instruction": system or None, "temperature": self.temperature,
                "max_output_tokens": self.max_tokens}

    def _prepare(self, system: str, messages: List[Dict[str, str]]) -> Any:
        client = self._client()
        contents = self._contents(messages)
        if not contents:
            raise AIUnavailable("There is nothing to send to the model.")
        return client, contents

    def stream_chat(self, system, messages, tools=None):
        client, contents = self._prepare(system, messages)
        try:
            stream = client.models.generate_content_stream(model=self.model_id, contents=contents,
                                                           config=self._config(system))
        except Exception as exc:
            raise AIUnavailable(friendly_ai_error(exc)) from exc
        try:
            for chunk in stream:
                piece = getattr(chunk, "text", None)
                if piece:
                    yield piece
        except Exception as exc:
            raise AIUnavailable(friendly_ai_error(exc)) from exc

    def complete_chat(self, system, messages, tools=None):
        client, contents = self._prepare(system, messages)
        try:
            resp = client.models.generate_content(model=self.model_id, contents=contents,
                                                   config=self._config(system))
        except Exception as exc:
            raise AIUnavailable(friendly_ai_error(exc)) from exc
        return ((getattr(resp, "text", "") or "").strip(), [])


def make_provider(spec: ModelSpec) -> AIProvider:
    if spec.provider == "gemini":
        return GeminiProvider(spec)
    return GroqProvider(spec)


def ai_diagnostics() -> Dict[str, Any]:
    return {
        "app_version": APP_VERSION,
        "groq_package": Groq is not None,
        "gemini_package": google_genai is not None,
        "groq_key_configured": bool(provider_key("groq")),
        "gemini_key_configured": bool(provider_key("gemini")),
        "telegram_configured": bool(secret("TELEGRAM_BOT_TOKEN") and secret("TELEGRAM_CHAT_ID")),
    }


# ===========================================================================
# SECTION 2 - CURRENCY, NUMBER VALIDATION, FORMATTING
# ---------------------------------------------------------------------------
# Every instrument is displayed in ITS OWN currency. Prices are never
# silently converted and never rendered with a foreign symbol.
# ===========================================================================
CURRENCY_SYMBOLS: Dict[str, str] = {
    "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "HKD": "HK$",
    "AUD": "A$", "CAD": "C$", "SGD": "S$", "KRW": "₩", "CNY": "¥", "CHF": "CHF ",
    "TWD": "NT$", "NZD": "NZ$", "SEK": "SEK ", "NOK": "NOK ", "DKK": "DKK ",
    "ZAR": "R", "BRL": "R$", "MXN": "MX$", "AED": "AED ", "SAR": "SAR ",
    "THB": "฿", "IDR": "Rp", "MYR": "RM", "PHP": "₱", "VND": "₫", "PLN": "zł",
    "TRY": "₺", "ILS": "₪", "RUB": "₽", "HUF": "Ft", "CZK": "Kč", "CLP": "CLP ",
    "ARS": "AR$", "NOK ": "NOK ", "VND ": "₫",
}

SUFFIX_CURRENCY: Dict[str, str] = {
    ".NS": "INR", ".BO": "INR", ".TO": "CAD", ".V": "CAD", ".L": "GBP", ".IL": "GBP",
    ".DE": "EUR", ".PA": "EUR", ".AS": "EUR", ".MI": "EUR", ".MC": "EUR", ".LS": "EUR",
    ".IR": "EUR", ".HE": "EUR", ".VI": "EUR", ".ST": "SEK", ".OL": "NOK", ".CO": "DKK",
    ".SW": "CHF", ".T": "JPY", ".HK": "HKD", ".AX": "AUD", ".NZ": "NZD", ".SI": "SGD",
    ".KS": "KRW", ".KQ": "KRW", ".SS": "CNY", ".SZ": "CNY", ".TW": "TWD", ".JO": "ZAR",
    ".SA": "BRL", ".MX": "MXN", ".BA": "ARS", ".IS": "TRY", ".WA": "PLN", ".PR": "CZK",
}

INDEX_CURRENCY: Dict[str, str] = {
    "^NSEI": "INR", "^BSESN": "INR", "^GSPC": "USD", "^DJI": "USD", "^IXIC": "USD",
    "^RUT": "USD", "^VIX": "USD", "^TNX": "USD", "^FTSE": "GBP", "^FTMC": "GBP",
    "^GDAXI": "EUR", "^FCHI": "EUR", "^STOXX50E": "EUR", "^N225": "JPY", "^HSI": "HKD",
    "^AXJO": "AUD", "^STI": "SGD", "^KS11": "KRW", "^SSEC": "CNY", "^GSPTSE": "CAD",
    "^MXX": "MXN", "^BVSP": "BRL", "000001.SS": "CNY", "399001.SZ": "CNY",
}


def _num(value: Any) -> Optional[float]:
    """float-or-None that rejects NaN / inf / junk strings."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").strip()
        if cleaned in ("", "-", "--", "N/A", "n/a", "None", "null"):
            return None
        value = cleaned
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def valid_price(value: Any) -> Optional[float]:
    """A usable price: finite and strictly positive (never 0.0, never NaN)."""
    number = _num(value)
    return number if number is not None and number > 0 else None


def clamp_number_input(value: Any, minimum: float = 0.0, fallback: float = 0.0) -> float:
    """Safe value for st.number_input(min_value=...)."""
    number = _num(value)
    if number is None or number < minimum:
        return float(fallback)
    return float(number)


def _group_indian(int_digits: str) -> str:
    if len(int_digits) <= 3:
        return int_digits
    head, tail = int_digits[:-3], int_digits[-3:]
    parts: List[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def fmt_price(value: Any, currency: str = "USD", decimals: Optional[int] = None,
              with_code: bool = False) -> str:
    """Format a price in its own currency. INR uses Indian digit grouping."""
    number = _num(value)
    if number is None or number <= 0:
        return "—"
    currency = (currency or "USD").upper()
    if decimals is None:
        decimals = 4 if abs(number) < 1 else (2 if abs(number) < 1000 else 2)
    symbol = CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    if currency == "INR":
        digits = f"{abs(number):.{decimals}f}"
        int_part, _, frac = digits.partition(".")
        body = _group_indian(int_part) + (f".{frac}" if frac else "")
        text = f"{'−' if number < 0 else ''}₹{body}"
    else:
        text = f"{symbol}{number:,.{decimals}f}"
    return f"{text} {currency}" if with_code else text


def fmt_pct(value: Any, decimals: int = 2) -> str:
    number = _num(value)
    if number is None:
        return "—"
    return f"{'+' if number >= 0 else ''}{number:.{decimals}f}%"


def fmt_volume(value: Any) -> str:
    number = _num(value)
    if number is None or number < 0:
        return "—"
    if number >= 1e12:
        return f"{number / 1e12:.2f}T"
    if number >= 1e9:
        return f"{number / 1e9:.2f}B"
    if number >= 1e6:
        return f"{number / 1e6:.2f}M"
    if number >= 1e3:
        return f"{number / 1e3:.1f}K"
    return f"{number:,.0f}"


def fmt_cap(value: Any, currency: str = "USD") -> str:
    """Market capitalisation, compacted, in the instrument's own currency."""
    number = _num(value)
    if number is None or number <= 0:
        return "—"
    symbol = CURRENCY_SYMBOLS.get((currency or "USD").upper(), f"{currency} ")
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if number >= scale:
            return f"{symbol}{number / scale:,.2f}{suffix}"
    return f"{symbol}{number:,.0f}"


def guess_currency(yf_symbol: str, exchange_key: str = "", market_key: str = "") -> str:
    """Deterministic currency inference: suffix → market → instrument type."""
    raw = (yf_symbol or "").strip().upper()
    if not raw:
        return EXCHANGES.get(exchange_key, ExchangeFallback).currency if exchange_key else "USD"
    for item in MARKET_ITEMS:
        if raw in (item["yf"].upper(), item["symbol"].upper()):
            return item["currency"]
    if raw.startswith("^"):
        return INDEX_CURRENCY.get(raw, "USD")
    if raw.endswith("-USD"):
        return "USD"
    if raw.endswith("=F"):
        return "USD"
    if raw.endswith("=X"):
        tail = raw[-6:-3] if len(raw) >= 6 else ""
        return tail if tail in CURRENCY_SYMBOLS else "USD"
    for suffix, currency in SUFFIX_CURRENCY.items():
        if raw.endswith(suffix):
            return currency
    if exchange_key and exchange_key in EXCHANGES:
        return EXCHANGES[exchange_key].currency
    if market_key:
        for market in MARKETS:
            if market.key == market_key:
                return market.currency
    return "USD"


def currency_display(currency: str) -> str:
    code = (currency or "USD").upper()
    return f"{CURRENCY_SYMBOLS.get(code, code + ' ')}{code}"


def _shorten(text: Any, limit: int = 20) -> str:
    """Trim a display label on a word boundary (used on heat-map tiles)."""
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    cut = value[:limit]
    if " " in cut:
        cut = cut[:cut.rfind(" ")]
    return cut.rstrip(" ,.-–") + "…"


# Sector labels for instruments the provider gives no sector for (crypto, futures,
# FX, indices) so the heat map never shows an "Unclassified" blob for them.
MARKET_SECTOR_FALLBACK: Dict[str, str] = {
    "crypto": "Crypto",
    "commodities": "Commodities",
    "forex": "Forex",
    "indices": "Indices",
}


def catalog_meta(yf_symbol: str) -> Dict[str, str]:
    """Bundled name / sector / currency for a symbol, when the catalog knows it."""
    raw = (yf_symbol or "").strip().upper()
    item = MARKET_ITEMS_BY_NAME.get(raw)
    if not item:
        item = MARKET_ITEMS_BY_NAME.get(raw.split(".")[0])
    if not item:
        return {}
    return {"name": item.get("name") or "", "sector": item.get("sector") or "",
            "currency": item.get("currency") or ""}


# ===========================================================================
# SECTION 3 - MARKET / EXCHANGE / ASSET CATALOG
# ---------------------------------------------------------------------------
# Only markets/exchanges that the configured provider (Yahoo Finance via
# yfinance) can actually return data for are exposed.
# ===========================================================================
@dataclass(frozen=True)
class Exchange:
    key: str
    market: str
    name: str
    tz: str
    currency: str
    suffix: str
    open_at: str
    close_at: str
    pre_at: str = ""
    after_at: str = ""
    session_kind: str = "regular"          # regular | continuous | extended24x5
    screener_region: str = ""
    screener_exchanges: Tuple[str, ...] = ()
    index_symbol: str = ""
    index_label: str = ""

    @property
    def is_screener_capable(self) -> bool:
        return bool(self.screener_exchanges) and self.session_kind == "regular"


ExchangeFallback = Exchange("USD", "global", "Unknown exchange", "UTC", "USD", "", "00:00", "23:59")


EXCHANGE_SPECS: List[Dict[str, Any]] = [
    dict(key="NSE", market="india", name="NSE · National Stock Exchange of India",
         tz="Asia/Kolkata", currency="INR", suffix=".NS", open_at="09:15", close_at="15:30",
         pre_at="09:00", after_at="15:45", screener_region="in", screener_exchanges=("NSI",),
         index_symbol="^NSEI", index_label="NIFTY 50"),
    dict(key="BSE", market="india", name="BSE · Bombay Stock Exchange",
         tz="Asia/Kolkata", currency="INR", suffix=".BO", open_at="09:15", close_at="15:30",
         pre_at="09:00", after_at="15:45", screener_region="in", screener_exchanges=("BSE",),
         index_symbol="^BSESN", index_label="BSE SENSEX"),
    dict(key="NASDAQ", market="us", name="NASDAQ (US)",
         tz="America/New_York", currency="USD", suffix="", open_at="09:30", close_at="16:00",
         pre_at="04:00", after_at="20:00", screener_region="us", screener_exchanges=("NMS", "NCM"),
         index_symbol="^IXIC", index_label="NASDAQ Composite"),
    dict(key="NYSE", market="us", name="NYSE (US)",
         tz="America/New_York", currency="USD", suffix="", open_at="09:30", close_at="16:00",
         pre_at="04:00", after_at="20:00", screener_region="us", screener_exchanges=("NYQ",),
         index_symbol="^GSPC", index_label="S&P 500"),
    dict(key="AMEX", market="us", name="NYSE American / AMEX",
         tz="America/New_York", currency="USD", suffix="", open_at="09:30", close_at="16:00",
         pre_at="04:00", after_at="20:00", screener_region="us", screener_exchanges=("ASE",),
         index_symbol="^RUT", index_label="Russell 2000"),
    dict(key="TSX", market="canada", name="TSX · Toronto Stock Exchange",
         tz="America/Toronto", currency="CAD", suffix=".TO", open_at="09:30", close_at="16:00",
         pre_at="07:00", after_at="17:00", screener_region="ca", screener_exchanges=("TOR",),
         index_symbol="^GSPTSE", index_label="S&P/TSX Composite"),
    dict(key="LSE", market="uk", name="LSE · London Stock Exchange",
         tz="Europe/London", currency="GBP", suffix=".L", open_at="08:00", close_at="16:30",
         pre_at="07:00", after_at="17:15", screener_region="gb", screener_exchanges=("LSE",),
         index_symbol="^FTSE", index_label="FTSE 100"),
    dict(key="XETRA", market="germany", name="XETRA · Deutsche Börse",
         tz="Europe/Berlin", currency="EUR", suffix=".DE", open_at="09:00", close_at="17:30",
         pre_at="08:00", after_at="20:00", screener_region="de", screener_exchanges=("GER",),
         index_symbol="^GDAXI", index_label="DAX"),
    dict(key="TSE", market="japan", name="TSE · Tokyo Stock Exchange",
         tz="Asia/Tokyo", currency="JPY", suffix=".T", open_at="09:00", close_at="15:00",
         pre_at="08:00", after_at="15:30", screener_region="jp", screener_exchanges=("JPX",),
         index_symbol="^N225", index_label="Nikkei 225"),
    dict(key="HKEX", market="hongkong", name="HKEX · Hong Kong Exchange",
         tz="Asia/Hong_Kong", currency="HKD", suffix=".HK", open_at="09:30", close_at="16:00",
         pre_at="09:00", after_at="16:30", screener_region="hk", screener_exchanges=("HKG",),
         index_symbol="^HSI", index_label="Hang Seng"),
    dict(key="ASX", market="australia", name="ASX · Australian Securities Exchange",
         tz="Australia/Sydney", currency="AUD", suffix=".AX", open_at="10:00", close_at="16:00",
         pre_at="09:00", after_at="18:00", screener_region="au", screener_exchanges=("ASX",),
         index_symbol="^AXJO", index_label="S&P/ASX 200"),
    dict(key="SGX", market="singapore", name="SGX · Singapore Exchange",
         tz="Asia/Singapore", currency="SGD", suffix=".SI", open_at="09:00", close_at="17:00",
         pre_at="08:30", after_at="17:15", screener_region="sg", screener_exchanges=("SES",),
         index_symbol="^STI", index_label="Straits Times Index"),
    dict(key="KRX", market="korea", name="KRX · Korea Exchange",
         tz="Asia/Seoul", currency="KRW", suffix=".KS", open_at="09:00", close_at="15:30",
         pre_at="08:30", after_at="16:00", screener_region="kr", screener_exchanges=("KSC",),
         index_symbol="^KS11", index_label="KOSPI"),
    dict(key="SSE", market="china", name="SSE · Shanghai Stock Exchange",
         tz="Asia/Shanghai", currency="CNY", suffix=".SS", open_at="09:30", close_at="15:00",
         pre_at="09:15", after_at="15:30", screener_region="cn", screener_exchanges=("SHH",),
         index_symbol="000001.SS", index_label="SSE Composite"),
    dict(key="SZSE", market="china", name="SZSE · Shenzhen Stock Exchange",
         tz="Asia/Shanghai", currency="CNY", suffix=".SZ", open_at="09:30", close_at="15:00",
         pre_at="09:15", after_at="15:30", screener_region="cn", screener_exchanges=("SHZ",),
         index_symbol="399001.SZ", index_label="SZSE Component"),
    dict(key="CRYPTO", market="crypto", name="Crypto (24/7)",
         tz="UTC", currency="USD", suffix="-USD", open_at="00:00", close_at="23:59",
         session_kind="continuous", index_symbol="", index_label=""),
    dict(key="CME", market="commodities", name="CME / COMEX / NYMEX futures",
         tz="America/Chicago", currency="USD", suffix="=F", open_at="17:00", close_at="16:00",
         session_kind="extended24x5", index_symbol="", index_label=""),
    dict(key="FX", market="forex", name="Global FX (interbank)",
         tz="UTC", currency="USD", suffix="=X", open_at="00:00", close_at="23:59",
         session_kind="extended24x5", index_symbol="", index_label=""),
    dict(key="GLOBAL", market="indices", name="Global indices",
         tz="UTC", currency="USD", suffix="", open_at="00:00", close_at="23:59",
         session_kind="extended24x5", index_symbol="", index_label=""),
]

EXCHANGES: Dict[str, Exchange] = {spec["key"]: Exchange(**spec) for spec in EXCHANGE_SPECS}


@dataclass(frozen=True)
class Market:
    key: str
    label: str
    asset_class: str
    currency: str
    exchanges: Tuple[str, ...]
    flag: str = ""


MARKETS: List[Market] = [
    Market("india", "India", "Equity", "INR", ("NSE", "BSE"), "🇮🇳"),
    Market("us", "United States", "Equity", "USD", ("NASDAQ", "NYSE", "AMEX"), "🇺🇸"),
    Market("canada", "Canada", "Equity", "CAD", ("TSX",), "🇨🇦"),
    Market("uk", "United Kingdom", "Equity", "GBP", ("LSE",), "🇬🇧"),
    Market("germany", "Germany", "Equity", "EUR", ("XETRA",), "🇩🇪"),
    Market("japan", "Japan", "Equity", "JPY", ("TSE",), "🇯🇵"),
    Market("hongkong", "Hong Kong", "Equity", "HKD", ("HKEX",), "🇭🇰"),
    Market("australia", "Australia", "Equity", "AUD", ("ASX",), "🇦🇺"),
    Market("singapore", "Singapore", "Equity", "SGD", ("SGX",), "🇸🇬"),
    Market("korea", "South Korea", "Equity", "KRW", ("KRX",), "🇰🇷"),
    Market("china", "China", "Equity", "CNY", ("SSE", "SZSE"), "🇨🇳"),
    Market("crypto", "Crypto", "Crypto", "USD", ("CRYPTO",), "🪙"),
    Market("commodities", "Commodities", "Commodity", "USD", ("CME",), "🛢️"),
    Market("forex", "Forex", "Forex", "USD", ("FX",), "💱"),
    Market("indices", "Global Indices", "Index", "USD", ("GLOBAL",), "🌐"),
]

MARKETS_BY_KEY: Dict[str, Market] = {m.key: m for m in MARKETS}

MARKET_CATEGORIES = [
    {"key": "india", "label": "India NSE"},
    {"key": "us", "label": "US Stocks"},
    {"key": "crypto", "label": "Crypto"},
    {"key": "commodities", "label": "Commodities"},
    {"key": "forex", "label": "Forex"},
    {"key": "indices", "label": "Global Indices"},
]

# Curated quick-pick instruments (preserved from the original build, now with
# an explicit currency so nothing is ever rendered with the wrong symbol).
MARKET_ITEMS: List[Dict[str, Any]] = [
    {"symbol": "RELIANCE", "name": "Reliance Industries", "category": "india", "market": "india", "sector": "Energy", "yf": "RELIANCE.NS", "currency": "INR"},
    {"symbol": "TCS", "name": "Tata Consultancy", "category": "india", "market": "india", "sector": "IT", "yf": "TCS.NS", "currency": "INR"},
    {"symbol": "INFY", "name": "Infosys", "category": "india", "market": "india", "sector": "IT", "yf": "INFY.NS", "currency": "INR"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank", "category": "india", "market": "india", "sector": "Banking", "yf": "HDFCBANK.NS", "currency": "INR"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank", "category": "india", "market": "india", "sector": "Banking", "yf": "ICICIBANK.NS", "currency": "INR"},
    {"symbol": "SBIN", "name": "State Bank of India", "category": "india", "market": "india", "sector": "Banking", "yf": "SBIN.NS", "currency": "INR"},
    {"symbol": "TATAMOTORS", "name": "Tata Motors", "category": "india", "market": "india", "sector": "Auto", "yf": "TATAMOTORS.NS", "currency": "INR"},
    {"symbol": "ITC", "name": "ITC Ltd", "category": "india", "market": "india", "sector": "FMCG", "yf": "ITC.NS", "currency": "INR"},
    {"symbol": "SUNPHARMA", "name": "Sun Pharma", "category": "india", "market": "india", "sector": "Pharma", "yf": "SUNPHARMA.NS", "currency": "INR"},
    {"symbol": "TATASTEEL", "name": "Tata Steel", "category": "india", "market": "india", "sector": "Metal", "yf": "TATASTEEL.NS", "currency": "INR"},
    {"symbol": "AAPL", "name": "Apple", "category": "us", "market": "us", "sector": "US Tech", "yf": "AAPL", "currency": "USD"},
    {"symbol": "MSFT", "name": "Microsoft", "category": "us", "market": "us", "sector": "US Tech", "yf": "MSFT", "currency": "USD"},
    {"symbol": "NVDA", "name": "NVIDIA", "category": "us", "market": "us", "sector": "US Tech", "yf": "NVDA", "currency": "USD"},
    {"symbol": "GOOGL", "name": "Alphabet", "category": "us", "market": "us", "sector": "US Tech", "yf": "GOOGL", "currency": "USD"},
    {"symbol": "AMZN", "name": "Amazon", "category": "us", "market": "us", "sector": "US Tech", "yf": "AMZN", "currency": "USD"},
    {"symbol": "META", "name": "Meta", "category": "us", "market": "us", "sector": "US Tech", "yf": "META", "currency": "USD"},
    {"symbol": "TSLA", "name": "Tesla", "category": "us", "market": "us", "sector": "US Auto", "yf": "TSLA", "currency": "USD"},
    {"symbol": "JPM", "name": "JPMorgan", "category": "us", "market": "us", "sector": "US Banking", "yf": "JPM", "currency": "USD"},
    {"symbol": "BTC", "name": "Bitcoin", "category": "crypto", "market": "crypto", "sector": "Crypto", "yf": "BTC-USD", "currency": "USD"},
    {"symbol": "ETH", "name": "Ethereum", "category": "crypto", "market": "crypto", "sector": "Crypto", "yf": "ETH-USD", "currency": "USD"},
    {"symbol": "BNB", "name": "BNB", "category": "crypto", "market": "crypto", "sector": "Crypto", "yf": "BNB-USD", "currency": "USD"},
    {"symbol": "SOL", "name": "Solana", "category": "crypto", "market": "crypto", "sector": "Crypto", "yf": "SOL-USD", "currency": "USD"},
    {"symbol": "XRP", "name": "XRP", "category": "crypto", "market": "crypto", "sector": "Crypto", "yf": "XRP-USD", "currency": "USD"},
    {"symbol": "ADA", "name": "Cardano", "category": "crypto", "market": "crypto", "sector": "Crypto", "yf": "ADA-USD", "currency": "USD"},
    {"symbol": "DOGE", "name": "Dogecoin", "category": "crypto", "market": "crypto", "sector": "Crypto", "yf": "DOGE-USD", "currency": "USD"},
    {"symbol": "GC", "name": "Gold", "category": "commodities", "market": "commodities", "sector": "Commodities", "yf": "GC=F", "currency": "USD"},
    {"symbol": "SI", "name": "Silver", "category": "commodities", "market": "commodities", "sector": "Commodities", "yf": "SI=F", "currency": "USD"},
    {"symbol": "CL", "name": "Crude Oil", "category": "commodities", "market": "commodities", "sector": "Commodities", "yf": "CL=F", "currency": "USD"},
    {"symbol": "NG", "name": "Natural Gas", "category": "commodities", "market": "commodities", "sector": "Commodities", "yf": "NG=F", "currency": "USD"},
    {"symbol": "HG", "name": "Copper", "category": "commodities", "market": "commodities", "sector": "Commodities", "yf": "HG=F", "currency": "USD"},
    {"symbol": "USDINR", "name": "USD/INR", "category": "forex", "market": "forex", "sector": "Forex", "yf": "USDINR=X", "currency": "INR"},
    {"symbol": "EURUSD", "name": "EUR/USD", "category": "forex", "market": "forex", "sector": "Forex", "yf": "EURUSD=X", "currency": "USD"},
    {"symbol": "GBPUSD", "name": "GBP/USD", "category": "forex", "market": "forex", "sector": "Forex", "yf": "GBPUSD=X", "currency": "USD"},
    {"symbol": "USDJPY", "name": "USD/JPY", "category": "forex", "market": "forex", "sector": "Forex", "yf": "USDJPY=X", "currency": "JPY"},
    {"symbol": "AUDUSD", "name": "AUD/USD", "category": "forex", "market": "forex", "sector": "Forex", "yf": "AUDUSD=X", "currency": "USD"},
    {"symbol": "NIFTY", "name": "Nifty 50", "category": "indices", "market": "indices", "sector": "Indices", "yf": "^NSEI", "currency": "INR"},
    {"symbol": "SENSEX", "name": "Sensex", "category": "indices", "market": "indices", "sector": "Indices", "yf": "^BSESN", "currency": "INR"},
    {"symbol": "SPX", "name": "S&P 500", "category": "indices", "market": "indices", "sector": "Indices", "yf": "^GSPC", "currency": "USD"},
    {"symbol": "DJI", "name": "Dow Jones", "category": "indices", "market": "indices", "sector": "Indices", "yf": "^DJI", "currency": "USD"},
    {"symbol": "IXIC", "name": "Nasdaq", "category": "indices", "market": "indices", "sector": "Indices", "yf": "^IXIC", "currency": "USD"},
]
MARKET_ITEMS_BY_NAME: Dict[str, Dict[str, Any]] = {}
for _item in MARKET_ITEMS:
    MARKET_ITEMS_BY_NAME[_item["symbol"].upper()] = _item
    MARKET_ITEMS_BY_NAME[_item["yf"].upper()] = _item
    MARKET_ITEMS_BY_NAME[_item["name"].upper()] = _item


# ---------------------------------------------------------------------------
# Session / market status per exchange (exchange-local timezone, own calendar)
# ---------------------------------------------------------------------------
def _tzinfo(name: str):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception as exc:
            log_exception(f"timezone {name}", exc)
    return timezone.utc


def _parse_hhmm(text: str) -> Optional[dtime]:
    try:
        hour, minute = str(text).split(":")
        return dtime(int(hour), int(minute))
    except Exception:
        return None


def session_status(exchange_key: str, reference: Optional[datetime] = None) -> Dict[str, Any]:
    """OPEN / PRE-MARKET / AFTER-HOURS / CLOSED / UNKNOWN for one exchange."""
    exch = EXCHANGES.get(exchange_key)
    if exch is None:
        return {"exchange": exchange_key, "label": exchange_key, "status": "UNKNOWN",
                "reason": "Unknown exchange - no session calendar configured.",
                "local_time": "—", "tz": "UTC", "session_date": "", "is_open": False}
    tz = _tzinfo(exch.tz)
    now = reference.astimezone(tz) if reference else datetime.now(tz)
    open_t, close_t = _parse_hhmm(exch.open_at), _parse_hhmm(exch.close_at)
    weekday = now.weekday()

    if exch.session_kind == "continuous":
        status, reason = "OPEN", "Continuous 24/7 session."
    elif exch.session_kind == "extended24x5":
        if weekday == 5 or (weekday == 4 and now.time() >= dtime(17, 0)) or \
           (weekday == 6 and now.time() < dtime(17, 0)):
            status, reason = "CLOSED", "Weekend close of the continuous weekday session."
        else:
            status, reason = "OPEN", "Continuous weekday session (Sun 17:00 → Fri 17:00 convention)."
    elif weekday >= 5:
        status, reason = "CLOSED", "Weekend - the exchange does not trade today."
    elif open_t and close_t and open_t <= now.time() <= close_t:
        status, reason = "OPEN", "Inside regular session hours."
    elif open_t and now.time() < open_t:
        pre_t = _parse_hhmm(exch.pre_at) if exch.pre_at else None
        if pre_t and pre_t <= now.time():
            status, reason = "PRE-MARKET", "Inside the pre-market window."
        else:
            status, reason = "CLOSED", "Before the session opens."
    elif close_t and now.time() > close_t:
        after_t = _parse_hhmm(exch.after_at) if exch.after_at else None
        if after_t and now.time() <= after_t:
            status, reason = "AFTER-HOURS", "Inside the after-hours window."
        else:
            status, reason = "CLOSED", "Session finished for today."
    else:  # pragma: no cover - defensive
        status, reason = "UNKNOWN", "Session calendar incomplete."

    return {
        "exchange": exch.key,
        "label": exch.name,
        "status": status,
        "reason": reason,
        "is_open": status in ("OPEN", "PRE-MARKET", "AFTER-HOURS"),
        "local_time": now.strftime("%a %d %b %Y %H:%M") + f" ({now.tzname() or exch.tz})",
        "local_iso": now.isoformat(),
        "session_date": now.date().isoformat(),
        "tz": exch.tz,
        "currency": exch.currency,
    }


def expected_last_session_date(exchange_key: str) -> str:
    """The most recent calendar date on which this exchange should have traded."""
    exch = EXCHANGES.get(exchange_key)
    tz = _tzinfo(exch.tz) if exch else timezone.utc
    now = datetime.now(tz)
    if exch and exch.session_kind == "continuous":
        return now.date().isoformat()
    probe = now
    for _ in range(10):
        status = session_status(exchange_key, probe)
        if status["status"] in ("CLOSED",) and probe.date() != now.date():
            return probe.date().isoformat()
        if status["status"] == "CLOSED" and probe.time() < dtime(9, 0):
            probe = probe - timedelta(days=1)
            continue
        if status["status"] == "OPEN":
            probe = probe - timedelta(days=1)
            continue
        break
    days_back = 1
    while days_back < 10:
        candidate = now - timedelta(days=days_back)
        if candidate.weekday() < 5:
            return candidate.date().isoformat()
        days_back += 1
    return now.date().isoformat()


# ===========================================================================
# SECTION 4 - UNIVERSES  (index proxies / dynamic exchange queries / curated)
# ===========================================================================
@dataclass(frozen=True)
class Universe:
    key: str
    label: str
    exchange: str
    kind: str                       # index_proxy | screener_cap | screener_all | curated
    size: int = 100
    fallback: Tuple[str, ...] = ()
    note: str = ""


CURATED_FALLBACKS: Dict[str, Tuple[str, ...]] = {
    "NSE": tuple("RELIANCE TCS HDFCBANK ICICIBANK INFY SBIN BHARTIARTL ITC LT HINDUNILVR "
                 "KOTAKBANK AXISBANK BAJFINANCE MARUTI ASIANPAINT HCLTECH WIPRO SUNPHARMA TITAN "
                 "ULTRACEMCO POWERGRID NTPC TATAMOTORS TATASTEEL JSWSTEEL M&M ADANIENT ADANIPORTS "
                 "COALINDIA ONGC GRASIM HINDALCO DRREDDY CIPLA DIVISLAB NESTLEIND BRITANNIA TECHM "
                 "LTIM BAJAJFINSV BAJAJ-AUTO HEROMOTOCO EICHERMOT INDUSINDBK APOLLOHOSP BPCL "
                 "SHREECEM SBILIFE HDFCLIFE TATACONSUM TRENT".split()),
    "BSE": tuple("RELIANCE TCS HDFCBANK INFY SBIN ITC LT TATAMOTORS AXISBANK BAJAJFINSV "
                 "RELIANCE TATASTEEL ADANIENT HINDALCO IOC BPCL".split()),
    "NASDAQ": tuple("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO COST NFLX AMD PEP ADBE CSCO "
                    "INTC QCOM TXN AMAT INTU BKNG ISRG HON MU LRCX ADI PANW KLAC SNPS CDNS "
                    "MRVL PYPL CMCSA TMUS AMGN GILD MDLZ".split()),
    "NYSE": tuple("JPM V UNH XOM JNJ WMT PG MA HD BAC KO ORCL CVX ABBV MRK WFC DIS "
                  "CSCO PFE TMO ABT NKE MCD DHR LIN T CAT GS HON IBM AXP BLK".split()),
    "AMEX": tuple("IWM SPY GLD SLV USO".split()),
    "TSX": tuple("RY TD ENB BNS CNR CP SU TRP BMO MFC ABX WCN CSU ATD CNQ BCE NTR SHOP "
                 "TRI IMO FTS POW".split()),
    "LSE": tuple("SHEL AZN HSBA ULVR BP BATS GSK REL LLOY RIO BARC DGE GLEN TSCO NG "
                 "VOD BAE PRU IHG AAL".split()),
    "XETRA": tuple("SAP SIE ALV DTE MBG BMW AIR VOW3 BAS BAYN MUV2 RWE IFX DB1 HEN3 "
                   "FRE ADS MRK1 LIN BEI".split()),
    "TSE": tuple("7203 6758 6861 9984 8306 6098 4063 8035 7974 9432 8031 6501 4502 "
                 "7267 8058 9983 4519 2914 6594 6367".split()),
    "HKEX": tuple("0700 0005 0941 1299 9988 3690 0388 2318 1810 9618 0016 2020 0883 "
                  "0386 1113 0001 2382 6690 2269 1093".split()),
    "ASX": tuple("BHP CBA CSL NAB WBC ANZ WES MQG WOW FMG TLS RIO TCL ORG GMG STO "
                 "SUN QBE ALL JHX".split()),
    "SGX": tuple("D05 O39 U11 Z74 C38U A17U S68 G13 BUOU Y92 C6L S63 U96 K71 N2IU T39".split()),
    "KRX": tuple("005930 000660 373220 207940 005380 000270 068270 006400 051910 035420 "
                 "005490 012330 105560 055550 015760 017670 096770 032830 009150 034730".split()),
    "SSE": tuple("600519 601318 600036 600030 601398 601288 601857 600028 601988 600900 "
                 "601088 600276 600309 601012 601166".split()),
    "SZSE": tuple("000001 000002 000858 002594 300750 000333 002415 300059 002304 000651".split()),
}

UNIVERSES: Dict[str, Universe] = {}


def _register_universe(universe: Universe) -> None:
    UNIVERSES[universe.key] = universe


for _ex in EXCHANGES.values():
    _fallback = CURATED_FALLBACKS.get(_ex.key, ())
    if _ex.index_symbol:
        _register_universe(Universe(
            key=f"{_ex.key}_INDEX",
            label=f"{_ex.index_label} · index proxy (largest constituents)",
            exchange=_ex.key, kind="index_proxy", size=50, fallback=_fallback,
            note=("The data provider does not publish index constituent lists, so this universe is a live "
                  "liquidity/market-cap proxy for the index - not the official constituent set.")))
    if _ex.is_screener_capable:
        _register_universe(Universe(key=f"{_ex.key}_TOP50", label="Top 50 by market cap (live query)",
                                    exchange=_ex.key, kind="screener_cap", size=50, fallback=_fallback))
        _register_universe(Universe(key=f"{_ex.key}_TOP100", label="Top 100 by market cap (live query)",
                                    exchange=_ex.key, kind="screener_cap", size=100, fallback=_fallback))
        _register_universe(Universe(key=f"{_ex.key}_TOP250", label="Top 250 by market cap (live query)",
                                    exchange=_ex.key, kind="screener_cap", size=250, fallback=_fallback))
        _register_universe(Universe(key=f"{_ex.key}_ALL", label="All available equities (live query, capped)",
                                    exchange=_ex.key, kind="screener_all", size=300, fallback=_fallback,
                                    note="The live exchange-wide query is requested in pages; the UI caps the "
                                         "result so a browser session stays responsive."))
    if _fallback:
        _register_universe(Universe(key=f"{_ex.key}_CURATED", label="Built-in reference list (offline fallback)",
                                    exchange=_ex.key, kind="curated", size=len(_fallback), fallback=_fallback,
                                    note="Static list bundled with the app; used only when the live provider "
                                         "cannot be reached."))

for _category, _key, _label in (
    ("crypto", "CRYPTO_MAJORS", "Crypto majors (built-in list)"),
    ("commodities", "CME_FUTURES", "Commodity futures (built-in list)"),
    ("forex", "FX_MAJORS", "FX majors (built-in list)"),
    ("indices", "GLOBAL_INDICES", "Global indices (built-in list)"),
):
    _symbols = tuple(item["yf"] for item in MARKET_ITEMS if item["category"] == _category)
    _exchange_key = {"crypto": "CRYPTO", "commodities": "CME", "forex": "FX", "indices": "GLOBAL"}[_category]
    _register_universe(Universe(key=_key, label=_label, exchange=_exchange_key, kind="curated",
                                size=len(_symbols), fallback=_symbols))


def universes_for_exchange(exchange_key: str) -> List[Universe]:
    return sorted([u for u in UNIVERSES.values() if u.exchange == exchange_key], key=lambda u: u.key)


def resolve_universe(exchange_key: str, universe_key: str) -> Universe:
    universe = UNIVERSES.get(universe_key)
    if universe and universe.exchange == exchange_key:
        return universe
    options = universes_for_exchange(exchange_key)
    if options:
        return options[0]
    return Universe(key=f"{exchange_key}_CURATED", label="Built-in reference list",
                    exchange=exchange_key, kind="curated",
                    fallback=CURATED_FALLBACKS.get(exchange_key, ()))


# ===========================================================================
# SECTION 5 - CACHE + LAST-SESSION PERSISTENCE
# ---------------------------------------------------------------------------
# The heat map must survive a closed market, a provider outage and a cold
# start. Snapshots are written to disk (JSON) so a restart can still render
# the last completed session, and the UI always labels where the data came from.
# ===========================================================================
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / ".cache"
SNAPSHOT_FILE = CACHE_DIR / "last_snapshot.json"
SNAPSHOT_VERSION = 1


def ensure_cache_dir() -> Path:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log_exception("cache dir", exc)
    return CACHE_DIR


def snapshot_key(exchange_key: str, universe_key: str) -> str:
    return f"{exchange_key}|{universe_key}"


def _read_snapshot_file() -> Dict[str, Any]:
    try:
        if not SNAPSHOT_FILE.exists():
            return {}
        with SNAPSHOT_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and payload.get("_version") == SNAPSHOT_VERSION:
            return payload.get("snapshots", {}) or {}
    except Exception as exc:
        log_exception("snapshot read", exc)
    return {}


def _write_snapshot_file(snapshots: Dict[str, Any]) -> None:
    try:
        ensure_cache_dir()
        trimmed = dict(list(snapshots.items())[-24:])   # keep the file small
        with SNAPSHOT_FILE.open("w", encoding="utf-8") as handle:
            json.dump({"_version": SNAPSHOT_VERSION, "snapshots": trimmed}, handle)
    except Exception as exc:
        log_exception("snapshot write", exc)


def load_snapshot(exchange_key: str, universe_key: str) -> Optional[Dict[str, Any]]:
    """Last successful normalized snapshot for this exchange/universe."""
    key = snapshot_key(exchange_key, universe_key)
    memory = st.session_state.get("_snapshots")
    if isinstance(memory, dict) and key in memory:
        return memory[key]
    return _read_snapshot_file().get(key)


def save_snapshot(exchange_key: str, universe_key: str, rows: List[Dict[str, Any]],
                  data_status: str, session_date: str, currency: str) -> None:
    if not rows:
        return
    key = snapshot_key(exchange_key, universe_key)
    snapshots = st.session_state.get("_snapshots")
    if not isinstance(snapshots, dict):
        snapshots = _read_snapshot_file()
    snapshots[key] = {
        "rows": rows,
        "data_status": data_status,
        "session_date": session_date,
        "currency": currency,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "exchange": exchange_key,
        "universe": universe_key,
    }
    st.session_state["_snapshots"] = snapshots
    _write_snapshot_file(snapshots)


def drop_snapshot(exchange_key: str, universe_key: str) -> None:
    key = snapshot_key(exchange_key, universe_key)
    snapshots = st.session_state.get("_snapshots")
    if not isinstance(snapshots, dict):
        snapshots = _read_snapshot_file()
    snapshots.pop(key, None)
    st.session_state["_snapshots"] = snapshots
    _write_snapshot_file(snapshots)


# ===========================================================================
# SECTION 6 - MARKET DATA PROVIDER ABSTRACTION
# ---------------------------------------------------------------------------
# class MarketDataProvider:
#     get_exchanges / get_universes / get_symbols / get_quotes / get_history /
#     get_market_status
# The only implementation shipped is YahooProvider (yfinance + Yahoo screener).
# A real-time feed (broker WebSocket / exchange subscription) can be added by
# implementing the same interface - the UI and the AI tools never change.
# ===========================================================================
DATA_STATUS_ORDER = ("LIVE", "DELAYED", "LAST SESSION", "CACHED", "UNAVAILABLE")


class MarketDataProvider:
    """Interface every data source must implement."""

    key = "base"
    display_name = "Base provider"

    def get_exchanges(self) -> List[Exchange]:
        raise NotImplementedError

    def get_universes(self, exchange_key: str) -> List[Universe]:
        raise NotImplementedError

    def get_symbols(self, exchange_key: str, universe_key: str) -> List[Dict[str, str]]:
        raise NotImplementedError

    def get_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_history(self, yf_symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        raise NotImplementedError

    def get_market_status(self, exchange_key: str) -> Dict[str, Any]:
        return session_status(exchange_key)

    def capabilities(self) -> Dict[str, str]:
        return {}


class YahooProvider(MarketDataProvider):
    key = "yahoo"
    display_name = "Yahoo Finance (yfinance + Yahoo screener)"

    def get_exchanges(self) -> List[Exchange]:
        return list(EXCHANGES.values())

    def get_universes(self, exchange_key: str) -> List[Universe]:
        return universes_for_exchange(exchange_key)

    def capabilities(self) -> Dict[str, str]:
        return {
            "real_time": "Not guaranteed. Yahoo publishes quotes with a provider-dependent delay "
                         "(typically ~15 min for many non-US exchanges, near-live for US equities). "
                         "This app therefore never labels data LIVE unless the exchange session is "
                         "open AND the provider returned a timestamp inside the current session.",
            "delayed": "Yes - the normal case for non-US exchanges.",
            "historical": "Yes - daily and intraday history via yfinance.",
            "full_market": "Partial. The Yahoo screener is queried per exchange in pages (cap applied in the UI); "
                           "index constituents are proxied by a live liquidity/market-cap query.",
            "streaming": "No WebSocket streaming. Refresh + caching only.",
            "market_cap": "Available for most listed equities (used for heat-map tile sizing).",
        }

    # ---- symbol discovery -------------------------------------------------
    def get_symbols(self, exchange_key: str, universe_key: str) -> List[Dict[str, str]]:
        universe = resolve_universe(exchange_key, universe_key)
        cache_buster = int(time.time() // 900)   # 15-minute buckets
        return _symbols_cached(exchange_key, universe.key, cache_buster)

    def _universe_symbols(self, exchange_key: str, universe_key: str) -> List[Dict[str, str]]:
        universe = resolve_universe(exchange_key, universe_key)
        exch = EXCHANGES.get(exchange_key)
        suffix = exch.suffix if exch else ""
        symbols: List[str] = []
        note = universe.note

        if universe.kind in ("screener_cap", "screener_all") and exch and exch.is_screener_capable:
            symbols = _yahoo_screener_symbols(exch, universe.size)
        if not symbols and universe.kind == "index_proxy" and exch:
            symbols = _yahoo_screener_symbols(exch, universe.size) if exch.is_screener_capable else []
        if not symbols:
            symbols = list(universe.fallback)
            if not symbols:
                note = "No symbol source is available for this universe."

        out: List[Dict[str, str]] = []
        seen: set = set()
        for raw in symbols:
            token = str(raw).strip().upper()
            if not token or token in seen:
                continue
            seen.add(token)
            yf_symbol = token if (any(token.endswith(s) for s in SUFFIX_CURRENCY) or token.startswith("^")
                                  or "-" in token or "=" in token) else f"{token}{suffix}"
            meta = catalog_meta(yf_symbol)
            out.append({
                "symbol": token.replace(suffix, "") if suffix and token.endswith(suffix) else token,
                "yf_symbol": yf_symbol,
                "name": meta.get("name") or token,
                "name_source": "catalog" if meta.get("name") else "provider",
                "sector": meta.get("sector") or "",
                "currency": meta.get("currency") or guess_currency(yf_symbol, exchange_key),
                "exchange": exchange_key,
                "universe": universe.key,
                "source": "live screener" if universe.kind.startswith("screener") else
                          ("index proxy" if universe.kind == "index_proxy" else "built-in list"),
                "note": note or "",
            })
        return out

    # ---- quotes -----------------------------------------------------------
    def get_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        return _quotes_cached(tuple(symbols), int(time.time() // 120))

    def get_history(self, yf_symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        return _history_cached(yf_symbol, period, interval)


YAHOO = YahooProvider()
PROVIDER: MarketDataProvider = YAHOO


def _yahoo_screener_symbols(exch: Exchange, size: int) -> List[str]:
    """Live exchange-wide query through the Yahoo screener (paged)."""
    out: List[str] = []
    page_size = 100
    for page in range(0, max(size, 1), page_size):
        try:
            response = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/screener",
                params={
                    "formatted": "true", "lang": "en-US", "region": "US",
                    "crumb": "1", "corsDomain": "finance.yahoo.com",
                },
                json={
                    "size": min(page_size, max(size - page, 1)),
                    "offset": page,
                    "sortField": "intradaymarketcap" if exch.session_kind == "regular" else "intradaymarketcap",
                    "sortType": "DESC",
                    "quoteType": "EQUITY",
                    "topOperator": "AND",
                    "query": {
                        "operator": "AND",
                        "operands": [
                            {"operator": "or", "operands": [
                                {"operator": "EQ", "operands": ["region", exch.screener_region]}]},
                            {"operator": "or", "operands": [
                                {"operator": "EQ", "operands": ["exchange", code]}
                                for code in exch.screener_exchanges]},
                        ],
                    },
                    "userId": "", "userIdType": "guid",
                },
                headers={"User-Agent": HTTP_UA, "Accept": "application/json"},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            quotes = (((payload or {}).get("finance") or {}).get("result") or [{}])[0].get("quotes", []) or []
            for quote in quotes:
                token = quote.get("symbol")
                if token:
                    out.append(str(token))
            if len(quotes) < page_size:
                break
        except Exception as exc:
            log_exception(f"yahoo screener {exch.key} page {page}", exc)
            break
    return out


@st.cache_data(ttl=900, show_spinner=False)
def _symbols_cached(exchange_key: str, universe_key: str, cache_buster: int) -> List[Dict[str, str]]:
    try:
        return YAHOO._universe_symbols(exchange_key, universe_key)
    except Exception as exc:
        log_exception(f"symbols {exchange_key}/{universe_key}", exc)
        universe = resolve_universe(exchange_key, universe_key)
        exch = EXCHANGES.get(exchange_key)
        suffix = exch.suffix if exch else ""
        return [{"symbol": token, "yf_symbol": f"{token}{suffix}", "name": token,
                 "currency": exch.currency if exch else "USD", "exchange": exchange_key,
                 "universe": universe.key, "source": "built-in list", "note": "Live lookup failed."}
                for token in universe.fallback]


def _download_ohlc(symbols: Sequence[str], period: str, interval: str) -> pd.DataFrame:
    """One batched, threaded yfinance download for many symbols."""
    try:
        return yf.download(list(symbols), period=period, interval=interval, group_by="ticker",
                           auto_adjust=False, progress=False, threads=True, timeout=25)
    except Exception as exc:
        log_exception("yf.download batch", exc)
        return pd.DataFrame()


CANONICAL_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def _clean_ohlc(frame: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Flatten whatever column layout the provider returned into canonical OHLCV names.

    WHY THIS EXISTS (the KeyError: 'Close' bug):
    `yf.download([symbol], group_by="ticker")` returns MultiIndex columns
    [('CL=F','Open'), ('CL=F','Close'), ...] even for a SINGLE symbol, while
    `yf.download([a, b], ...)` also returns MultiIndex. The old code only unwrapped the
    MultiIndex when more than one symbol was requested, so every single-instrument quote
    (dashboard, watchlist, alerts, AI tools) hit `frame["Close"]` -> KeyError: 'Close'.
    The layout is now normalized in one place, for every call, and a missing Close column
    is reported as a clean provider message instead of an exception.
    """
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame()
    out = frame
    try:
        if isinstance(out.columns, pd.MultiIndex):
            picked = None
            for level in range(out.columns.nlevels):
                values = [str(v) for v in out.columns.get_level_values(level)]
                if symbol and symbol in values:
                    picked = out.xs(symbol, axis=1, level=level)
                    break
            if picked is None:
                if symbol:
                    # The caller asked for a specific symbol and it is not in this frame.
                    # Returning the other symbols' columns here would silently show the
                    # WRONG instrument's price, so report "no data" instead.
                    return pd.DataFrame()
                picked = out.copy()
                picked.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in picked.columns]
            out = picked
        if isinstance(out.columns, pd.MultiIndex):     # still nested after xs()
            out.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in out.columns]
        rename: Dict[Any, str] = {}
        for column in out.columns:
            key = str(column).strip().lower()
            for canonical in CANONICAL_COLUMNS:
                if key == canonical.lower():
                    rename[column] = canonical
                    break
        if rename:
            out = out.rename(columns=rename)
        out = out.loc[:, ~out.columns.duplicated()]     # keep the first Close, never a DataFrame
        return out.dropna(how="all")
    except Exception as exc:
        log_exception(f"clean ohlc {symbol}", exc)
        return pd.DataFrame()


def _slice_symbol_frame(frame: pd.DataFrame, symbol: str, many: bool = True) -> pd.DataFrame:
    """One symbol's bars from a batch download - works for 1 symbol or N symbols."""
    cleaned = _clean_ohlc(frame, symbol)
    if cleaned.empty:
        return cleaned
    if "Close" not in cleaned.columns:
        return pd.DataFrame()
    return cleaned


@st.cache_data(ttl=120, show_spinner=False)
def _quotes_cached(symbols: Tuple[str, ...], cache_buster: int) -> List[Dict[str, Any]]:
    """Normalized quote rows for many symbols - one bad symbol is skipped, never fatal."""
    symbols = [s for s in symbols if s]
    if not symbols:
        return []
    frame = _download_ohlc(symbols, period="7d", interval="1d")
    rows: List[Dict[str, Any]] = []
    for symbol in symbols:
        try:
            block = _slice_symbol_frame(frame, symbol)
            exch_key = _exchange_for_yf_symbol(symbol)
            row: Dict[str, Any] = {
                "symbol": symbol, "yf_symbol": symbol, "exchange": exch_key,
                "currency": guess_currency(symbol, exch_key), "ok": False, "error": "",
                "source": "yahoo", "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            if block.empty:
                row["error"] = "no rows returned"
                rows.append(row)
                continue
            if "Close" not in block.columns:
                row["error"] = "provider returned no Close column for this instrument"
                rows.append(row)
                continue
            close_column = block["Close"]
            if isinstance(close_column, pd.DataFrame):
                close_column = close_column.iloc[:, 0]
            closes = pd.to_numeric(close_column, errors="coerce").dropna()
            if closes.empty:
                row["error"] = "close column empty"
                rows.append(row)
                continue
            last_close = float(closes.iloc[-1])
            prev_close = float(closes.iloc[-2]) if len(closes) > 1 else last_close
            last_row = block.loc[closes.index[-1]]
            high = valid_price(last_row.get("High")) or last_close
            low = valid_price(last_row.get("Low")) or last_close
            open_price = valid_price(last_row.get("Open")) or last_close
            volume = _num(last_row.get("Volume")) or 0.0
            change = last_close - prev_close
            row.update({
                "price": last_close,
                "previous_close": prev_close,
                "change": change,
                "change_percent": (change / prev_close * 100) if prev_close else 0.0,
                "open": open_price, "day_high": high, "day_low": low,
                "volume": volume,
                "last_bar": str(closes.index[-1])[:19],
                "last_bar_date": str(pd.Timestamp(closes.index[-1]).date()),
                "ok": valid_price(last_close) is not None,
            })
            if not row["ok"]:
                row["error"] = "invalid price (zero/NaN)"
            rows.append(row)
        except Exception as exc:
            log_exception(f"quote {symbol}", exc)
            rows.append({"symbol": symbol, "yf_symbol": symbol, "ok": False,
                         "error": safe_error(exc), "price": None,
                         "currency": guess_currency(symbol), "exchange": _exchange_for_yf_symbol(symbol)})
    return rows


def _history_fetch(yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        ticker = yf.Ticker(yf_symbol)
        frame = ticker.history(period=period, interval=interval, auto_adjust=False)
        return _clean_ohlc(frame, yf_symbol)
    except Exception as exc:
        log_exception(f"history {yf_symbol}", exc)
        return pd.DataFrame()


_history_cached = st.cache_data(ttl=300, show_spinner=False)(_history_fetch)


def _history_safe(yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Cache lookup that still works when called from a worker thread."""
    try:
        return _history_cached(yf_symbol, period, interval)
    except Exception:
        return _history_fetch(yf_symbol, period, interval)


def _exchange_for_yf_symbol(yf_symbol: str) -> str:
    raw = (yf_symbol or "").upper()
    if raw.startswith("^"):
        for exch in EXCHANGES.values():
            if exch.index_symbol and exch.index_symbol.upper() == raw:
                return exch.key
        return "GLOBAL"
    if raw.endswith("-USD"):
        return "CRYPTO"
    if raw.endswith("=F"):
        return "CME"
    if raw.endswith("=X"):
        return "FX"
    for suffix, exch_key in ((".NS", "NSE"), (".BO", "BSE"), (".TO", "TSX"), (".V", "TSX"),
                             (".L", "LSE"), (".IL", "LSE"), (".DE", "XETRA"), (".PA", "XETRA"),
                             (".AS", "XETRA"), (".MI", "XETRA"), (".MC", "XETRA"), (".T", "TSE"),
                             (".HK", "HKEX"), (".AX", "ASX"), (".NZ", "ASX"), (".SI", "SGX"),
                             (".KS", "KRX"), (".KQ", "KRX"), (".SS", "SSE"), (".SZ", "SZSE")):
        if raw.endswith(suffix):
            return exch_key
    return "NASDAQ"


def _profile_fetch(yf_symbol: str) -> Dict[str, Any]:
    """Name / sector / industry / market cap, best effort and fully guarded."""
    out: Dict[str, Any] = {"name": "", "long_name": "", "sector": "", "industry": "", "market_cap": None,
                           "currency": "", "quote_type": "", "exchange_name": ""}
    try:
        ticker = yf.Ticker(yf_symbol)
        info: Dict[str, Any] = {}
        try:
            info = ticker.info or {}
        except Exception as exc:
            log_exception(f"info {yf_symbol}", exc)
        if isinstance(info, dict):
            out["name"] = info.get("shortName") or info.get("longName") or ""
            out["long_name"] = info.get("longName") or info.get("shortName") or ""
            out["sector"] = info.get("sector") or info.get("sectorDisp") or ""
            out["industry"] = info.get("industry") or ""
            out["currency"] = (info.get("currency") or "").upper()
            out["quote_type"] = info.get("quoteType") or ""
            out["exchange_name"] = info.get("fullExchangeName") or info.get("exchange") or ""
            cap = _num(info.get("marketCap"))
            out["market_cap"] = cap if cap and cap > 0 else None
        if out["market_cap"] is None:
            try:
                fast = ticker.fast_info
                cap = _num(fast.get("market_cap"))
                out["market_cap"] = cap if cap and cap > 0 else None
                if not out["currency"]:
                    out["currency"] = (str(fast.get("currency") or "")).upper()
            except Exception as exc:
                log_exception(f"fast_info {yf_symbol}", exc)
    except Exception as exc:
        log_exception(f"profile {yf_symbol}", exc)
    return out


_profile_cached = st.cache_data(ttl=3600, show_spinner=False)(_profile_fetch)


def _profile_safe(yf_symbol: str) -> Dict[str, Any]:
    """Cache lookup that still works when called from a worker thread."""
    try:
        return _profile_cached(yf_symbol)
    except Exception:
        return _profile_fetch(yf_symbol)


def enrich_profiles(rows: List[Dict[str, Any]], limit: int = 160, workers: int = 8) -> List[Dict[str, Any]]:
    """Attach name/sector/industry/market cap to at most `limit` rows, in parallel."""
    if not rows:
        return rows
    targets = rows[:limit]
    try:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda r: _profile_safe(r.get("yf_symbol") or ""), targets))
    except Exception as exc:
        log_exception("enrich profiles", exc)
        results = [{} for _ in targets]
    for row, profile in zip(targets, results):
        if not isinstance(profile, dict):
            continue
        provider_name = profile.get("name") or ""
        # A curated label (e.g. "Gold" for GC=F) is kept as the short display name; the
        # provider's longer legal name is preserved separately for the tooltip.
        row["long_name"] = provider_name or row.get("name") or row.get("symbol")
        if not (row.get("name_source") == "catalog" and row.get("name")):
            row["name"] = provider_name or row.get("name") or row.get("symbol")
        row["sector"] = profile.get("sector") or row.get("sector") or ""
        row["industry"] = profile.get("industry") or ""
        row["market_cap"] = profile.get("market_cap")
        if profile.get("currency"):
            row["currency"] = profile["currency"]
        if profile.get("exchange_name"):
            row["exchange_name"] = profile["exchange_name"]
        if profile.get("quote_type"):
            row["quote_type"] = profile["quote_type"]
    for row in rows[limit:]:
        row.setdefault("sector", "")
        row.setdefault("market_cap", None)
        row.setdefault("name", row.get("symbol"))
        row.setdefault("long_name", row.get("name"))
    return rows


# ===========================================================================
# SECTION 7 - NORMALIZED MARKET FRAME + DATA STATUS
# ===========================================================================
def _frame_columns() -> List[str]:
    return ["symbol", "yf_symbol", "name", "long_name", "sector", "industry", "exchange", "market",
            "currency", "price", "previous_close", "change", "change_percent", "market_cap", "volume",
            "day_high", "day_low", "open", "last_bar", "session_date", "data_status",
            "timestamp", "ok", "error", "source"]


def normalize_rows(rows: List[Dict[str, Any]], data_status: str, session_date: str) -> pd.DataFrame:
    """Turn provider rows into a validated, UI-safe DataFrame."""
    cleaned: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        record = {key: row.get(key) for key in _frame_columns()}
        record["price"] = valid_price(row.get("price"))
        record["previous_close"] = valid_price(row.get("previous_close"))
        record["change"] = _num(row.get("change"))
        record["change_percent"] = _num(row.get("change_percent"))
        record["volume"] = max(0.0, _num(row.get("volume")) or 0.0)
        cap = _num(row.get("market_cap"))
        record["market_cap"] = cap if cap and cap > 0 else None
        record["day_high"] = valid_price(row.get("day_high"))
        record["day_low"] = valid_price(row.get("day_low"))
        record["open"] = valid_price(row.get("open"))
        record["data_status"] = data_status
        record["session_date"] = session_date
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        record["ok"] = bool(record["price"]) and bool(row.get("ok", True))
        exch = EXCHANGES.get(str(row.get("exchange") or ""))
        record["market"] = exch.market if exch else "global"
        record["currency"] = (row.get("currency") or guess_currency(row.get("yf_symbol") or "", row.get("exchange") or "")).upper()
        record["name"] = row.get("name") or record["symbol"]
        record["long_name"] = row.get("long_name") or record["name"]
        if not record["sector"]:
            record["sector"] = MARKET_SECTOR_FALLBACK.get(record["market"], "")
        cleaned.append(record)
    if not cleaned:
        return pd.DataFrame(columns=_frame_columns())
    frame = pd.DataFrame(cleaned)
    frame = frame[frame["price"].notna()]
    return frame.reset_index(drop=True)


def infer_data_status(exchange_key: str, rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """LIVE / DELAYED / LAST SESSION / CACHED / UNAVAILABLE - never guessed upward."""
    status = session_status(exchange_key)
    exch = EXCHANGES.get(exchange_key)
    expected = expected_last_session_date(exchange_key)
    bar_dates = sorted({str(r.get("last_bar_date") or "") for r in rows if r.get("last_bar_date")})
    latest_bar = bar_dates[-1] if bar_dates else ""
    if not rows:
        return "UNAVAILABLE", expected
    if exch and exch.session_kind in ("continuous", "extended24x5") and latest_bar:
        return "LIVE", latest_bar
    if status["status"] in ("OPEN", "PRE-MARKET", "AFTER-HOURS"):
        if latest_bar == status["session_date"]:
            return "LIVE", latest_bar
        if latest_bar:
            return "DELAYED", latest_bar
        return "DELAYED", status["session_date"]
    if latest_bar:
        return "LAST SESSION", latest_bar
    return "LAST SESSION", expected


def status_pill(status: str) -> str:
    css = {
        "LIVE": "pill-open", "DELAYED": "pill-delayed", "LAST SESSION": "pill-last",
        "CACHED": "pill-cached", "UNAVAILABLE": "pill-unavailable",
    }.get(status, "pill-neutral")
    dot = {"LIVE": "●", "DELAYED": "◐", "LAST SESSION": "○", "CACHED": "◌", "UNAVAILABLE": "✕"}.get(status, "•")
    return f"<span class='pill {css}'>{dot} {status}</span>"


def session_pill(status: str) -> str:
    css = {"OPEN": "pill-open", "CLOSED": "pill-closed", "PRE-MARKET": "pill-pre",
           "AFTER-HOURS": "pill-pre", "HALTED": "pill-unavailable",
           "UNKNOWN": "pill-neutral"}.get(status, "pill-neutral")
    return f"<span class='pill {css}'>{status}</span>"


def data_status_explanation(status: str) -> str:
    return {
        "LIVE": "The provider returned a bar timestamped inside the current session, and the exchange is open.",
        "DELAYED": "The exchange is open but the newest bar the provider returned is older than the current session "
                   "(Yahoo publishes many non-US exchanges with a delay).",
        "LAST SESSION": "The exchange is closed. This is the most recent completed trading session from the provider.",
        "CACHED": "The provider could not be reached or returned nothing usable. Showing the last successful snapshot "
                  "saved by this app - not a live feed.",
        "UNAVAILABLE": "No data could be obtained for this universe right now.",
    }.get(status, "")


# ===========================================================================
# SECTION 8 - MARKET SERVICE (the single entry point used by UI + AI tools)
# ===========================================================================
def load_market_frame(exchange_key: str, universe_key: str,
                      enrich: bool = True, enrich_limit: int = 160,
                      force_refresh: bool = False) -> Dict[str, Any]:
    """
    Returns:
        {"frame", "data_status", "session_date", "session", "currency", "exchange",
         "universe", "universe_note", "symbol_source", "from_cache", "errors", "rows"}
    Fallback chain: live provider -> (stale) provider cache -> last saved snapshot -> empty.
    """
    exchange_key = exchange_key if exchange_key in EXCHANGES else "NSE"
    exch = EXCHANGES[exchange_key]
    universe = resolve_universe(exchange_key, universe_key)
    symbols = PROVIDER.get_symbols(exchange_key, universe.key)
    errors: List[str] = []
    if force_refresh:
        try:
            _quotes_cached.clear()
            _symbols_cached.clear()
        except Exception as exc:
            log_exception("cache clear", exc)

    quotes: List[Dict[str, Any]] = []
    if symbols:
        try:
            quotes = PROVIDER.get_quotes([s["yf_symbol"] for s in symbols])
        except Exception as exc:
            log_exception(f"quotes {exchange_key}/{universe.key}", exc)
            errors.append(safe_error(exc))

    meta = {s["yf_symbol"]: s for s in symbols}
    merged: List[Dict[str, Any]] = []
    for quote in quotes or []:
        if not isinstance(quote, dict):
            continue
        info = meta.get(quote.get("yf_symbol") or "", {})
        row = dict(quote)
        row.setdefault("name", info.get("name") or row.get("symbol"))
        row["exchange"] = exchange_key
        row["currency"] = (info.get("currency") or row.get("currency") or exch.currency)
        merged.append(row)

    ok_rows = [r for r in merged if valid_price(r.get("price"))]
    failed = [r for r in merged if not valid_price(r.get("price"))]
    if failed:
        errors.append(f"{len(failed)} symbol(s) returned no usable price and were skipped: "
                      + ", ".join(str(r.get("symbol")) for r in failed[:6]))
    if not ok_rows:
        snapshot = load_snapshot(exchange_key, universe.key)
        if snapshot and snapshot.get("rows"):
            frame = normalize_rows(snapshot["rows"], "CACHED", snapshot.get("session_date") or "")
            return {
                "frame": frame, "data_status": "CACHED",
                "session_date": snapshot.get("session_date") or "",
                "session": session_status(exchange_key), "currency": exch.currency,
                "exchange": exchange_key, "universe": universe.key, "universe_note": universe.note,
                "symbol_source": (symbols[0].get("source") if symbols else "none"),
                "from_cache": True, "errors": errors + ["Provider returned no usable rows."],
                "rows": snapshot["rows"],
            }
        return {
            "frame": normalize_rows([], "UNAVAILABLE", ""), "data_status": "UNAVAILABLE",
            "session_date": "", "session": session_status(exchange_key), "currency": exch.currency,
            "exchange": exchange_key, "universe": universe.key, "universe_note": universe.note,
            "symbol_source": (symbols[0].get("source") if symbols else "none"),
            "from_cache": False, "errors": errors, "rows": [],
        }

    if enrich:
        ok_rows = enrich_profiles(ok_rows, limit=enrich_limit)

    data_status, session_date = infer_data_status(exchange_key, ok_rows)
    frame = normalize_rows(ok_rows, data_status, session_date)
    save_snapshot(exchange_key, universe.key, ok_rows, data_status, session_date, exch.currency)
    return {
        "frame": frame, "data_status": data_status, "session_date": session_date,
        "session": session_status(exchange_key), "currency": exch.currency,
        "exchange": exchange_key, "universe": universe.key, "universe_note": universe.note,
        "symbol_source": (symbols[0].get("source") if symbols else "none"),
        "from_cache": False, "errors": errors, "rows": ok_rows,
    }


# ---- derived analytics ----------------------------------------------------
def market_breadth(frame: pd.DataFrame) -> Dict[str, Any]:
    """Advances / declines / unchanged / A-D ratio, computed only from real rows."""
    out: Dict[str, Any] = {"advances": 0, "declines": 0, "unchanged": 0, "total": 0,
                           "ad_ratio": None, "avg_change": None, "median_change": None,
                           "high_52w": None, "low_52w": None, "note": ""}
    if frame is None or frame.empty:
        out["note"] = "No market data available."
        return out
    work = frame.copy()
    work["change_percent"] = pd.to_numeric(work["change_percent"], errors="coerce")
    work = work[work["change_percent"].notna()]
    if work.empty:
        out["note"] = "Percentage change unavailable for the loaded universe."
        return out
    out["advances"] = int((work["change_percent"] > 0.05).sum())
    out["declines"] = int((work["change_percent"] < -0.05).sum())
    out["unchanged"] = int(len(work) - out["advances"] - out["declines"])
    out["total"] = int(len(work))
    out["ad_ratio"] = round(out["advances"] / out["declines"], 2) if out["declines"] else None
    out["avg_change"] = float(work["change_percent"].mean())
    out["median_change"] = float(work["change_percent"].median())
    out["note"] = ("52-week high/low counts require per-symbol yearly history and are not computed for the "
                   "whole universe on every refresh.")
    return out


def sector_table(frame: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["sector", "count", "avg_change", "market_cap", "advances", "declines"])
    work = frame.copy()
    work["sector"] = work["sector"].fillna("").replace("", "Unclassified")
    work["change_percent"] = pd.to_numeric(work["change_percent"], errors="coerce")
    work["market_cap"] = pd.to_numeric(work["market_cap"], errors="coerce")
    grouped = work.groupby("sector", dropna=False)
    table = pd.DataFrame({
        "count": grouped.size(),
        "avg_change": grouped["change_percent"].mean().round(2),
        "market_cap": grouped["market_cap"].sum(min_count=1),
        "advances": grouped["change_percent"].apply(lambda s: int((s > 0.05).sum())),
        "declines": grouped["change_percent"].apply(lambda s: int((s < -0.05).sum())),
    }).reset_index()
    return table.sort_values("market_cap", ascending=False, na_position="last").head(top)


def breadth_52w(frame: pd.DataFrame, limit: int = 60) -> Dict[str, Any]:
    """52-week high/low counts for a bounded slice of the universe (honest, capped)."""
    result = {"highs": None, "lows": None, "checked": 0, "note": ""}
    if frame is None or frame.empty:
        result["note"] = "No data."
        return result
    symbols = [s for s in frame["yf_symbol"].tolist() if s][:limit]
    highs = lows = 0
    def probe(symbol: str):
        year = _history_safe(symbol, "1y", "1d")
        if year is None or year.empty:
            return None
        try:
            last = float(pd.to_numeric(year["Close"], errors="coerce").dropna().iloc[-1])
            hi = float(pd.to_numeric(year["High"], errors="coerce").max())
            lo = float(pd.to_numeric(year["Low"], errors="coerce").min())
        except Exception:
            return None
        if not math.isfinite(last):
            return None
        return (last >= hi * 0.995, last <= lo * 1.005)
    try:
        with futures.ThreadPoolExecutor(max_workers=6) as pool:
            for outcome in pool.map(probe, symbols):
                if not outcome:
                    continue
                result["checked"] += 1
                if outcome[0]:
                    highs += 1
                if outcome[1]:
                    lows += 1
    except Exception as exc:
        log_exception("52w breadth", exc)
        result["note"] = safe_error(exc)
        return result
    result["highs"], result["lows"] = highs, lows
    result["note"] = f"Checked {result['checked']} of {len(frame)} symbols (capped for responsiveness)."
    return result


def build_heatmap_figure(frame: pd.DataFrame, size_mode: str = "Market cap",
                         color_mode: str = "% change", group_mode: str = "Sectors",
                         currency: str = "USD", max_tiles: int = 250):
    """UPGRADED (v3.1): delegates to the standalone `heatmap.py` module — tiles
    coloured strictly by daily % change, labelled with ticker + FULL stock name."""
    import heatmap as _heatmap_module
    return _heatmap_module.build_heatmap_figure(frame, size_mode=size_mode,
                                                color_mode=color_mode,
                                                group_mode=group_mode,
                                                currency=currency,
                                                max_tiles=max_tiles)


# ===========================================================================
# SECTION 9 - NEWS SERVICE (crash-proof parser)
# ---------------------------------------------------------------------------
# ROOT CAUSE OF THE ORIGINAL AttributeError (traceback line 945):
#     link = n.get("link") or (n.get("content") or {}).get("clickThroughUrl", {}).get("url") or "#"
# Yahoo returns BOTH shapes and sometimes junk:
#     {"title", "publisher", "link", "providerPublishTime"}                      (legacy)
#     {"id", "content": {"title", "provider": {"displayName"},
#                        "clickThroughUrl": {"url"}, ...}}                       (current)
# and `content` can be a str / list / None / int. Calling .get() on a str raised
# AttributeError and killed the whole script.
# Every field now goes through a typed accessor - one malformed article can
# never take down the app, and missing pieces degrade to "—"/"#".
# ===========================================================================
def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        for entry in value:
            text = _as_str(entry)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("text", "title", "name", "url", "value"):
            text = _as_str(value.get(key))
            if text:
                return text
    return ""


def _first_url(*candidates: Any) -> str:
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("url", "href", "link"):
                url = _as_str(candidate.get(key))
                if url.startswith(("http://", "https://")):
                    return url
        elif isinstance(candidate, (list, tuple)):
            nested = _first_url(*candidate)
            if nested:
                return nested
        else:
            url = _as_str(candidate)
            if url.startswith(("http://", "https://")):
                return url
    return ""


def safe_news_link(item: Any) -> str:
    """The robust replacement for the line that crashed. Never raises."""
    if not isinstance(item, dict):
        return ""
    direct = _as_str(item.get("link"))
    if direct.startswith(("http://", "https://")):
        return direct
    content = item.get("content")
    if isinstance(content, dict):
        for key in ("clickThroughUrl", "canonicalUrl", "previewUrl", "providerUrl"):
            url = _first_url(content.get(key))
            if url:
                return url
        url = _first_url(content.get("url"))
        if url:
            return url
    for key in ("clickThroughUrl", "canonicalUrl", "url"):
        url = _first_url(item.get(key))
        if url:
            return url
    return ""


def safe_news_image(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    content = _as_dict(item.get("content"))
    thumb = _as_dict(content.get("thumbnail"))
    resolutions = thumb.get("resolutions")
    if isinstance(resolutions, list):
        for entry in resolutions:
            url = _first_url(entry)
            if url:
                return url
    for key in ("thumbnail", "image", "img"):
        url = _first_url(item.get(key))
        if url:
            return url
    return ""


def parse_news_items(raw: Any, symbol: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """Normalize ANY news payload into a list of safe dicts. Never raises."""
    if isinstance(raw, dict):
        candidates: Any = raw.get("news") or raw.get("items") or raw.get("data") or []
    else:
        candidates = raw
    if not isinstance(candidates, (list, tuple)):
        return []
    items: List[Dict[str, Any]] = []
    for entry in candidates:
        try:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content") if isinstance(entry.get("content"), dict) else {}
            title = (_as_str(entry.get("title")) or _as_str(content.get("title"))
                     or _as_str(content.get("headline")) or _as_str(entry.get("headline")))
            summary = (_as_str(entry.get("summary")) or _as_str(entry.get("description"))
                       or _as_str(content.get("summary")) or _as_str(content.get("description")))
            publisher = (_as_str(entry.get("publisher")) or _as_str(entry.get("provider"))
                         or _as_str(_as_dict(content.get("provider")).get("displayName"))
                         or _as_str(_as_dict(content.get("provider")).get("name"))
                         or _as_str(entry.get("source")) or "Unknown source")
            link = safe_news_link(entry)
            image = safe_news_image(entry)
            stamp = (entry.get("providerPublishTime") or entry.get("published")
                     or content.get("pubDate") or content.get("datePublished")
                     or content.get("displayTime")
                     or entry.get("pubDate") or entry.get("time"))
            published = None
            if isinstance(stamp, datetime):
                # v3.1.1 FIX: already a normalized datetime (fetch_news re-parses
                # its own parsed items) - the old int/float/str-only chain dropped
                # it and every headline showed "time unknown".
                published = stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
            elif isinstance(stamp, (int, float)):
                try:
                    published = datetime.fromtimestamp(float(stamp), tz=timezone.utc)
                except Exception:
                    published = None
            elif isinstance(stamp, str) and stamp.strip():
                text = stamp.strip().replace("Z", "+00:00")
                for parser in (lambda t: datetime.fromisoformat(t),
                               lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z"),
                               lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S")):
                    try:
                        published = parser(text)
                        break
                    except Exception:
                        continue
            if published is not None and published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if not title:
                continue
            items.append({
                "title": title,
                "summary": summary,
                "publisher": publisher,
                "link": link,
                "image": image,
                "published": published,
                "age_label": _relative_age(published),
                "symbol": symbol,
            })
        except Exception as exc:      # a single bad article is logged, never fatal
            log_exception("parse news item", exc)
            continue
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for item in items:
        key = item["title"].lower()[:110]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    unique.sort(key=lambda i: i["published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return unique[:limit]


def _relative_age(published: Optional[datetime]) -> str:
    if not isinstance(published, datetime):
        return "time unknown"
    now = datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    seconds = (now - published).total_seconds()
    if seconds < 0:
        return "just now"
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 7 * 86400:
        return f"{int(seconds // 86400)}d ago"
    return published.strftime("%d %b %Y")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(query: str, yf_symbol: str = "", limit: int = 20) -> Dict[str, Any]:
    """
    Real news only. Provider order:
      1. yfinance ticker.news  (per-instrument)
      2. Yahoo Finance search endpoint with newsCount (per query / per symbol)
    If both are empty the UI says so - no fabricated headlines, ever.
    """
    items: List[Dict[str, Any]] = []
    sources: List[str] = []
    errors: List[str] = []

    if yf_symbol:
        try:
            raw = yf.Ticker(yf_symbol).news
            parsed = parse_news_items(raw, symbol=yf_symbol, limit=limit)
            if parsed:
                items.extend(parsed)
                sources.append("yfinance ticker.news")
        except Exception as exc:
            log_exception(f"ticker.news {yf_symbol}", exc)
            errors.append(safe_error(exc))

    for search_term in [t for t in (query, yf_symbol) if t]:
        if len(items) >= limit:
            break
        try:
            response = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": search_term, "quotesCount": 0, "newsCount": min(limit, 20),
                        "enableFuzzyQuery": "false", "newsQueryId": "news_cie_vespa"},
                headers={"User-Agent": HTTP_UA, "Accept": "application/json"}, timeout=10)
            response.raise_for_status()
            payload = response.json() if response.content else {}
            parsed = parse_news_items(payload, symbol=yf_symbol or search_term, limit=limit)
            if parsed:
                items.extend(parsed)
                sources.append(f"Yahoo news search ('{search_term}')")
        except Exception as exc:
            log_exception(f"news search {search_term}", exc)
            errors.append(safe_error(exc))

    combined = parse_news_items(items, symbol=yf_symbol, limit=limit)
    return {"items": combined, "sources": sources, "errors": errors,
            "ok": bool(combined), "fetched_at": datetime.now(timezone.utc).isoformat()}


# Two lexicons: general business words plus market-specific event words
# (results, guidance, fundraise, order book ...) so a headline like
# "Q2 results beat estimates" is not scored Neutral just because it lacks the word "surge".
GENERAL_BULLISH = ("surge", "rally", "beat", "beats", "growth", "record", "profit", "gain", "gains",
                   "strong", "expansion", "bullish", "rise", "rises", "high", "upbeat", "upgrade",
                   "outperform", "positive", "approval", "wins", "jump", "soar", "boost", "buyback",
                   "dividend", "partnership", "recovery", "rebound", "optimism", "inflow", "buy")
GENERAL_BEARISH = ("fall", "falls", "drop", "drops", "loss", "losses", "miss", "misses", "cut", "cuts",
                   "weak", "down", "decline", "crash", "fear", "bearish", "selloff", "sell-off",
                   "concern", "concerns", "downgrade", "probe", "fine", "penalty", "lawsuit", "fraud",
                   "default", "resign", "layoff", "layoffs", "slump", "plunge", "halt", "recall",
                   "sell", "outflow", "pessimism")
MARKET_BULLISH = ("results beat", "beats estimates", "profit rises", "profit jumps", "revenue up",
                  "margin expansion", "order win", "order book", "new order", "wins contract", "bags order",
                  "fundraise", "fund raising", "capital raise", "qip", "ipo", "stake sale", "buyback",
                  "bonus issue", "stock split", "target raised", "price target raised", "upgrade",
                  "initiates coverage", "accumulate", "add rating", "all-time high", "52-week high",
                  "capex", "expansion plan", "capacity addition", "approval received", "regulatory approval",
                  "tie-up", "tie up", "joint venture", "acquisition", "to acquire", "demerger", "inflow")
MARKET_BEARISH = ("results miss", "misses estimates", "profit falls", "profit drops", "revenue down",
                  "margin pressure", "guidance cut", "cuts guidance", "target cut", "price target cut",
                  "downgrade", "reduce rating", "block deal", "bulk deal", "promoter sells", "pledge",
                  "insider selling", "auditor resigns", "delisting", "default", "insolvency", "bankruptcy",
                  "regulatory action", "show-cause", "sebi probe", "tax raid", "penalty", "order cancelled",
                  "contract terminated", "recall", "outage", "strike", "halt", "trading halt",
                  "data breach", "impairment", "write-off", "writedown", "outflow")


def _lexicon_hits(text: str, words: Iterable[str]) -> List[str]:
    return [word for word in words if word in text]


def news_sentiment(title: str, extra_text: str = "") -> Dict[str, Any]:
    """
    Keyword sentiment over a general + market-event lexicon. Deliberately transparent:
    it is labelled a heuristic in the UI, and the matched words are returned so the
    score can be audited. The news page's "Summary & impact" action adds a model read.
    """
    text = f"{title or ''} {extra_text or ''}".lower()
    bullish = _lexicon_hits(text, GENERAL_BULLISH) + _lexicon_hits(text, MARKET_BULLISH)
    bearish = _lexicon_hits(text, GENERAL_BEARISH) + _lexicon_hits(text, MARKET_BEARISH)
    if not text.strip():
        return {"score": None, "label": "No headline text", "colour": "#94a3b8",
                "bullish_terms": [], "bearish_terms": []}
    score = 5.5 + 0.55 * len(bullish) - 0.55 * len(bearish)
    score = max(1.5, min(9.5, round(score, 1)))
    if score >= 6.5:
        label, colour = "Bullish", "#34d399"
    elif score <= 4.5:
        label, colour = "Bearish", "#f87171"
    else:
        label, colour = "Neutral", "#fbbf24"
    return {"score": score, "label": label, "colour": colour,
            "bullish_terms": bullish[:6], "bearish_terms": bearish[:6]}


# ===========================================================================
# SECTION 10 - TECHNICAL ANALYSIS (Gann / Square-of-9, pivots, indicators)
# ---------------------------------------------------------------------------
# All of the original trading logic is preserved (classic pivots, the corrected
# Square-of-9 degree ladder anchored on the first 15-minute close) and extended
# with ATR / VWAP / RSI / EMA / relative-volume / swing structure - every value
# computed from real bars, "Insufficient data" when the bars are not there.
# ===========================================================================
GANN_DEGREES: Tuple[float, ...] = (22.5, 45.0, 67.5, 90.0, 180.0)


def generate_levels(high: float, low: float, close: float) -> Dict[str, float]:
    """Legacy classic pivot levels (kept as the fallback when intraday bars are missing)."""
    high, low, close = _num(high), _num(low), _num(close)
    if not high or not low or not close or high <= 0 or low <= 0 or close <= 0:
        return {}
    pivot = (high + low + close) / 3.0
    span = high - low
    return {
        "pivot": pivot,
        "R1": 2 * pivot - low, "R2": pivot + span, "R3": high + 2 * (pivot - low),
        "R4": pivot + 2 * span, "R5": pivot + 3 * span,
        "S1": 2 * pivot - high, "S2": pivot - span, "S3": low - 2 * (high - pivot),
        "S4": pivot - 2 * span, "S5": pivot - 3 * span,
    }


def gann_degree_levels(ref_price: float) -> Dict[str, Any]:
    """
    Square-of-9 Gann degree ladder (unchanged formula, kept exactly as it was):
        0° anchor  = reference price
        Resistance = (sqrt(anchor) + degree / 180) ^ 2
        Support    = (sqrt(anchor) - degree / 180) ^ 2
    """
    reference = valid_price(ref_price)
    if reference is None:
        return {}
    anchor_sqrt = math.sqrt(reference)
    levels: Dict[str, Any] = {"_ref_close": reference, "_type": "gann_sq9", "0": reference}
    for degree in GANN_DEGREES:
        suffix = f"{degree:g}"
        levels[f"R{suffix}"] = (anchor_sqrt + degree / 180.0) ** 2
        levels[f"S{suffix}"] = max(0.01, (anchor_sqrt - degree / 180.0) ** 2)
    return levels


def gann_degree_table(levels: Dict[str, Any], price: Optional[float],
                      currency: str = "USD") -> List[Dict[str, Any]]:
    """Ordered ladder (R180 → R22.5, anchor, S22.5 → S180) ready for rendering."""
    reference = valid_price(levels.get("_ref_close")) if levels else None
    if reference is None:
        return []
    rows: List[Dict[str, Any]] = []
    for degree in (180.0, 90.0, 67.5, 45.0, 22.5):
        value = _num(levels.get(f"R{degree:g}"))
        rows.append({"kind": "resistance", "degree": degree, "label": f"R{degree:g}",
                     "value": value, "pct": ((value - price) / price * 100) if (value and price) else None,
                     "text": fmt_price(value, currency)})
    rows.append({"kind": "anchor", "degree": 0.0, "label": "0° anchor", "value": reference,
                 "pct": ((reference - price) / price * 100) if price else None,
                 "text": fmt_price(reference, currency)})
    for degree in (22.5, 45.0, 67.5, 90.0, 180.0):
        value = _num(levels.get(f"S{degree:g}"))
        rows.append({"kind": "support", "degree": degree, "label": f"S{degree:g}",
                     "value": value, "pct": ((value - price) / price * 100) if (value and price) else None,
                     "text": fmt_price(value, currency)})
    return rows


def get_session_fixed_levels(yf_symbol: str) -> Dict[str, Any]:
    """0° reference = first 15-minute CLOSING price of the session (fixed for the session)."""
    try:
        intraday = _history_cached(yf_symbol, "5d", "15m")
        if intraday is not None and not intraday.empty:
            frame = intraday.copy()
            frame["_date"] = [ts.date() if hasattr(ts, "date") else None for ts in frame.index]
            last_day = frame["_date"].iloc[-1]
            day_bars = frame[frame["_date"] == last_day] if last_day is not None else frame
            if day_bars.empty:
                day_bars = frame.tail(26)
            first = day_bars.iloc[0]
            reference = valid_price(first.get("Close"))
            if reference:
                levels = gann_degree_levels(reference)
                levels["_ref"] = "first_15m"
                levels["_ref_time"] = str(day_bars.index[0])[:19]
                levels["_ref_high"] = _num(first.get("High"))
                levels["_ref_low"] = _num(first.get("Low"))
                levels["_session_date"] = str(last_day) if last_day else ""
                levels.update({k: v for k, v in generate_levels(levels["_ref_high"], levels["_ref_low"],
                                                                reference).items()})
                return levels
        daily = _history_cached(yf_symbol, "10d", "1d")
        if daily is not None and not daily.empty:
            last = daily.iloc[-1]
            reference = valid_price(last.get("Close"))
            if reference:
                levels = gann_degree_levels(reference)
                levels["_ref"] = "daily_fallback"
                levels["_ref_time"] = str(daily.index[-1])[:19]
                levels["_session_date"] = str(pd.Timestamp(daily.index[-1]).date())
                levels.update({k: v for k, v in generate_levels(last.get("High"), last.get("Low"),
                                                                reference).items()})
                return levels
    except Exception as exc:
        log_exception(f"session levels {yf_symbol}", exc)
    return {}


def _true_range(frame: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    prev_close = close.shift(1)
    return pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def _rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    series = pd.to_numeric(closes, errors="coerce").dropna()
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    last_gain, last_loss = gain.iloc[-1], loss.iloc[-1]
    if not math.isfinite(last_gain) or not math.isfinite(last_loss):
        return None
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def technical_indicators(yf_symbol: str) -> Dict[str, Any]:
    """
    ATR(14), VWAP (last session, from 15m bars), EMA(9/50), RSI(14), swing high/low,
    trend, relative volume and liquidity - all from real bars, or an explicit
    "Insufficient data" flag.
    """
    out: Dict[str, Any] = {
        "symbol": yf_symbol, "atr": None, "atr_percent": None, "vwap": None, "ema9": None,
        "ema50": None, "ema_bias": "", "rsi": None, "rsi_label": "", "swing_high": None,
        "swing_low": None, "trend": "", "relative_volume": None, "avg_volume": None,
        "liquidity": None, "candle_pattern": "", "sufficient": False, "note": "",
    }
    try:
        daily = _history_cached(yf_symbol, "6mo", "1d")
        if daily is None or daily.empty or len(daily) < 5:
            out["note"] = "Insufficient data (fewer than 5 daily bars available)."
            return out
        close = pd.to_numeric(daily["Close"], errors="coerce").dropna()
        if close.empty:
            out["note"] = "Insufficient data (no usable close prices)."
            return out
        out["sufficient"] = True
        if len(daily) >= 15:
            atr_series = _true_range(daily).rolling(14).mean()
            atr = _num(atr_series.iloc[-1])
            out["atr"] = atr
            out["atr_percent"] = (atr / close.iloc[-1] * 100) if (atr and close.iloc[-1]) else None
        out["ema9"] = _num(close.ewm(span=9, adjust=False).mean().iloc[-1])
        if len(close) >= 20:
            out["ema50"] = _num(close.ewm(span=50, adjust=False).mean().iloc[-1])
        if out["ema9"] and out["ema50"]:
            out["ema_bias"] = ("Bullish - EMA9 above EMA50" if out["ema9"] > out["ema50"]
                               else "Bearish - EMA9 below EMA50")
        out["rsi"] = _rsi(close)
        if out["rsi"] is not None:
            out["rsi_label"] = ("Overbought" if out["rsi"] >= 70 else
                                "Oversold" if out["rsi"] <= 30 else "Neutral")
        window = daily.tail(60)
        out["swing_high"] = _num(pd.to_numeric(window["High"], errors="coerce").max())
        out["swing_low"] = _num(pd.to_numeric(window["Low"], errors="coerce").min())
        last_close = float(close.iloc[-1])
        if out["ema9"] and out["ema50"] and out["swing_high"] and out["swing_low"]:
            if last_close > out["ema9"] > out["ema50"]:
                out["trend"] = "Uptrend"
            elif last_close < out["ema9"] < out["ema50"]:
                out["trend"] = "Downtrend"
            else:
                out["trend"] = "Sideways / mixed"
        volumes = pd.to_numeric(daily["Volume"], errors="coerce").dropna()
        if len(volumes) >= 21:
            avg_volume = float(volumes.tail(20).mean())
            out["avg_volume"] = avg_volume
            out["relative_volume"] = float(volumes.iloc[-1] / avg_volume) if avg_volume else None
            out["liquidity"] = avg_volume * last_close
        out["candle_pattern"] = detect_candle_pattern(daily.tail(5))
        intraday = _history_cached(yf_symbol, "5d", "15m")
        if intraday is not None and not intraday.empty:
            frame = intraday.copy()
            frame["_date"] = [ts.date() if hasattr(ts, "date") else None for ts in frame.index]
            last_day = frame["_date"].iloc[-1]
            day_bars = frame[frame["_date"] == last_day]
            if day_bars.empty:
                day_bars = frame
            price = pd.to_numeric(day_bars["Close"], errors="coerce")
            volume = pd.to_numeric(day_bars["Volume"], errors="coerce").fillna(0)
            total_volume = float(volume.sum())
            if total_volume > 0 and price.notna().any():
                typical = (pd.to_numeric(day_bars["High"], errors="coerce")
                           + pd.to_numeric(day_bars["Low"], errors="coerce")
                           + price) / 3.0
                out["vwap"] = float((typical * volume).sum() / total_volume)
    except Exception as exc:
        log_exception(f"indicators {yf_symbol}", exc)
        out["note"] = safe_error(exc)
    return out


def detect_candle_pattern(frame: pd.DataFrame) -> str:
    if frame is None or len(frame) < 2:
        return "Insufficient data"
    try:
        close = pd.to_numeric(frame["Close"], errors="coerce")
        open_ = pd.to_numeric(frame["Open"], errors="coerce")
        if close.isna().iloc[-1] or open_.isna().iloc[-1]:
            return "Insufficient data"
        if close.iloc[-1] > open_.iloc[-1] and close.iloc[-2] < open_.iloc[-2] and close.iloc[-1] >= open_.iloc[-2]:
            return "Bullish engulfing"
        if close.iloc[-1] < open_.iloc[-1] and close.iloc[-2] > open_.iloc[-2] and close.iloc[-1] <= open_.iloc[-2]:
            return "Bearish engulfing"
        body = abs(close.iloc[-1] - open_.iloc[-1])
        span = abs(close.iloc[-1] - open_.iloc[-1]) + 1e-9
        if body / span > 0.9:
            return "Strong body (directional)"
        return "No clear pattern"
    except Exception as exc:
        log_exception("candle pattern", exc)
        return "Insufficient data"


def generate_suggestions(price: Optional[float], levels: Dict[str, Any],
                         indicators: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Both directions from the ACTUAL level set (no invented signals).
    If the data needed for a plan is not there, nothing is produced.
    """
    price = valid_price(price)
    if not levels or price is None:
        return []
    atr = _num((indicators or {}).get("atr"))
    support_candidates = sorted([(v, k) for k, v in levels.items()
                                 if str(k).startswith("S") and not str(k).startswith("_") and valid_price(v) and v < price],
                                reverse=True)
    resistance_candidates = sorted([(v, k) for k, v in levels.items()
                                    if str(k).startswith("R") and not str(k).startswith("_") and valid_price(v) and v > price])
    plans: List[Dict[str, Any]] = []

    if len(support_candidates) >= 2 and resistance_candidates:
        entry, entry_label = support_candidates[0]
        stop, stop_label = support_candidates[1]
        target1, target1_label = resistance_candidates[0]
        target2 = resistance_candidates[1][0] if len(resistance_candidates) > 1 else target1
        risk = abs(entry - stop) or (atr or 0)
        if risk > 0:
            plans.append({
                "direction": "bullish", "basis": f"{entry_label} entry / {stop_label} stop / {target1_label} target",
                "entry": entry, "stop": stop, "t1": target1, "t2": target2,
                "rr1": round(abs(target1 - entry) / risk, 2), "rr2": round(abs(target2 - entry) / risk, 2),
            })
    if len(resistance_candidates) >= 2 and support_candidates:
        entry, entry_label = resistance_candidates[0]
        stop, stop_label = resistance_candidates[1]
        target1, target1_label = support_candidates[0]
        target2 = support_candidates[1][0] if len(support_candidates) > 1 else target1
        risk = abs(stop - entry) or (atr or 0)
        if risk > 0:
            plans.append({
                "direction": "bearish", "basis": f"{entry_label} entry / {stop_label} stop / {target1_label} target",
                "entry": entry, "stop": stop, "t1": target1, "t2": target2,
                "rr1": round(abs(entry - target1) / risk, 2), "rr2": round(abs(entry - target2) / risk, 2),
            })
    return plans


def symbol_snapshot(yf_symbol: str) -> Dict[str, Any]:
    """One-call snapshot for a single instrument: quote + profile + currency."""
    quotes = PROVIDER.get_quotes([yf_symbol]) or []
    quote = quotes[0] if quotes else {}
    profile = _profile_cached(yf_symbol)
    exchange_key = str(quote.get("exchange") or _exchange_for_yf_symbol(yf_symbol))
    currency = (profile.get("currency") or quote.get("currency")
                or guess_currency(yf_symbol, exchange_key)).upper()
    price = valid_price(quote.get("price"))
    market_key = EXCHANGES[exchange_key].market if exchange_key in EXCHANGES else ""
    sector = profile.get("sector") or MARKET_SECTOR_FALLBACK.get(market_key, "")
    return {
        "yf_symbol": yf_symbol,
        "symbol": yf_symbol,
        "name": profile.get("name") or catalog_meta(yf_symbol).get("name") or yf_symbol,
        "long_name": profile.get("long_name") or profile.get("name") or yf_symbol,
        "exchange": exchange_key,
        "market": EXCHANGES[exchange_key].market if exchange_key in EXCHANGES else "global",
        "currency": currency,
        "price": price,
        "previous_close": valid_price(quote.get("previous_close")),
        "change": _num(quote.get("change")),
        "change_percent": _num(quote.get("change_percent")),
        "day_high": valid_price(quote.get("day_high")),
        "day_low": valid_price(quote.get("day_low")),
        "open": valid_price(quote.get("open")),
        "volume": _num(quote.get("volume")),
        "market_cap": profile.get("market_cap"),
        "sector": sector,
        "industry": profile.get("industry") or "",
        "last_bar": quote.get("last_bar") or "",
        "last_bar_date": quote.get("last_bar_date") or "",
        "ok": price is not None,
        "error": quote.get("error") or "",
    }


def resolve_yf_symbol(user_input: str, market_key: str = "india", exchange_key: str = "") -> str:
    """Map free text to a provider symbol, honouring the selected market."""
    raw = (user_input or "").strip().upper()
    if not raw:
        return ""
    known = MARKET_ITEMS_BY_NAME.get(raw)
    if known:
        return known["yf"]
    if any(token in raw for token in (".NS", ".BO", "-USD", "=F", "=X", "^")):
        return raw
    if exchange_key and exchange_key in EXCHANGES:
        suffix = EXCHANGES[exchange_key].suffix
        if suffix and not raw.endswith(suffix):
            return f"{raw}{suffix}"
    market = MARKETS_BY_KEY.get(market_key)
    if market and market.exchanges:
        suffix = EXCHANGES[market.exchanges[0]].suffix
        if suffix:
            return f"{raw}{suffix}"
    return raw


@st.cache_data(ttl=1800, show_spinner=False)
def yahoo_symbol_search(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Universal symbol search - any listed instrument worldwide."""
    try:
        response = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                                params={"q": query, "quotesCount": limit, "newsCount": 0},
                                headers={"User-Agent": HTTP_UA, "Accept": "application/json"}, timeout=8)
        response.raise_for_status()
        payload = response.json() or {}
        out: List[Dict[str, Any]] = []
        for quote in payload.get("quotes", []) or []:
            if not isinstance(quote, dict) or not quote.get("symbol"):
                continue
            symbol = str(quote["symbol"])
            out.append({
                "symbol": symbol,
                "name": quote.get("shortname") or quote.get("longname") or symbol,
                "exchange": quote.get("exchDisp") or quote.get("exchange") or "",
                "type": quote.get("quoteType") or "",
                "currency": guess_currency(symbol, _exchange_for_yf_symbol(symbol)),
            })
        return out
    except Exception as exc:
        log_exception(f"symbol search {query}", exc)
        return []


# ===========================================================================
# SECTION 11 - AI TOOL LAYER
# ---------------------------------------------------------------------------
# The assistant retrieves data on demand instead of being handed the whole
# application state. Every tool validates its input, tolerates a dead provider
# and returns a structured result - a failing tool can never break the chat.
# ===========================================================================
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Current quote, currency, session and data status for ONE instrument "
                           "(e.g. RELIANCE.NS, AAPL, BTC-USD, ^NSEI).",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "Provider symbol, e.g. RELIANCE.NS"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": "ATR(14), VWAP, EMA(9/50), RSI(14), swing high/low, trend, relative volume and "
                           "liquidity for one instrument.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gann_levels",
            "description": "Square-of-9 Gann degree ladder plus classic pivot support/resistance for one "
                           "instrument (0 degree anchor = first 15-minute close of the session).",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_status",
            "description": "Session status (OPEN / PRE-MARKET / AFTER-HOURS / CLOSED) with the exchange-local "
                           "time for one exchange, plus the currency of that exchange.",
            "parameters": {
                "type": "object",
                "properties": {"exchange": {"type": "string",
                                            "description": "Exchange key, e.g. NSE, NASDAQ, LSE, TSE, HKEX"}},
                "required": ["exchange"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_heatmap_data",
            "description": "Normalized heat-map rows (price, change %, market cap, sector, data status) for an "
                           "exchange + universe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "exchange": {"type": "string", "description": "e.g. NSE, NASDAQ"},
                    "universe": {"type": "string",
                                 "description": "Universe key, e.g. NSE_TOP50, NSE_INDEX, NSE_CURATED. "
                                                "Omit to use the app's current selection."},
                    "limit": {"type": "integer", "description": "Max rows to return (default 25)"},
                },
                "required": ["exchange"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_data",
            "description": "Sector aggregation (count, average change, total market cap, advances/declines) for an "
                           "exchange + universe.",
            "parameters": {
                "type": "object",
                "properties": {"exchange": {"type": "string"}, "universe": {"type": "string"}},
                "required": ["exchange"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Real news headlines from the configured provider for a symbol or free-text topic. "
                           "Returns an empty list when nothing is available - never invented headlines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search topic, e.g. 'Reliance Industries'"},
                    "symbol": {"type": "string", "description": "Optional provider symbol, e.g. RELIANCE.NS"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]


def _tool_error(message: str) -> Dict[str, Any]:
    return {"ok": False, "error": message}


def tool_get_stock_data(symbol: str = "", **_: Any) -> Dict[str, Any]:
    try:
        if not symbol or not str(symbol).strip():
            return _tool_error("A symbol is required, e.g. RELIANCE.NS or AAPL.")
        snapshot = symbol_snapshot(str(symbol).strip())
        snapshot["data_status"] = infer_data_status(snapshot["exchange"], [{"last_bar_date": snapshot["last_bar_date"]}])[0]
        return {"ok": bool(snapshot.get("ok")), "data": snapshot}
    except Exception as exc:
        log_exception("tool get_stock_data", exc)
        return _tool_error(safe_error(exc))


def tool_get_technical_indicators(symbol: str = "", **_: Any) -> Dict[str, Any]:
    try:
        if not symbol:
            return _tool_error("A symbol is required.")
        return {"ok": True, "data": technical_indicators(str(symbol).strip())}
    except Exception as exc:
        log_exception("tool indicators", exc)
        return _tool_error(safe_error(exc))


def tool_get_gann_levels(symbol: str = "", **_: Any) -> Dict[str, Any]:
    try:
        if not symbol:
            return _tool_error("A symbol is required.")
        symbol = str(symbol).strip()
        levels = get_session_fixed_levels(symbol)
        if not levels:
            return {"ok": False, "error": "No level data available for this instrument right now."}
        quote = symbol_snapshot(symbol)
        currency = quote.get("currency") or guess_currency(symbol)
        table = gann_degree_table(levels, quote.get("price"), currency)
        return {"ok": True, "data": {
            "symbol": symbol, "currency": currency, "anchor": levels.get("_ref_close"),
            "anchor_source": levels.get("_ref"), "anchor_time": levels.get("_ref_time"),
            "ladder": [{"label": r["label"], "value": r["value"], "pct_from_price": r["pct"]} for r in table],
            "classic_pivots": {k: v for k, v in levels.items()
                               if k.startswith(("R", "S", "pivot")) and not k.startswith(("R_", "S_"))},
        }}
    except Exception as exc:
        log_exception("tool gann", exc)
        return _tool_error(safe_error(exc))


def tool_get_market_status(exchange: str = "", **_: Any) -> Dict[str, Any]:
    try:
        key = str(exchange or "").strip().upper()
        if key not in EXCHANGES:
            return _tool_error(f"Unknown exchange '{exchange}'. Known: {', '.join(sorted(EXCHANGES))}")
        return {"ok": True, "data": session_status(key)}
    except Exception as exc:
        log_exception("tool market status", exc)
        return _tool_error(safe_error(exc))


def tool_get_heatmap_data(exchange: str = "", universe: str = "", limit: int = 25, **_: Any) -> Dict[str, Any]:
    try:
        key = str(exchange or "").strip().upper()
        if key not in EXCHANGES:
            return _tool_error(f"Unknown exchange '{exchange}'.")
        state = st.session_state
        universe_key = str(universe or "").strip() or (
            state.get("universe_key") if state.get("exchange_key") == key else "")
        if not universe_key:
            options = universes_for_exchange(key)
            universe_key = options[0].key if options else ""
        payload = load_market_frame(key, universe_key, enrich=True, enrich_limit=60)
        frame = payload.get("frame")
        try:
            count = int(limit)
        except Exception:
            count = 25
        rows = [] if frame is None or frame.empty else [
            {"symbol": r["symbol"], "name": r["name"], "price": r["price"], "currency": r["currency"],
             "change_percent": r["change_percent"], "market_cap": r["market_cap"],
             "volume": r["volume"], "sector": r["sector"]}
            for _, r in frame.head(max(1, min(count, 120))).iterrows()]
        return {"ok": bool(rows), "data": {
            "exchange": key, "universe": universe_key, "universe_label": resolve_universe(key, universe_key).label,
            "data_status": payload.get("data_status"), "session_date": payload.get("session_date"),
            "currency": payload.get("currency"), "rows": rows,
            "breadth": market_breadth(frame) if frame is not None else {},
            "note": payload.get("universe_note") or "",
        }}
    except Exception as exc:
        log_exception("tool heatmap", exc)
        return _tool_error(safe_error(exc))


def tool_get_sector_data(exchange: str = "", universe: str = "", **_: Any) -> Dict[str, Any]:
    try:
        key = str(exchange or "").strip().upper()
        if key not in EXCHANGES:
            return _tool_error(f"Unknown exchange '{exchange}'.")
        state = st.session_state
        universe_key = str(universe or "").strip() or (
            state.get("universe_key") if state.get("exchange_key") == key else "")
        if not universe_key:
            options = universes_for_exchange(key)
            universe_key = options[0].key if options else ""
        payload = load_market_frame(key, universe_key, enrich=True, enrich_limit=120)
        table = sector_table(payload.get("frame"))
        rows = [] if table.empty else table.to_dict("records")
        return {"ok": bool(rows), "data": {"exchange": key, "universe": universe_key,
                                           "data_status": payload.get("data_status"), "sectors": rows}}
    except Exception as exc:
        log_exception("tool sectors", exc)
        return _tool_error(safe_error(exc))


def tool_get_news(query: str = "", symbol: str = "", limit: int = 6, **_: Any) -> Dict[str, Any]:
    try:
        term = str(query or "").strip() or str(symbol or "").strip()
        if not term:
            return _tool_error("A query or symbol is required.")
        try:
            count = max(1, min(int(limit), 12))
        except Exception:
            count = 6
        payload = fetch_news(term, str(symbol or "").strip(), count)
        items = [{"title": i["title"], "publisher": i["publisher"], "age": i["age_label"],
                  "link": i["link"], "sentiment": news_sentiment(i["title"])["label"]}
                 for i in payload.get("items", [])]
        return {"ok": bool(items), "data": {"query": term, "items": items,
                                            "sources": payload.get("sources", []),
                                            "note": "" if items else "The provider returned no news for this query."}}
    except Exception as exc:
        log_exception("tool news", exc)
        return _tool_error(safe_error(exc))


TOOL_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "get_stock_data": tool_get_stock_data,
    "get_technical_indicators": tool_get_technical_indicators,
    "get_gann_levels": tool_get_gann_levels,
    "get_market_status": tool_get_market_status,
    "get_heatmap_data": tool_get_heatmap_data,
    "get_sector_data": tool_get_sector_data,
    "get_news": tool_get_news,
}


def run_tool(name: str, arguments: Any) -> Dict[str, Any]:
    """Execute one tool call with validated input. Never raises."""
    handler = TOOL_REGISTRY.get(str(name))
    if handler is None:
        return _tool_error(f"Unknown tool '{name}'.")
    args: Dict[str, Any] = {}
    if isinstance(arguments, dict):
        args = arguments
    elif isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                args = parsed
        except Exception:
            return _tool_error("Tool arguments were not valid JSON.")
    try:
        return handler(**args)
    except Exception as exc:
        log_exception(f"tool {name}", exc)
        return _tool_error(safe_error(exc))


# ===========================================================================
# SECTION 12 - CONTEXT BUILDER + ASSISTANT PERSONAS + CHAT ORCHESTRATOR
# ---------------------------------------------------------------------------
# The chat is a GENERAL conversational assistant: you can talk about anything -
# ideas, code, life, markets, half-formed thoughts. Trading-terminal data is
# offered as optional, clearly-labelled context; it is never a cage and the
# assistant is explicitly told not to force every answer back to stocks.
# ===========================================================================
ASSISTANT_PERSONAS: List[Dict[str, str]] = [
    {
        "key": "general",
        "label": "General assistant (talk about anything)",
        "prompt": (
            "You are a warm, sharp, genuinely conversational general assistant living inside a trading "
            "terminal app. The user can talk to you about ANYTHING - thoughts, feelings, ideas, plans, code, "
            "writing, studying, decisions, jokes, half-formed questions. Never force a market or stock angle "
            "onto a conversation that is not about markets. If market context is supplied below, use it only "
            "when it is actually relevant to what the user asked. Ask a clarifying question when the request "
            "is genuinely ambiguous. Never pretend to have data you were not given."),
    },
    {
        "key": "market",
        "label": "Market analyst (breadth, sectors, heat map)",
        "prompt": (
            "You are a market analyst for this terminal. Use the supplied market context and the tools to "
            "discuss breadth, sector rotation, the heat map and relative strength. Quote the data status "
            "(LIVE / DELAYED / LAST SESSION / CACHED) and the timestamp for every figure you cite. If the "
            "data does not establish something, say so plainly instead of guessing."),
    },
    {
        "key": "technical",
        "label": "Technical & Gann analyst (levels, structure)",
        "prompt": (
            "You are a technical analyst specialising in support/resistance, market structure, ATR, VWAP and "
            "the Square-of-9 Gann degree ladder used by this terminal. Always work from the real levels in "
            "context or from a tool call - never invent a price level. Present levels as numbers with their "
            "currency, explain what would invalidate a view, and state clearly when there is insufficient data."),
    },
    {
        "key": "news",
        "label": "News & catalysts",
        "prompt": (
            "You are a news/catalyst analyst. Only discuss headlines that were actually retrieved by the "
            "get_news tool or supplied in context, and always name the publisher. If no verified news is "
            "available, say: 'I don't have verified news data explaining this move.' Never fabricate a reason "
            "for a price move."),
    },
    {
        "key": "risk",
        "label": "Risk & position framing",
        "prompt": (
            "You are a risk coach. Frame every idea in terms of invalidation level, position sizing logic, "
            "risk-to-reward and what would make the thesis wrong. You do not give financial advice and you say "
            "so briefly when the user appears to be asking for a guaranteed outcome."),
    },
]
PERSONAS_BY_KEY = {p["key"]: p for p in ASSISTANT_PERSONAS}

BASE_RULES = """
HOW TO ANSWER
- Answer the actual question first, in the first sentence. No filler openings.
- For market questions, structure as: current context -> technical picture -> key levels -> volume/liquidity
  -> news (only if retrieved) -> risk considerations -> what the data does NOT establish.
- Separate OBSERVED provider data, CALCULATED values, RETRIEVED news and YOUR OWN interpretation. Never
  present interpretation as fact.
- Respect the data status: if it is LAST SESSION or CACHED, say "the latest available session data shows..."
  instead of "is currently trading at".
- Every price you quote must carry the instrument's own currency (e.g. ₹1,234.50 for .NS / .BO names,
  $189.40 for US names). Never use a dollar sign for an Indian instrument.
- Never invent prices, news, tool results or certainty. If something is unavailable, say it is unavailable.
- You are not a licensed financial adviser; keep a short risk note where a trade decision is involved.
"""


def build_context_block() -> str:
    """A concise snapshot of what the user is looking at right now."""
    state = st.session_state
    lines: List[str] = []
    try:
        exchange_key = state.get("exchange_key", "NSE")
        universe_key = state.get("universe_key", "")
        selected = state.get("selected_yf", "")
        session = session_status(exchange_key)
        exch = EXCHANGES.get(exchange_key)
        lines.append("CURRENT TERMINAL CONTEXT")
        lines.append(f"- Selected market: {MARKETS_BY_KEY.get(exch.market, Market('', '', '', '', ())).label if exch else '—'}")
        lines.append(f"- Selected exchange: {exchange_key} ({exch.name if exch else '—'})")
        lines.append(f"- Exchange-local time: {session['local_time']}")
        lines.append(f"- Session status: {session['status']} ({session['reason']})")
        if universe_key:
            universe = resolve_universe(exchange_key, universe_key)
            lines.append(f"- Loaded universe: {universe.label} [{universe.key}]")
        if selected:
            snapshot = symbol_snapshot(selected)
            status = infer_data_status(snapshot.get("exchange") or exchange_key,
                                       [{"last_bar_date": snapshot.get("last_bar_date")}])[0]
            lines.append(f"- Selected instrument: {snapshot.get('name') or selected} ({selected})")
            lines.append(f"- Currency: {snapshot.get('currency')} | Data status: {status}"
                         f" | Last bar: {snapshot.get('last_bar_date') or '—'}")
            if snapshot.get("price"):
                lines.append(f"- Price: {fmt_price(snapshot['price'], snapshot['currency'])} "
                             f"({fmt_pct(snapshot.get('change_percent'))})")
            if snapshot.get("sector"):
                lines.append(f"- Sector: {snapshot['sector']}"
                             + (f" | Industry: {snapshot['industry']}" if snapshot.get("industry") else ""))
            if snapshot.get("market_cap"):
                lines.append(f"- Market cap: {fmt_cap(snapshot['market_cap'], snapshot['currency'])}")
            levels = get_session_fixed_levels(selected)
            if levels:
                anchor = valid_price(levels.get("_ref_close"))
                lines.append(f"- Gann 0° anchor: {fmt_price(anchor, snapshot['currency'])} "
                             f"({levels.get('_ref')} @ {levels.get('_ref_time')})")
                ladder = [f"{r['label']}={fmt_price(r['value'], snapshot['currency'])}"
                          for r in gann_degree_table(levels, snapshot.get("price"), snapshot["currency"])]
                if ladder:
                    lines.append("- Degree ladder: " + " | ".join(ladder))
        frame = state.get("_last_frame")
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            breadth = market_breadth(frame)
            lines.append(f"- Loaded universe breadth: {breadth['advances']} up / {breadth['declines']} down / "
                         f"{breadth['unchanged']} flat (A/D {breadth['ad_ratio']}), average "
                         f"{fmt_pct(breadth['avg_change'])}, data status {state.get('_last_status', '—')}")
    except Exception as exc:
        log_exception("context builder", exc)
        lines.append("(Some terminal context could not be assembled this run.)")
    lines.append("")
    lines.append("The user may ask about anything at all - do not assume the question is about markets.")
    return "\n".join(lines)


@dataclass
class ChatState:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    model_label: str = ""
    custom_provider: str = "groq"
    custom_model_id: str = ""
    persona_key: str = "general"
    use_tools: bool = True
    stream: bool = True
    pending: Optional[str] = None
    last_error: str = ""
    last_used_tools: List[str] = field(default_factory=list)

    def spec(self) -> ModelSpec:
        return get_model_spec(self.model_label, self.custom_provider, self.custom_model_id)

    def persona(self) -> Dict[str, str]:
        return PERSONAS_BY_KEY.get(self.persona_key, ASSISTANT_PERSONAS[0])


def init_chat_state() -> ChatState:
    state = st.session_state
    chat = state.get("chat")
    if not isinstance(chat, ChatState):
        chat = ChatState(model_label=model_labels()[0])
        state["chat"] = chat
    if not chat.model_label or (chat.model_label not in model_labels()):
        chat.model_label = model_labels()[0]
    return chat


MARKET_HINTS = (
    "stock", "share", "price", "chart", "support", "resistance", "level", "gann", "square", "vwap",
    "atr", "rsi", "ema", "volume", "market", "nifty", "sensex", "nasdaq", "s&p", "index", "sector",
    "heatmap", "heat map", "breadth", "news", "alert", "trade", "entry", "stop", "target", "trend",
    "breakout", "candle", "bullish", "bearish", "portfolio", "watchlist", "crypto", "bitcoin", "gold",
    "forex", "usd", "inr", "earnings", "valuation",
)


def looks_market_related(text: str, selected_symbol: str = "") -> bool:
    low = (text or "").lower()
    if any(hint in low for hint in MARKET_HINTS):
        return True
    if selected_symbol and selected_symbol.split(".")[0].lower() in low:
        return True
    return bool(re.search(r"\b[A-Z]{2,10}(\.(NS|BO|TO|L|DE|T|HK|AX|SI|KS|SS|SZ))?\b", text or ""))


def chat_completion(chat: ChatState, system: str, tools: Optional[List[Dict[str, Any]]] = None
                    ) -> Tuple[str, List[Dict[str, str]]]:
    provider = make_provider(chat.spec())
    if not provider.is_configured():
        raise AIUnavailable(f"{provider.label} is not configured. Add "
                            f"{'GROQ_KEY' if chat.spec().provider == 'groq' else 'GEMINI_KEY'} "
                            f"in Streamlit secrets to use this model.")
    return provider.complete_chat(system, chat.messages, tools=tools)


def chat_stream(chat: ChatState, system: str) -> Iterable[str]:
    provider = make_provider(chat.spec())
    if not provider.is_configured():
        raise AIUnavailable(f"{provider.label} is not configured. Add "
                            f"{'GROQ_KEY' if chat.spec().provider == 'groq' else 'GEMINI_KEY'} "
                            f"in Streamlit secrets to use this model.")
    yield from provider.stream_chat(system, chat.messages)


def _execute_tool_calls(calls: List[Dict[str, str]]) -> List[str]:
    """Run requested tools, feeding results back into the conversation."""
    used: List[str] = []
    for call in calls:
        name = str(call.get("name") or "")
        result = run_tool(name, call.get("arguments"))
        used.append(name)
        chat_append({"role": "system", "name": name,
                     "content": f"TOOL RESULT from {name}: {json.dumps(result, default=str)[:4000]}"})
    return used


def chat_append(message: Dict[str, Any]) -> None:
    chat = init_chat_state()
    chat.messages.append(message)
    if len(chat.messages) > 60:
        chat.messages = chat.messages[-60:]


def trim_history(chat: ChatState, keep: int = 20) -> List[Dict[str, str]]:
    """Recent turns only - the context window stays small and cheap."""
    history: List[Dict[str, str]] = []
    for message in chat.messages[-keep:]:
        role = message.get("role")
        if role in ("user", "assistant") and (message.get("content") or "").strip():
            history.append({"role": role, "content": message["content"]})
    return history


def chat_generate(chat: ChatState, placeholder, user_text: str) -> str:
    """
    One full assistant turn: optional tool round -> streaming answer.
    Returns the assistant text ('' when the provider failed).
    """
    chat.last_error = ""
    chat.last_used_tools = []
    system = chat.persona()["prompt"] + "\n" + BASE_RULES + "\n" + build_context_block()

    # --- optional tool round (only when the question is actually about the terminal) ---
    if chat.use_tools and looks_market_related(user_text, st.session_state.get("selected_symbol", "")):
        try:
            spec = chat.spec()
            if make_provider(spec).supports_tools:
                placeholder.markdown("*Checking live terminal data…*")
                _, calls = chat_completion(chat, system, tools=TOOL_SCHEMAS)
                if calls:
                    chat.last_used_tools = _execute_tool_calls(calls)
        except AIUnavailable as exc:
            chat.last_error = str(exc)
            placeholder.markdown(f"⚠️ {exc}")
            return ""
        except Exception as exc:
            log_exception("chat tool round", exc)   # a failed tool round must not kill the answer
            chat.last_error = safe_error(exc)

    # --- the answer itself ---
    collected: List[str] = []
    try:
        if chat.stream:
            placeholder.markdown("*AI is analyzing…*")
            for piece in chat_stream(chat, system):
                collected.append(piece)
                placeholder.markdown("".join(collected))
        else:
            with st.spinner("AI is analyzing…"):
                text, _ = chat_completion(chat, system)
                collected.append(text)
                placeholder.markdown(text)
    except AIUnavailable as exc:
        chat.last_error = str(exc)
        placeholder.markdown(f"⚠️ {exc}")
        return ""
    except Exception as exc:
        log_exception("chat stream", exc)
        chat.last_error = friendly_ai_error(exc)
        placeholder.markdown(f"⚠️ {chat.last_error}")
        return ""

    answer = "".join(collected).strip()
    if not answer:
        chat.last_error = "The model returned an empty response."
        placeholder.markdown("⚠️ AI response unavailable. Please try again.")
        return ""
    placeholder.markdown(answer)
    return answer


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model reply (handles ``` fences and prose)."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return raw[start:end + 1]
    return raw


def ai_impact_summary(title: str, symbol: str, currency: str = "USD",
                      model_label: str = "") -> Optional[str]:
    """A real model call for the news 'Summary & impact' button (no canned text)."""
    try:
        spec = get_model_spec(model_label or model_labels()[0])
        provider = make_provider(spec)
        if not provider.is_configured():
            return None
        system = ("You are a concise financial news explainer. In exactly two short sentences: (1) what this "
                  "headline means, (2) how it could plausibly affect the named instrument. Be practical, hedge "
                  "appropriately, and never invent figures that are not in the headline. Note that prices for "
                  f"this instrument are quoted in {currency}.")
        text, _ = provider.complete_chat(system, [{"role": "user",
                                                   "content": f"Headline: {title}\nInstrument: {symbol}"}])
        return text.strip() or None
    except Exception as exc:
        log_exception("news impact summary", exc)
        return None


# ===========================================================================
# SECTION 13 - UI PRIMITIVES
# ===========================================================================
def metric_html(label: str, value: str, cls: str = "", sub: str = "") -> str:
    sub_html = f"<div class='card-sub' style='margin:2px 0 0 0'>{sub}</div>" if sub else ""
    return (f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value {cls}'>{value}</div>{sub_html}")


def change_class(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "flat"
    return "up" if number > 0 else ("down" if number < 0 else "flat")


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='card-header' style='font-size:1.15rem'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='card-sub'>{subtitle}</div>", unsafe_allow_html=True)


def render_global_selector() -> bool:
    """🌎 GLOBAL MARKETS → Market / Exchange / Universe (adapts to provider support)."""
    state = st.session_state
    market_keys = [m.key for m in MARKETS]
    market_labels = [f"{m.flag} {m.label}" for m in MARKETS]
    col_market, col_exchange, col_universe, col_refresh = st.columns([1.3, 1.7, 2.6, 0.7])

    with col_market:
        m_idx = market_keys.index(state.market_key) if state.market_key in market_keys else 0
        pick_market = st.selectbox("🌎 Market", market_labels, index=m_idx, key="market_selector")
        new_market = market_keys[market_labels.index(pick_market)]
    market = MARKETS_BY_KEY.get(new_market, MARKETS[0])

    exchange_keys = [k for k in market.exchanges if k in EXCHANGES]
    with col_exchange:
        e_labels = [EXCHANGES[k].name for k in exchange_keys]
        e_idx = exchange_keys.index(state.exchange_key) if state.exchange_key in exchange_keys else 0
        pick_exchange = st.selectbox("Exchange", e_labels, index=e_idx,
                                     key=f"exchange_selector_{new_market}")
        new_exchange = exchange_keys[e_labels.index(pick_exchange)] if exchange_keys else state.exchange_key

    options = universes_for_exchange(new_exchange)
    with col_universe:
        if options:
            u_labels = [u.label for u in options]
            u_keys = [u.key for u in options]
            u_idx = u_keys.index(state.universe_key) if state.universe_key in u_keys else 0
            pick_universe = st.selectbox("Universe", u_labels, index=u_idx,
                                         key=f"universe_selector_{new_exchange}")
            new_universe = u_keys[u_labels.index(pick_universe)]
        else:
            new_universe = ""
            st.selectbox("Universe", ["No universe configured"], disabled=True)

    with col_refresh:
        st.write("")
        refresh = st.button("⟳", use_container_width=True, help="Force a fresh pull from the provider")

    if (new_market, new_exchange, new_universe) != (state.market_key, state.exchange_key, state.universe_key):
        state.market_key, state.exchange_key, state.universe_key = new_market, new_exchange, new_universe
        st.rerun()
    return refresh


def render_session_strip(exchange_key: str, data_status: str = "", session_date: str = "",
                         currency: str = "", extra: str = "") -> None:
    session = session_status(exchange_key)
    exch = EXCHANGES.get(exchange_key)
    parts = [
        f"<b>{exch.name if exch else exchange_key}</b>",
        session_pill(session["status"]),
        f"<span style='color:#94a3b8'>Local time {session['local_time']}</span>",
    ]
    if currency:
        parts.append(f"<span style='color:#94a3b8'>Currency {currency_display(currency)}</span>")
    if data_status:
        parts.append(status_pill(data_status))
        parts.append(f"<span style='color:#64748b'>as of {session_date or '—'}</span>")
    if extra:
        parts.append(f"<span style='color:#64748b'>{extra}</span>")
    st.markdown("<div style='display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:6px'>"
                + "".join(parts) + "</div>", unsafe_allow_html=True)


def candles_chart(yf_symbol: str, levels: Dict[str, Any], price: Optional[float],
                  currency: str, height: int = 400):
    frame = _history_cached(yf_symbol, "5d", "15m")
    label = "15-minute bars · last 5 sessions"
    if frame is None or frame.empty:
        frame = _history_cached(yf_symbol, "1mo", "1d")
        label = "Daily bars · last month (intraday bars unavailable)"
    if frame is None or frame.empty:
        return None, "No candle data available from the provider right now."
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=frame.index, open=frame["Open"], high=frame["High"], low=frame["Low"], close=frame["Close"],
        increasing_line_color="#34d399", increasing_fillcolor="#34d399",
        decreasing_line_color="#f87171", decreasing_fillcolor="#f87171", name="Price"))
    for key, colour in (("R2", "#f87171"), ("R1", "#fb923c"), ("S1", "#4ade80"), ("S2", "#34d399")):
        value = _num(levels.get(key)) if levels else None
        if value and value > 0:
            fig.add_hline(y=value, line_dash="dot", line_color=colour, line_width=1,
                          annotation_text=key, annotation_position="right",
                          annotation_font_color=colour, annotation_font_size=10)
    if valid_price(price):
        fig.add_hline(y=float(price), line_color="#fbbf24", line_width=1.5,
                      annotation_text=fmt_price(price, currency), annotation_position="left",
                      annotation_font_color="#fbbf24", annotation_font_size=11)
    fig.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", height=height,
                      margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
                      showlegend=False, font=dict(color="#94a3b8", size=11),
                      xaxis=dict(gridcolor="rgba(51,65,85,0.3)"),
                      yaxis=dict(gridcolor="rgba(51,65,85,0.3)", side="right"))
    return fig, label


# ===========================================================================
# SECTION 14 - PAGE: DASHBOARD
# ===========================================================================
def explain_quote_error(error: str) -> str:
    """Turn a provider error string into a plain-language cause for the user."""
    low = (error or "").lower()
    if not error:
        return "the provider returned nothing for this instrument"
    if "close" in low:
        return "the provider's response contained no close column for this instrument"
    if "no rows" in low or "no data" in low or "empty" in low:
        return "the provider returned no price bars for this instrument"
    if "invalid price" in low or "zero" in low or "nan" in low:
        return "the provider returned a zero or non-numeric price"
    return error


def market_cap_note(snapshot: Dict[str, Any]) -> str:
    """Honest explanation for a missing market cap (futures/FX/crypto/indices have none)."""
    if snapshot.get("market_cap"):
        return ""
    exchange_key = snapshot.get("exchange") or ""
    symbol = str(snapshot.get("yf_symbol") or snapshot.get("symbol") or "")
    if exchange_key == "CME" or symbol.endswith("=F"):
        return "n/a — futures contracts have no market cap"
    if exchange_key == "FX" or symbol.endswith("=X"):
        return "n/a — FX pairs have no market cap"
    if exchange_key == "CRYPTO" or symbol.endswith("-USD"):
        return "n/a — crypto has no market cap from this provider"
    if exchange_key == "GLOBAL" or symbol.startswith("^"):
        return "n/a — index level, not a company"
    return "not reported by the provider"


def page_dashboard(refresh: bool) -> None:
    state = st.session_state
    with st.container(border=True):
        page_header("Select a market &amp; find a symbol",
                    "Search any listed instrument worldwide, or quick-pick from the curated list.")
        quick_col, search_col = st.columns([1, 1.4])
        with quick_col:
            cat_keys = [c["key"] for c in MARKET_CATEGORIES]
            cat_labels = [c["label"] for c in MARKET_CATEGORIES]
            current_cat = state.get("market_cat", "india")
            cat_idx = cat_keys.index(current_cat) if current_cat in cat_keys else 0
            picked_cat = st.selectbox("Curated group", cat_labels, index=cat_idx, key="curated_group")
            new_cat = cat_keys[cat_labels.index(picked_cat)]
            quick_items = [m for m in MARKET_ITEMS if m["category"] == new_cat]
            quick_labels = [f"{m['symbol']} — {m['name']} ({m['currency']})" for m in quick_items]
            pick_quick = st.selectbox("Quick pick", quick_labels, key=f"quick_pick_{new_cat}")
            if st.button("Load selected symbol", use_container_width=True, key="quick_load"):
                chosen = quick_items[quick_labels.index(pick_quick)]
                state.selected_yf = chosen["yf"]
                state.selected_symbol = chosen["symbol"]
                state.market_cat = new_cat
                state.exchange_key = _exchange_for_yf_symbol(chosen["yf"])
                st.rerun()
            if new_cat != current_cat:
                state.market_cat = new_cat
        with search_col:
            search_input = st.text_input("Search any stock / index / crypto / FX worldwide",
                                         placeholder="e.g. Reliance, Apple, Bitcoin, Nifty 50, EURUSD, SAP",
                                         key="global_search")
            if search_input.strip():
                results = yahoo_symbol_search(search_input.strip())
                if results:
                    labels = [f"{r['symbol']} — {r['name']} ({r['exchange']}) · {r['currency']}" for r in results]
                    pick = st.selectbox("Matching symbols", labels, key="search_pick")
                    if st.button("Analyze this symbol", use_container_width=True, key="search_go"):
                        chosen = results[labels.index(pick)]
                        state.selected_yf = chosen["symbol"]
                        state.selected_symbol = chosen["symbol"].replace(".NS", "").replace(".BO", "")
                        state.exchange_key = _exchange_for_yf_symbol(chosen["symbol"])
                        st.rerun()
                else:
                    st.caption("No matches from the provider — try a different spelling or the full ticker.")

    yf_symbol = state.get("selected_yf", "RELIANCE.NS")
    snapshot = symbol_snapshot(yf_symbol)
    currency = snapshot.get("currency") or "USD"
    price = snapshot.get("price")
    levels = get_session_fixed_levels(yf_symbol)
    if not levels:
        levels = generate_levels(snapshot.get("day_high"), snapshot.get("day_low"),
                                 snapshot.get("previous_close") or price)
    indicators = technical_indicators(yf_symbol)
    status = infer_data_status(snapshot.get("exchange") or state.get("exchange_key", "NSE"),
                               [{"last_bar_date": snapshot.get("last_bar_date")}])[0]
    if refresh:
        for cached in (_quotes_cached, _profile_cached, _history_cached, _symbols_cached):
            try:
                cached.clear()
            except Exception as exc:
                log_exception("cache clear", exc)

    with st.container(border=True):
        st.markdown(f"<div class='card-header'>{snapshot.get('name') or yf_symbol} "
                    f"<span style='color:#64748b;font-size:0.85rem'>{state.get('selected_symbol', '')}</span></div>",
                    unsafe_allow_html=True)
        render_session_strip(snapshot.get("exchange") or state.get("exchange_key", "NSE"), status,
                             snapshot.get("last_bar_date") or "", currency,
                             extra=f"{yf_symbol} · {snapshot.get('sector') or 'sector n/a'}")
        if not snapshot.get("ok"):
            st.warning(f"No usable quote for **{yf_symbol}** right now — "
                       f"{explain_quote_error(snapshot.get('error'))}. "
                       "Levels below, if any, come from the last available bars.")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(metric_html("Price", fmt_price(price, currency),
                                change_class(snapshot.get("change_percent")),
                                f"{fmt_pct(snapshot.get('change_percent'))} · {currency}"), unsafe_allow_html=True)
        c2.markdown(metric_html("Day range",
                                f"{fmt_price(snapshot.get('day_low'), currency)} – {fmt_price(snapshot.get('day_high'), currency)}",
                                sub=f"open {fmt_price(snapshot.get('open'), currency)}"), unsafe_allow_html=True)
        c3.markdown(metric_html("Previous close", fmt_price(snapshot.get("previous_close"), currency),
                                sub=f"chg {fmt_price(snapshot.get('change'), currency)}"), unsafe_allow_html=True)
        c4.markdown(metric_html("Volume", fmt_volume(snapshot.get("volume")),
                                sub=f"rel-vol {indicators.get('relative_volume'):.2f}×"
                                    if indicators.get("relative_volume") else "rel-vol n/a"), unsafe_allow_html=True)
        c5.markdown(metric_html("Market cap", fmt_cap(snapshot.get("market_cap"), currency),
                                sub=market_cap_note(snapshot) or
                                    f"ATR {fmt_price(indicators.get('atr'), currency)}"), unsafe_allow_html=True)
        st.markdown(f"<div class='card-sub' style='margin-top:8px'>{data_status_explanation(status)}</div>",
                    unsafe_allow_html=True)

    chart_col, plan_col = st.columns([3, 2])
    with chart_col:
        with st.container(border=True):
            st.markdown(f"<div class='card-header'>{_shorten(snapshot.get('name') or state.get('selected_symbol', ''), 34)}"
                        f" · {state.get('selected_symbol', '')} · Price action</div>", unsafe_allow_html=True)
            figure, note = candles_chart(yf_symbol, levels, price, currency)
            st.markdown(f"<div class='card-sub'>{note}</div>", unsafe_allow_html=True)
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True)
            else:
                st.info("No candle data available — the provider returned no bars for this instrument.")
    with plan_col:
        with st.container(border=True):
            st.markdown("<div class='card-header'>⚡ Trade plans from real levels</div>", unsafe_allow_html=True)
            st.markdown("<div class='card-sub'>Derived only from the level set below — no invented signals</div>",
                        unsafe_allow_html=True)
            plans = generate_suggestions(price, levels, indicators)
            bullish = next((p for p in plans if p["direction"] == "bullish"), None)
            bearish = next((p for p in plans if p["direction"] == "bearish"), None)
            if bullish:
                st.markdown(f"""<div class="plan-card plan-bull">
<div class="plan-title-bull">▲ BULLISH PLAN</div>
<div>Entry <b>{fmt_price(bullish['entry'], currency)}</b> · Stop <b>{fmt_price(bullish['stop'], currency)}</b></div>
<div>T1 <b>{fmt_price(bullish['t1'], currency)}</b> (R:R {bullish['rr1']}) ·
T2 <b>{fmt_price(bullish['t2'], currency)}</b> (R:R {bullish['rr2']})</div>
<div style="font-size:0.72rem;color:#94a3b8;margin-top:4px">{bullish['basis']}</div></div>""",
                            unsafe_allow_html=True)
            if bearish:
                st.markdown(f"""<div class="plan-card plan-bear">
<div class="plan-title-bear">▼ BEARISH PLAN</div>
<div>Entry <b>{fmt_price(bearish['entry'], currency)}</b> · Stop <b>{fmt_price(bearish['stop'], currency)}</b></div>
<div>T1 <b>{fmt_price(bearish['t1'], currency)}</b> (R:R {bearish['rr1']}) ·
T2 <b>{fmt_price(bearish['t2'], currency)}</b> (R:R {bearish['rr2']})</div>
<div style="font-size:0.72rem;color:#94a3b8;margin-top:4px">{bearish['basis']}</div></div>""",
                            unsafe_allow_html=True)
            if not plans:
                st.info("Insufficient data to build a plan — a support and a resistance level are both required.")
            st.caption("Educational levels only, not investment advice. Size positions so a stop-out is survivable.")

    ladder_col, indicator_col = st.columns([1.35, 1])
    with ladder_col:
        with st.container(border=True):
            st.markdown("<div class='card-header'>Square-of-9 Gann degree ladder</div>", unsafe_allow_html=True)
            reference = valid_price(levels.get("_ref_close")) if levels else None
            st.markdown(f"<div class='card-sub'>0° = first 15-minute closing price "
                        f"({fmt_price(reference, currency)}) · R/S levels use √price ± degree/180 · "
                        f"fixed for the session · source: {levels.get('_ref', 'n/a') if levels else 'n/a'}</div>",
                        unsafe_allow_html=True)
            table = gann_degree_table(levels, price, currency)
            if table:
                for row in table:
                    if row["kind"] == "anchor":
                        st.markdown(f"<div style='text-align:center;padding:10px;background:rgba(251,191,36,0.14);"
                                    f"border-radius:10px;margin:8px 0;border:1px solid rgba(251,191,36,0.4)'>"
                                    f"<span class='amber'><b>0°</b> anchor {row['text']}</span>"
                                    f"<span style='color:#e2e8f0'> · current {fmt_price(price, currency)}</span>"
                                    f"</div>", unsafe_allow_html=True)
                        continue
                    css = "level-resistance" if row["kind"] == "resistance" else "level-support"
                    colour = "#f87171" if row["kind"] == "resistance" else "#34d399"
                    st.markdown(f"<div class='level-row {css}'>"
                                f"<span><b style='color:{colour}'>{row['degree']:g}° {row['label'][0]}</b> "
                                f"&nbsp; {row['text']}</span>"
                                f"<span class='{change_class(row['pct'])}'>{fmt_pct(row['pct'])}</span></div>",
                                unsafe_allow_html=True)
            else:
                st.info("Levels unavailable for this instrument right now.")
            with st.expander("Classic pivot support / resistance"):
                classic = {k: v for k, v in (levels or {}).items()
                           if k.startswith(("R", "S")) and not k.startswith(("R_", "S_")) and not k.startswith("R0")}
                if classic:
                    rows = [{"Level": k, "Price": fmt_price(v, currency),
                             "% vs price": fmt_pct(((v - price) / price * 100) if price else None)}
                            for k, v in sorted(classic.items())]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("Classic pivots need a session high/low/close — not available yet.")

    with indicator_col:
        with st.container(border=True):
            st.markdown("<div class='card-header'>Technical read</div>", unsafe_allow_html=True)
            if indicators.get("sufficient"):
                rows = [
                    ("Trend", indicators.get("trend") or "—"),
                    ("EMA bias", indicators.get("ema_bias") or "—"),
                    ("EMA 9 / 50", f"{fmt_price(indicators.get('ema9'), currency)} / "
                                   f"{fmt_price(indicators.get('ema50'), currency)}"),
                    ("RSI(14)", f"{indicators['rsi']:.1f} ({indicators.get('rsi_label')})"
                     if indicators.get("rsi") is not None else "—"),
                    ("ATR(14)", f"{fmt_price(indicators.get('atr'), currency)}"
                                + (f" ({indicators['atr_percent']:.2f}%)" if indicators.get("atr_percent") else "")),
                    ("VWAP (session)", fmt_price(indicators.get("vwap"), currency)),
                    ("Swing high / low (60d)", f"{fmt_price(indicators.get('swing_high'), currency)} / "
                                               f"{fmt_price(indicators.get('swing_low'), currency)}"),
                    ("Candle pattern", indicators.get("candle_pattern") or "—"),
                    ("Liquidity (20d avg)", fmt_cap(indicators.get("liquidity"), currency)),
                ]
                st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]),
                             use_container_width=True, hide_index=True)
            else:
                st.info(indicators.get("note") or "Insufficient data for indicators.")
            frame = state.get("_last_frame")
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                breadth = market_breadth(frame)
                st.markdown(f"<div class='card-sub' style='margin-top:8px'>Loaded universe breadth "
                            f"({state.get('exchange_key')} · {state.get('_last_status', '—')}): "
                            f"{breadth['advances']} up / {breadth['declines']} down / {breadth['unchanged']} flat</div>",
                            unsafe_allow_html=True)


# ===========================================================================
# SECTION 15 - PAGE: NEWS
# ---------------------------------------------------------------------------
# Every article goes through parse_news_items(). A malformed payload, a missing
# link or a string where a dict was expected degrades to a placeholder - it can
# never raise, which is the bug that produced the original AttributeError.
# ===========================================================================
def page_news() -> None:
    state = st.session_state
    symbol = state.get("selected_symbol", "")
    yf_symbol = state.get("selected_yf", "")
    snapshot = symbol_snapshot(yf_symbol) if yf_symbol else {}
    currency = snapshot.get("currency") or "USD"

    with st.container(border=True):
        page_header("News &amp; sentiment",
                    "Headlines come straight from the configured provider. Nothing is invented — when the "
                    "provider returns nothing, this page says so.")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            topic = st.text_input("Search news", value=snapshot.get("name") or symbol,
                                  placeholder="e.g. Reliance Industries, semiconductor tariffs, RBI policy",
                                  key="news_topic")
        with col_b:
            st.write("")
            only_symbol = st.checkbox("Instrument feed only", value=False, key="news_instrument_only",
                                      help="Query the instrument's own news feed instead of the free-text topic.")

    query = yf_symbol if (only_symbol and yf_symbol) else (topic or yf_symbol)
    with st.spinner("Loading news…"):
        payload = fetch_news(query, "" if only_symbol else yf_symbol, limit=12)
    items = payload.get("items", [])

    st.markdown("<div class='card-sub'>"
                + (f"Source: {', '.join(payload.get('sources', []))}" if payload.get("sources")
                   else "No provider source returned data")
                + f" · fetched {datetime.now(timezone.utc).strftime('%H:%M UTC')}</div>",
                unsafe_allow_html=True)

    if not items:
        st.info("No news available from the provider for this query right now. "
                "This build never substitutes fabricated headlines.")
        if payload.get("errors"):
            with st.expander("Provider errors (technical)"):
                st.code("\n".join(payload["errors"]))
        return

    if "news_open" not in state:
        state.news_open = None

    for index, item in enumerate(items):
        sentiment = news_sentiment(item["title"], item.get("summary", ""))
        score_text = sentiment["score"] if sentiment["score"] is not None else "–"
        matched = (sentiment.get("bullish_terms") or []) + (sentiment.get("bearish_terms") or [])
        audit = ("matched: " + ", ".join(matched[:4])) if matched else "no keyword matched"
        st.markdown(f"""
        <div style="background:#0B1220;border:1px solid rgba(51,65,85,0.5);border-radius:14px;padding:14px 16px;margin-bottom:8px;display:flex;gap:14px;align-items:flex-start;">
            <div style="min-width:44px;height:44px;border-radius:12px;background:rgba(30,41,59,0.9);color:{sentiment['colour']};font-weight:700;font-size:1.05rem;display:flex;align-items:center;justify-content:center;">{score_text}</div>
            <div style="flex:1;">
                <div style="color:#f1f5f9;font-weight:600;font-size:0.95rem;line-height:1.35;">{html.escape(item['title'])}</div>
                <div style="margin-top:6px;font-size:0.78rem;color:#64748b;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                    <span>{html.escape(item['publisher'])}</span><span>·</span><span>{item['age_label']}</span>
                    <span style="background:rgba(148,163,184,0.14);color:{sentiment['colour']};padding:2px 9px;border-radius:999px;font-weight:600;font-size:0.72rem;">{sentiment['label']} (keyword heuristic)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if item.get("summary"):
            st.caption(item["summary"][:400])
        action_left, action_right = st.columns([1, 1])
        with action_left:
            if st.button("Summary & impact", key=f"news_sum_{index}", use_container_width=True):
                state.news_open = index if state.news_open != index else None
                st.rerun()
        with action_right:
            if item.get("link"):
                st.markdown(f"<a href='{item['link']}' target='_blank' rel='noopener' "
                            f"style='display:block;text-align:center;padding:0.4rem 0.6rem;border-radius:8px;"
                            f"background:#1E293B;border:1px solid rgba(51,65,85,0.6);color:#e2e8f0;"
                            f"text-decoration:none;font-size:0.85rem;'>Open full article ↗</a>",
                            unsafe_allow_html=True)
            else:
                st.caption("No external link in the provider payload")

        if state.news_open == index:
            with st.container(border=True):
                st.markdown("<div class='card-sub'>Sentiment read</div>", unsafe_allow_html=True)
                st.caption(f"Keyword heuristic: {score_text}/10 · {sentiment['label']} · {audit}")
                st.markdown("<div class='card-sub'>Summary &amp; impact (generated by the configured AI model)</div>",
                            unsafe_allow_html=True)
                chat_state = state.get("chat")
                model_label = chat_state.model_label if isinstance(chat_state, ChatState) else ""
                summary = ai_impact_summary(item["title"], symbol or yf_symbol, currency, model_label)
                if summary:
                    st.markdown(summary)
                    st.caption("Model-generated interpretation — not a verified fact about the company.")
                else:
                    st.warning("AI summary unavailable (model not configured, or the provider returned an error). "
                               "The headline above is unchanged provider data.")


# ===========================================================================
# SECTION 16 - PAGE: HEAT MAP (dynamic, market-agnostic engine)
# ===========================================================================
def page_heatmap(refresh: bool) -> None:
    state = st.session_state
    exchange_key = state.get("exchange_key", "NSE")
    universe_key = state.get("universe_key", "")

    with st.container(border=True):
        page_header("Dynamic market heat map",
                    "Size = market cap (or volume / equal weight) · Colour = % change · Grouping by sector, "
                    "industry or flat. Built from whatever universe is selected, whatever its size.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            size_mode = st.selectbox("Tile size", ["Market cap", "Volume", "Equal weight"], index=0, key="heat_size")
        with c2:
            color_mode = st.selectbox("Tile colour", ["% change"], index=0, key="heat_color",
                                      help="Only percentage change is used for colour, because it is the metric "
                                           "the provider returns reliably for every instrument.")
        with c3:
            group_mode = st.selectbox("Grouping", ["Sectors", "Industries", "Flat"], index=0, key="heat_group")
        with c4:
            max_tiles = st.slider("Max tiles", min_value=20, max_value=400, value=150, step=10, key="heat_max")

    with st.spinner("Loading market data…"):
        payload = load_market_frame(exchange_key, universe_key, enrich=True, enrich_limit=160,
                                    force_refresh=refresh)
    frame = payload.get("frame")
    state["_last_frame"] = frame
    state["_last_status"] = payload.get("data_status")
    state["_last_session_date"] = payload.get("session_date")

    with st.container(border=True):
        render_session_strip(exchange_key, payload.get("data_status", ""), payload.get("session_date", ""),
                             payload.get("currency", ""),
                             extra=f"{len(frame) if frame is not None else 0} instruments · "
                                   f"symbols from {payload.get('symbol_source', 'n/a')}")
        if payload.get("universe_note"):
            st.caption(payload["universe_note"])
        if payload.get("errors"):
            with st.expander(f"Provider notes ({len(payload['errors'])})"):
                for message in payload["errors"]:
                    st.write("• " + message)
        if frame is None or frame.empty:
            st.warning("No market data currently available for this universe. Nothing is displayed rather than "
                       "showing invented values.")
            return
        figure = build_heatmap_figure(frame, size_mode=size_mode, color_mode=color_mode,
                                      group_mode=group_mode, currency=payload.get("currency", "USD"),
                                      max_tiles=int(max_tiles))
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.warning("The heat map could not be built from the available rows.")

    breadth = market_breadth(frame)
    breadth_col, sector_col = st.columns([1, 1.6])
    with breadth_col:
        with st.container(border=True):
            st.markdown("<div class='card-header'>Market breadth</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-sub'>{breadth['total']} instruments with a usable % change</div>",
                        unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            b1.markdown(metric_html("Advances", f"<span class='up'>{breadth['advances']}</span>"),
                        unsafe_allow_html=True)
            b2.markdown(metric_html("Declines", f"<span class='down'>{breadth['declines']}</span>"),
                        unsafe_allow_html=True)
            b3.markdown(metric_html("Unchanged", str(breadth["unchanged"])), unsafe_allow_html=True)
            st.markdown(metric_html("Advance / decline ratio",
                                    f"{breadth['ad_ratio']:.2f}" if breadth["ad_ratio"] else "—",
                                    sub=f"average change {fmt_pct(breadth['avg_change'])} · "
                                        f"median {fmt_pct(breadth['median_change'])}"),
                        unsafe_allow_html=True)
            st.caption(breadth["note"])
            if st.button("Compute 52-week highs / lows (capped)", key="breadth_52w", use_container_width=True):
                with st.spinner("Sampling one-year history…"):
                    result = breadth_52w(frame, limit=60)
                if result["highs"] is None:
                    st.warning("52-week breadth could not be computed: " + (result.get("note") or "unknown error"))
                else:
                    st.markdown(metric_html("Near 52-week high", str(result["highs"]),
                                            sub=f"near 52-week low {result['lows']} · {result['note']}"),
                                unsafe_allow_html=True)
    with sector_col:
        with st.container(border=True):
            st.markdown("<div class='card-header'>Sector aggregation</div>", unsafe_allow_html=True)
            table = sector_table(frame)
            if table.empty:
                st.info("No sector data available for this universe (the provider returned no sector field).")
            else:
                show = table.copy()
                show["avg_change"] = [fmt_pct(v) for v in table["avg_change"]]
                show["market_cap"] = [fmt_cap(cap, payload.get("currency", "USD")) for cap in table["market_cap"]]
                show.columns = ["Sector", "Names", "Avg % change", "Total market cap", "Advances", "Declines"]
                st.dataframe(show, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("<div class='card-header'>Inspect a row</div>", unsafe_allow_html=True)
        head = frame.head(200)
        labels = [f"{r['symbol']} — {r['name']}" for _, r in head.iterrows()]
        pick = st.selectbox("Instrument", labels, key="heat_jump")
        row = head.iloc[labels.index(pick)]
        jc1, jc2 = st.columns([3, 1])
        with jc1:
            st.markdown(metric_html(
                f"{row['symbol']} · {row['currency']}",
                f"{fmt_price(row['price'], row['currency'])} "
                f"<span style='font-size:0.9rem' class='{change_class(row['change_percent'])}'>"
                f"{fmt_pct(row['change_percent'])}</span>",
                sub=f"{row['sector'] or 'sector n/a'} · {row['industry'] or 'industry n/a'} · "
                    f"cap {fmt_cap(row['market_cap'], row['currency'])} · "
                    f"volume {fmt_volume(row['volume'])} · status {row['data_status']} · "
                    f"last bar {row['last_bar'] or '—'}"), unsafe_allow_html=True)
        with jc2:
            st.write("")
            if st.button("Open on dashboard", use_container_width=True, key="heat_jump_go"):
                state.selected_yf = row["yf_symbol"]
                state.selected_symbol = row["symbol"]
                state.page = "Dashboard"
                st.rerun()


# ===========================================================================
# SECTION 17 - PAGE: HISTORY
# ===========================================================================
def page_history() -> None:
    state = st.session_state
    yf_symbol = state.get("selected_yf", "RELIANCE.NS")
    snapshot = symbol_snapshot(yf_symbol)
    currency = snapshot.get("currency") or "USD"

    with st.container(border=True):
        page_header(f"Historical data — {snapshot.get('name') or yf_symbol}",
                    "Daily bars from the provider, in the instrument's own currency, with range, volume and "
                    "momentum read-outs.")

    history = _history_cached(yf_symbol, "3mo", "1d")
    if history is None or history.empty or len(history) < 2:
        st.warning("Not enough historical data (at least two daily bars are required). Try another instrument "
                   "or wait for the provider.")
        return
    if "Volume" not in history.columns:
        history = history.assign(Volume=0)

    def _day_values(row: Any) -> Dict[str, Any]:
        return {"open": valid_price(row.get("Open")), "high": valid_price(row.get("High")),
                "low": valid_price(row.get("Low")), "close": valid_price(row.get("Close")),
                "volume": _num(row.get("Volume")) or 0.0}

    d1, d2 = _day_values(history.iloc[-2]), _day_values(history.iloc[-1])
    d1_date = pd.Timestamp(history.index[-2]).strftime("%d %b %Y")
    d2_date = pd.Timestamp(history.index[-1]).strftime("%d %b %Y")

    with st.container(border=True):
        tail = history.tail(12)
        close_series = pd.to_numeric(tail["Close"], errors="coerce")
        open_series = pd.to_numeric(tail["Open"], errors="coerce")
        colours = ["#34d399" if (pd.notna(close_series.iloc[i]) and pd.notna(open_series.iloc[i])
                                and close_series.iloc[i] >= open_series.iloc[i]) else "#f87171"
                   for i in range(len(tail))]
        figure = go.Figure(go.Bar(x=[pd.Timestamp(i).strftime("%d %b") for i in tail.index],
                                  y=close_series, marker_color=colours,
                                  hovertemplate="%{x}<br>close %{y}<extra></extra>"))
        figure.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", height=240,
                             margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                             font=dict(color="#94a3b8"),
                             xaxis=dict(gridcolor="rgba(51,65,85,0.3)"),
                             yaxis=dict(gridcolor="rgba(51,65,85,0.3)", side="right"))
        st.plotly_chart(figure, use_container_width=True)

    col1, col2 = st.columns(2)
    for column, label, values, stamp in ((col1, "Previous session", d1, d1_date),
                                         (col2, "Latest session", d2, d2_date)):
        with column:
            with st.container(border=True):
                st.markdown(f"<div class='card-header'>{label} — {stamp}</div>", unsafe_allow_html=True)
                st.markdown(metric_html("Close", fmt_price(values["close"], currency),
                                        sub=f"open {fmt_price(values['open'], currency)} · "
                                            f"high {fmt_price(values['high'], currency)} · "
                                            f"low {fmt_price(values['low'], currency)} · "
                                            f"volume {fmt_volume(values['volume'])}"),
                            unsafe_allow_html=True)
                span = (values["high"] - values["low"]) if (values["high"] and values["low"]) else None
                if span and span > 0 and values["close"]:
                    position = (values["close"] - values["low"]) / span * 100
                    st.caption(f"Closed {position:.0f}% up its range "
                               f"({'near the high' if position >= 70 else 'near the low' if position <= 30 else 'mid-range'}).")

    with st.container(border=True):
        st.markdown("<div class='card-header'>Range, volume &amp; momentum</div>", unsafe_allow_html=True)
        if d1["high"] and d2["high"]:
            st.write(f"- Latest session **{'broke' if d2['high'] > d1['high'] else 'did not break'}** the "
                     f"previous high ({fmt_price(d1['high'], currency)}).")
        if d1["low"] and d2["low"]:
            st.write(f"- Latest session **{'broke' if d2['low'] < d1['low'] else 'did not break'}** the "
                     f"previous low ({fmt_price(d1['low'], currency)}).")
        volume_change = ((d2["volume"] - d1["volume"]) / d1["volume"] * 100) if d1["volume"] else None
        st.write(f"- Volume: {fmt_volume(d1['volume'])} → {fmt_volume(d2['volume'])}"
                 + (f" ({fmt_pct(volume_change)})" if volume_change is not None else ""))
        if volume_change is not None and d2["close"] and d2["open"]:
            if d2["close"] >= d2["open"] and volume_change > 0:
                st.write("- Rising volume on an up session — buyer participation expanded.")
            elif d2["close"] < d2["open"] and volume_change > 0:
                st.write("- Rising volume on a down session — selling pressure expanded.")
            else:
                st.write("- Volume did not expand with the move — conviction is mixed.")
        if d1["close"] and d2["close"]:
            st.write(f"- Net change between the two closes: "
                     f"**{fmt_pct((d2['close'] - d1['close']) / d1['close'] * 100)}**.")
        if d1["close"] and d2["open"]:
            gap_pct = (d2["open"] - d1["close"]) / d1["close"] * 100
            if abs(gap_pct) >= 0.3:
                st.write(f"- Gap between previous close and latest open: **{fmt_pct(gap_pct)}** "
                         f"({'gap up' if gap_pct > 0 else 'gap down'}).")
            else:
                st.write("- No significant opening gap between the two sessions.")
        closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if len(closes) >= 6:
            ma5 = closes.tail(5).mean()
            ma5_prev = closes.tail(6).head(5).mean()
            st.write(f"- 5-period moving average is **{'rising' if ma5 > ma5_prev else 'falling'}** "
                     f"({fmt_price(float(ma5), currency)}).")
        if len(closes) >= 9:
            st.write(f"- 9-period moving average: {fmt_price(float(closes.tail(9).mean()), currency)}.")
        swing_high = _num(pd.to_numeric(history["High"], errors="coerce").tail(10).max())
        swing_low = _num(pd.to_numeric(history["Low"], errors="coerce").tail(10).min())
        s1, s2 = st.columns(2)
        s1.markdown(f"<div class='level-row level-resistance'><b>Swing high (10d)</b> "
                    f"{fmt_price(swing_high, currency)}</div>", unsafe_allow_html=True)
        s2.markdown(f"<div class='level-row level-support'><b>Swing low (10d)</b> "
                    f"{fmt_price(swing_low, currency)}</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div class='card-header'>Recent bars</div>", unsafe_allow_html=True)
        table = history[["Open", "High", "Low", "Close", "Volume"]].tail(12).iloc[::-1].copy()
        table.index = [pd.Timestamp(i).strftime("%d %b %Y") for i in table.index]
        for column in ("Open", "High", "Low", "Close"):
            table[column] = [fmt_price(v, currency) for v in pd.to_numeric(table[column], errors="coerce")]
        table["Volume"] = [fmt_volume(v) for v in pd.to_numeric(table["Volume"], errors="coerce")]
        st.dataframe(table, use_container_width=True)
        st.caption(f"All prices shown in {currency_display(currency)} — the instrument's own currency, "
                   f"never silently converted.")


# ===========================================================================
# SECTION 18 - SIDEBAR: CHAT / WATCHLIST / ALERTS
# ===========================================================================
def render_chat_sidebar() -> None:
    state = st.session_state
    chat = init_chat_state()

    st.markdown("##### 🤖 AI Chat Assistant")
    st.caption("Talk about anything — markets, code, ideas, your day. Terminal data is available when it helps.")

    with st.expander("Chat settings", expanded=False):
        labels = model_labels()
        model_idx = labels.index(chat.model_label) if chat.model_label in labels else 0
        chat.model_label = st.selectbox("Model", labels, index=model_idx, key="chat_model_label")
        if chat.model_label == CUSTOM_MODEL_LABEL:
            chat.custom_provider = st.selectbox("Provider for the custom id", ["groq", "gemini"],
                                                index=0 if chat.custom_provider == "groq" else 1,
                                                key="chat_custom_provider")
            chat.custom_model_id = st.text_input("Custom model id", value=chat.custom_model_id,
                                                 placeholder="e.g. openai/gpt-oss-120b", key="chat_custom_id")
        persona_labels = [p["label"] for p in ASSISTANT_PERSONAS]
        persona_keys = [p["key"] for p in ASSISTANT_PERSONAS]
        current = chat.persona_key if chat.persona_key in persona_keys else "general"
        picked = st.selectbox("Persona", persona_labels, index=persona_keys.index(current), key="chat_persona")
        chat.persona_key = persona_keys[persona_labels.index(picked)]
        chat.use_tools = st.checkbox("Allow live data tools (market/news lookups)", value=chat.use_tools,
                                     key="chat_tools")
        chat.stream = st.checkbox("Stream the answer", value=chat.stream, key="chat_stream")
        diagnostics = ai_diagnostics()
        st.caption(f"Groq key: {'configured' if diagnostics['groq_key_configured'] else 'missing'} · "
                   f"Gemini key: {'configured' if diagnostics['gemini_key_configured'] else 'missing'} · "
                   f"groq pkg: {'yes' if diagnostics['groq_package'] else 'no'} · "
                   f"google-genai pkg: {'yes' if diagnostics['gemini_package'] else 'no'}")

    # ---- transcript (history survives every Streamlit rerun) ----
    transcript = [m for m in chat.messages if m.get("role") in ("user", "assistant") and m.get("content")]
    if not transcript:
        st.markdown("<div class='chat-assistant'>Hi — I'm your assistant for this terminal, but I'm not limited "
                    "to it. Ask me about a chart, a level, the news, or just tell me what's on your mind.</div>",
                    unsafe_allow_html=True)
    for message in transcript[-14:]:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-meta'>{message.get('time', '')}</div>"
                        f"<div class='chat-user'>{html.escape(message['content'])}</div>", unsafe_allow_html=True)
        else:
            tools_used = message.get("tools") or []
            meta = message.get("time", "")
            if tools_used:
                meta += " · tools: " + ", ".join(tools_used)
            st.markdown(f"<div class='chat-meta'>{meta}</div>", unsafe_allow_html=True)
            with st.container(border=False):
                st.markdown(message["content"])

    # ---- one-shot render of a pending turn (prevents duplicate answers on rerun) ----
    if chat.pending:
        prompt = chat.pending
        chat.pending = None
        placeholder = st.empty()
        answer = chat_generate(chat, placeholder, prompt)
        if answer:
            chat_append({"role": "assistant", "content": answer,
                         "time": datetime.now().strftime("%H:%M"), "tools": chat.last_used_tools})
        else:
            chat_append({"role": "assistant",
                         "content": "AI assistant temporarily unavailable. " + (chat.last_error or "Please try again."),
                         "time": datetime.now().strftime("%H:%M"), "tools": []})
        st.rerun()

    user_text = st.chat_input("Message your assistant…")
    if user_text is not None and str(user_text).strip():
        text = str(user_text).strip()
        if len(text) > 4000:
            text = text[:4000]
            st.caption("Message truncated to 4,000 characters.")
        chat_append({"role": "user", "content": text, "time": datetime.now().strftime("%H:%M")})
        chat.pending = text
        st.rerun()

    quick_a, quick_b = st.columns(2)
    with quick_a:
        if st.button("Support?", use_container_width=True, key="chat_quick_support"):
            chat_append({"role": "user", "content": "Where are the support levels on the selected instrument?",
                         "time": datetime.now().strftime("%H:%M")})
            chat.pending = "Where are the support levels on the selected instrument?"
            st.rerun()
    with quick_b:
        if st.button("Bullish?", use_container_width=True, key="chat_quick_bull"):
            chat_append({"role": "user", "content": "Is the selected instrument showing a bullish setup?",
                         "time": datetime.now().strftime("%H:%M")})
            chat.pending = "Is the selected instrument showing a bullish setup?"
            st.rerun()

    control_a, control_b = st.columns(2)
    with control_a:
        if st.button("🗑 Clear chat", use_container_width=True, key="chat_clear"):
            chat.messages = []
            chat.pending = None
            chat.last_error = ""
            st.rerun()
    with control_b:
        if st.button("✦ New chat", use_container_width=True, key="chat_new"):
            chat.messages = []
            chat.pending = None
            chat.last_error = ""
            chat.persona_key = "general"
            st.rerun()
    if chat.last_error:
        st.caption("Last error: " + chat.last_error)


def render_watchlist_sidebar() -> None:
    """UPGRADED (v3.1): watchlist persists to JSON via WatchlistStore, shows
    name + price + % change per row, and survives app restarts."""
    state = st.session_state
    st.markdown("##### 📋 Watchlist")
    new_ticker = st.text_input("Add instrument", placeholder="RELIANCE, AAPL, BTC-USD, ^NSEI",
                               key="watch_add_input")
    add_col, save_col = st.columns([2, 1])
    with add_col:
        if st.button("＋ Add", use_container_width=True, key="watch_add") and new_ticker.strip():
            symbol = resolve_yf_symbol(new_ticker, state.get("market_key", "india"),
                                       state.get("exchange_key", ""))
            if symbol and symbol not in state.watchlist:
                state.watchlist.append(symbol)
                _WATCHLIST_STORE.save(list(state.watchlist))
            st.rerun()
    with save_col:
        if st.button("💾 Save", use_container_width=True, key="watch_save"):
            _WATCHLIST_STORE.save(list(state.watchlist))
            st.toast("Watchlist saved.")
    if not state.watchlist:
        st.caption("Your watchlist is empty — add an instrument above.")
    for symbol in list(state.watchlist):
        snapshot = symbol_snapshot(symbol)
        currency = snapshot.get("currency") or "USD"
        left, right = st.columns([4, 1])
        with left:
            short = symbol.replace(".NS", "").replace(".BO", "").replace("^", "")
            name = str(snapshot.get("name") or "")[:14]
            label = (f"{short} · {name}  {fmt_price(snapshot.get('price'), currency)}  "
                     f"{fmt_pct(snapshot.get('change_percent'))}") if name else \
                    (f"{short}  {fmt_price(snapshot.get('price'), currency)}  "
                     f"{fmt_pct(snapshot.get('change_percent'))}")
            if st.button(label, key=f"watch_open_{symbol}", use_container_width=True):
                state.selected_yf = symbol
                state.selected_symbol = symbol.split(".")[0].replace("^", "")
                state.exchange_key = snapshot.get("exchange") or _exchange_for_yf_symbol(symbol)
                state.page = "Dashboard"
                st.rerun()
        with right:
            if st.button("🗑", key=f"watch_del_{symbol}"):
                state.watchlist = [s for s in state.watchlist if s != symbol]
                _WATCHLIST_STORE.save(list(state.watchlist))
                st.rerun()
    st.caption("Prices are shown in each instrument's own currency · 💾 saves the list to disk.")


def render_alerts_sidebar() -> None:
    """UPGRADED (v3.1): alerts use AlertEngine — de-duplication on create,
    60-minute Telegram cooldown, manual re-arm, and rules persist to JSON."""
    state = st.session_state
    st.markdown("##### 🔔 Price alerts")
    st.caption("Alerts are evaluated when this app reruns. A trigger sends ONE Telegram "
               "message, then waits 60 minutes before it may notify again (re-armable).")

    # hydrate persisted rules once per session
    if not state.get("_alerts_hydrated"):
        saved_watch, saved_alerts = _ALERT_ENGINE.load_state()
        if saved_alerts and not state.alerts:
            state.alerts = saved_alerts
        if saved_watch and state.watchlist == _defaults["watchlist"]:
            state.watchlist = saved_watch
        state._alerts_hydrated = True

    symbol_input = st.text_input("Symbol", value=state.get("selected_symbol", ""), key="alert_symbol")
    selected_snapshot = symbol_snapshot(state.get("selected_yf", ""))
    default_price = clamp_number_input(selected_snapshot.get("price"), minimum=0.0, fallback=0.0)
    target = st.number_input("Target price", min_value=0.0, value=default_price, format="%.4f", key="alert_price")
    direction = st.selectbox("Direction", ["Above", "Below"], key="alert_direction")
    if st.button("＋ Set alert", use_container_width=True, key="alert_set"):
        if target > 0:
            resolved = resolve_yf_symbol(symbol_input, state.get("market_key", "india"),
                                         state.get("exchange_key", ""))
            created, message = _ALERT_ENGINE.add(state.alerts, resolved, float(target), direction)
            if created:
                send_telegram(f"Alert set: {resolved} {direction} {target:g}",
                              token=secret("TELEGRAM_BOT_TOKEN"), chat_id=secret("TELEGRAM_CHAT_ID"))
                _ALERT_ENGINE.save_state(list(state.watchlist), list(state.alerts))
                st.success("Alert saved." + ("" if secret("TELEGRAM_BOT_TOKEN")
                                             else " (Telegram not configured — alert is in-app only.)"))
                st.rerun()
            else:
                st.warning(message)
        else:
            st.warning("Enter a target price above zero (the provider may be returning 0.0 for this instrument).")

    if not state.alerts:
        st.caption("No alerts configured.")
        return
    for index, alert in enumerate(list(state.alerts)):
        snapshot = symbol_snapshot(alert["symbol"])
        currency = snapshot.get("currency") or "USD"
        current = snapshot.get("price")
        triggered = _ALERT_ENGINE.is_triggered(alert, current)
        if triggered:
            st.warning(f"🔔 {alert['symbol']} {alert['direction']} "
                       f"{fmt_price(alert['price'], currency)} — now {fmt_price(current, currency)}")
            if _ALERT_ENGINE.should_notify(alert):
                sent = send_telegram(
                    f"🔔 <b>ALERT</b> {alert['symbol']}\n"
                    f"Now {fmt_price(current, currency)} ({fmt_pct(snapshot.get('change_percent'))})\n"
                    f"Trigger: {alert['direction']} {fmt_price(alert['price'], currency)}",
                    token=secret("TELEGRAM_BOT_TOKEN"), chat_id=secret("TELEGRAM_CHAT_ID"))
                if sent:
                    _ALERT_ENGINE.mark_notified(alert)
                    _ALERT_ENGINE.save_state(list(state.watchlist), list(state.alerts))
                    st.toast("Telegram notification sent.")
                else:
                    st.caption("Telegram send failed or is not configured — the alert stays armed.")
        else:
            armed = "⏳ cooling down" if alert.get("notified") else "armed"
            st.write(f"{alert['symbol']} {alert['direction']} {fmt_price(alert['price'], currency)} · "
                     f"now {fmt_price(current, currency)} · {armed}")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if alert.get("notified") and st.button("Re-arm", key=f"alert_rearm_{index}"):
                _ALERT_ENGINE.rearm(alert)
                _ALERT_ENGINE.save_state(list(state.watchlist), list(state.alerts))
                st.rerun()
        with btn_col2:
            if st.button("Remove", key=f"alert_remove_{index}"):
                state.alerts.pop(index)
                _ALERT_ENGINE.save_state(list(state.watchlist), list(state.alerts))
                st.rerun()


# ---------------------------------------------------------------------------
# UPGRADE (v3.1): Telegram + watchlist/alert persistence now live in the
# reusable module `watchlist_alerts.py`. The token is NEVER hardcoded here —
# it is read from st.secrets (Streamlit Cloud) or the environment / .env file.
# ---------------------------------------------------------------------------
from watchlist_alerts import AlertEngine, WatchlistStore, send_telegram  # noqa: E402

_ALERT_ENGINE = AlertEngine(state_path=SNAPSHOT_FILE.parent / "terminal_state.json",
                            cooldown_minutes=60)
_WATCHLIST_STORE = WatchlistStore(state_path=SNAPSHOT_FILE.parent / "terminal_state.json")


def telegram_status() -> str:
    """Human-readable Telegram configuration status for the diagnostics page."""
    token, chat_id = secret("TELEGRAM_BOT_TOKEN"), secret("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return "configured (secrets)"
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        return "configured (environment / .env)"
    return "NOT configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in " \
           ".streamlit/secrets.toml or a git-ignored .env file. NEVER in source code."


def render_sidebar() -> None:
    state = st.session_state
    tabs = st.radio("side", ["Chat", "Watch", "Alerts"], horizontal=True, label_visibility="collapsed",
                    index=["Chat", "Watch", "Alerts"].index(state.get("sidebar_tab", "Chat")),
                    key="sidebar_tabs")
    state.sidebar_tab = tabs
    if tabs == "Chat":
        render_chat_sidebar()
    elif tabs == "Watch":
        render_watchlist_sidebar()
    else:
        render_alerts_sidebar()


# ===========================================================================
# SECTION 19 - CROSS-MARKET OVERVIEW + DIAGNOSTICS
# ===========================================================================
def page_global(refresh: bool) -> None:
    with st.container(border=True):
        page_header("🌎 Cross-market overview",
                    "Only exchanges the configured provider can actually answer for are listed. "
                    "A dash means the provider returned nothing — never an invented value.")
        if refresh:
            try:
                _quotes_cached.clear()
            except Exception as exc:
                log_exception("cache clear", exc)
    rows: List[Dict[str, Any]] = []
    probes = [("india", "NSE", "^NSEI", "NIFTY 50"), ("india", "BSE", "^BSESN", "SENSEX"),
              ("us", "NASDAQ", "^IXIC", "NASDAQ Composite"), ("us", "NYSE", "^GSPC", "S&P 500"),
              ("uk", "LSE", "^FTSE", "FTSE 100"), ("germany", "XETRA", "^GDAXI", "DAX"),
              ("japan", "TSE", "^N225", "Nikkei 225"), ("hongkong", "HKEX", "^HSI", "Hang Seng"),
              ("australia", "ASX", "^AXJO", "S&P/ASX 200"), ("canada", "TSX", "^GSPTSE", "S&P/TSX")]
    with st.spinner("Sampling global indices…"):
        for market_key, exchange_key, symbol, label in probes:
            snapshot = symbol_snapshot(symbol)
            session = session_status(exchange_key)
            rows.append({
                "Market": MARKETS_BY_KEY.get(market_key).label if market_key in MARKETS_BY_KEY else market_key,
                "Index": label,
                "Symbol": symbol,
                "Level": fmt_price(snapshot.get("price"), snapshot.get("currency") or "USD"),
                "% change": fmt_pct(snapshot.get("change_percent")),
                "Session": session["status"],
                "Local time": session["local_time"],
                "Status": "ok" if snapshot.get("ok") else "no data",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Index levels are index points, not tradable prices. Session status uses each exchange's own "
               "timezone and weekday calendar; exchange holiday calendars are not bundled with this build.")


def page_diagnostics() -> None:
    with st.container(border=True):
        page_header("Diagnostics", "Everything this build knows about its own configuration and provider limits.")
        diagnostics = ai_diagnostics()
        st.markdown("##### AI providers")
        st.dataframe(pd.DataFrame([{"Item": k.replace("_", " "), "Value": str(v)}
                                   for k, v in diagnostics.items()]),
                     use_container_width=True, hide_index=True)
        st.markdown("##### Telegram notifications")
        st.write(telegram_status())
        st.markdown("##### Market data provider")
        st.dataframe(pd.DataFrame([{"Capability": k, "Detail": v} for k, v in PROVIDER.capabilities().items()]),
                     use_container_width=True, hide_index=True)
        st.markdown("##### Session calendar")
        st.dataframe(pd.DataFrame([{
            "Exchange": e.key, "Market": e.market, "Timezone": e.tz, "Currency": e.currency,
            "Session": f"{e.open_at}–{e.close_at} ({e.session_kind})",
            "Session status now": session_status(e.key)["status"],
        } for e in EXCHANGES.values()]), use_container_width=True, hide_index=True)
        st.markdown("##### Universes exposed")
        st.dataframe(pd.DataFrame([{"Universe": u.key, "Label": u.label, "Exchange": u.exchange,
                                    "Kind": u.kind, "Size": u.size,
                                    "Note": u.note or "—"} for u in UNIVERSES.values()]),
                     use_container_width=True, hide_index=True)
        st.markdown("##### Data status legend")
        for status in DATA_STATUS_ORDER:
            st.markdown(f"- {status_pill(status)} — {data_status_explanation(status)}", unsafe_allow_html=True)
        st.markdown("##### Snapshot cache")
        snapshots = st.session_state.get("_snapshots")
        if isinstance(snapshots, dict) and snapshots:
            st.dataframe(pd.DataFrame([{
                "Key": k, "Rows": len(v.get("rows", [])), "Status": v.get("data_status"),
                "Session": v.get("session_date"), "Saved": v.get("saved_at"),
            } for k, v in snapshots.items()]), use_container_width=True, hide_index=True)
        else:
            st.caption(f"No snapshot saved in this session yet. Snapshots persist to {SNAPSHOT_FILE}.")


# ===========================================================================
# SECTION 20 - BOOTSTRAP + ROUTER
# ===========================================================================
st.markdown(_CSS, unsafe_allow_html=True)

# --- session state defaults -------------------------------------------------
_defaults: Dict[str, Any] = {
    "page": "Dashboard",
    "market_key": "india",
    "exchange_key": "NSE",
    "universe_key": "NSE_TOP100",
    "market_cat": "india",
    "selected_symbol": "RELIANCE",
    "selected_yf": "RELIANCE.NS",
    "watchlist": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    "alerts": [],
    "sidebar_tab": "Chat",
    "news_open": None,
    "_snapshots": None,
}
for key, value in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value
if st.session_state.get("_snapshots") is None:
    st.session_state["_snapshots"] = _read_snapshot_file()
init_chat_state()

# --- header -----------------------------------------------------------------
header_left, header_right = st.columns([2, 5])
with header_left:
    st.markdown("### ⚡ **AI Trade Terminal**")
with header_right:
    pages = ["Dashboard", "Heatmap", "News", "History", "Global", "Diagnostics"]
    picked_page = st.radio("nav", pages, horizontal=True, label_visibility="collapsed",
                           index=pages.index(st.session_state.page) if st.session_state.page in pages else 0,
                           key="nav_radio")
    st.session_state.page = picked_page

with st.container(border=True):
    force_refresh = render_global_selector()

st.markdown("---")

main_col, side_col = st.columns([7, 3], gap="medium")

with main_col:
    try:
        if st.session_state.page == "Dashboard":
            page_dashboard(force_refresh)
        elif st.session_state.page == "Heatmap":
            page_heatmap(force_refresh)
        elif st.session_state.page == "News":
            page_news()
        elif st.session_state.page == "History":
            page_history()
        elif st.session_state.page == "Global":
            page_global(force_refresh)
        else:
            page_diagnostics()
    except Exception as exc:      # a page-level failure must never blank the terminal
        log_exception(f"page {st.session_state.page}", exc)
        st.error("This section hit an unexpected error and was contained. "
                 "The rest of the terminal is still usable.")
        with st.expander("Technical detail (secrets are redacted)"):
            st.code(safe_error(exc))

with side_col:
    with st.container(border=True):
        try:
            render_sidebar()
        except Exception as exc:
            log_exception("sidebar", exc)
            st.error("The sidebar hit an unexpected error and was contained.")
            with st.expander("Technical detail (secrets are redacted)"):
                st.code(safe_error(exc))

st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.75rem;margin-top:1.5rem'>"
    f"AI Trade Terminal v{APP_VERSION} · data: Yahoo Finance via yfinance (delayed unless stated) · "
    "AI: Groq + Google Gemini · when a market is closed the last completed session is shown and labelled "
    "LAST SESSION · educational tool, not investment advice.</div>",
    unsafe_allow_html=True)

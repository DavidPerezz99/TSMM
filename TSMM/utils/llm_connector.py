"""
LLM connector for open-source and hosted providers used by trading agents.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, Optional

import requests
import yaml


_PIPELINE_CACHE: Dict[str, Any] = {}


def _ollama_base_url(provider_cfg: Dict[str, Any]) -> str:
    return str(provider_cfg.get("base_url", "http://127.0.0.1:11434")).rstrip("/")


def _ollama_healthy(base_url: str, timeout_sec: int = 3) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout_sec)
        return r.status_code == 200
    except Exception:
        return False


def _candidate_ollama_executables(provider_cfg: Dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    configured = str(provider_cfg.get("executable_path", "")).strip()
    if configured:
        candidates.append(configured)

    discovered = shutil.which("ollama")
    if discovered:
        candidates.append(discovered)

    home = Path.home()
    common = [
        home / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
        home / "AppData" / "Local" / "Ollama" / "ollama.exe",
    ]
    candidates.extend(str(path) for path in common)

    ordered: list[str] = []
    seen = set()
    for candidate in candidates:
        normalized = str(candidate).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _start_ollama_server(provider_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_url = _ollama_base_url(provider_cfg)
    if _ollama_healthy(base_url):
        return {"ok": True, "already_running": True}

    executable = ""
    for candidate in _candidate_ollama_executables(provider_cfg):
        if os.path.exists(candidate):
            executable = candidate
            break
    if not executable:
        return {"ok": False, "error": "Ollama executable not found"}

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    try:
        subprocess.Popen(
            [executable, "serve"],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as exc:
        return {"ok": False, "error": f"Failed to start Ollama: {exc}"}

    startup_wait_seconds = max(int(provider_cfg.get("startup_wait_seconds", 20) or 20), 1)
    deadline = time.time() + float(startup_wait_seconds)
    while time.time() < deadline:
        if _ollama_healthy(base_url):
            return {"ok": True, "started": True, "executable": executable}
        time.sleep(1)

    return {"ok": False, "error": f"Ollama did not become healthy at {base_url}", "executable": executable}


def load_llm_providers_config(path: str = "config/llm_providers.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_secret(value: str) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value.startswith("env:"):
        return os.environ.get(value.split(":", 1)[1], "")
    return value


def _headers_with_auth(base: Optional[Dict[str, str]], token: str, scheme: str = "Bearer") -> Dict[str, str]:
    h = dict(base or {})
    if token:
        if scheme.lower() == "x-api-key":
            h["x-api-key"] = token
        else:
            h["Authorization"] = f"{scheme} {token}".strip()
    return h


def _safe_extract_text(payload: Any) -> str:
    if isinstance(payload, dict):
        if isinstance(payload.get("text"), str):
            return payload.get("text")
        if isinstance(payload.get("response"), str):
            return payload.get("response")
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                return content
            txt = choices[0].get("text") if isinstance(choices[0], dict) else None
            if isinstance(txt, str):
                return txt
        content = payload.get("content")
        if isinstance(content, list) and content:
            part = content[0]
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return part.get("text")
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            if isinstance(first.get("generated_text"), str):
                return first.get("generated_text")
    return ""


def _call_openai_compatible(provider_cfg: Dict[str, Any], prompt: str, timeout_sec: int) -> Dict[str, Any]:
    base_url = str(provider_cfg.get("base_url", "")).rstrip("/")
    model = str(provider_cfg.get("model", "")).strip()
    api_key = _resolve_secret(str(provider_cfg.get("api_key", "")))
    endpoint = provider_cfg.get("chat_endpoint", "/v1/chat/completions")

    url = f"{base_url}{endpoint}"
    headers = _headers_with_auth(
        base={"Content-Type": "application/json"},
        token=api_key,
        scheme=str(provider_cfg.get("auth_scheme", "Bearer")),
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": str(provider_cfg.get("system_prompt", "You are a trading assistant."))},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(provider_cfg.get("temperature", 0.2)),
    }
    max_tokens = provider_cfg.get("max_tokens")
    if isinstance(max_tokens, (int, float)) and int(max_tokens) > 0:
        payload["max_tokens"] = int(max_tokens)

    r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    return {
        "ok": r.status_code < 400,
        "status_code": r.status_code,
        "text": _safe_extract_text(data),
        "raw": data,
    }


def _call_anthropic(provider_cfg: Dict[str, Any], prompt: str, timeout_sec: int) -> Dict[str, Any]:
    base_url = str(provider_cfg.get("base_url", "https://api.anthropic.com")).rstrip("/")
    api_key = _resolve_secret(str(provider_cfg.get("api_key", "")))
    model = str(provider_cfg.get("model", "claude-3-5-sonnet-latest"))
    url = f"{base_url}/v1/messages"

    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": str(provider_cfg.get("anthropic_version", "2023-06-01")),
    }

    payload = {
        "model": model,
        "max_tokens": int(provider_cfg.get("max_tokens", 600)),
        "temperature": float(provider_cfg.get("temperature", 0.2)),
        "messages": [{"role": "user", "content": prompt}],
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    return {
        "ok": r.status_code < 400,
        "status_code": r.status_code,
        "text": _safe_extract_text(data),
        "raw": data,
    }


def _call_huggingface(provider_cfg: Dict[str, Any], prompt: str, timeout_sec: int) -> Dict[str, Any]:
    api_key = _resolve_secret(str(provider_cfg.get("api_key", "")))
    endpoint = str(provider_cfg.get("inference_endpoint", "")).strip()
    if not endpoint:
        model = str(provider_cfg.get("model", "")).strip()
        endpoint = f"https://api-inference.huggingface.co/models/{model}"

    headers = _headers_with_auth(
        base={"Content-Type": "application/json"},
        token=api_key,
        scheme="Bearer",
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": int(provider_cfg.get("max_new_tokens", 400)),
            "temperature": float(provider_cfg.get("temperature", 0.2)),
        },
    }

    r = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    return {
        "ok": r.status_code < 400,
        "status_code": r.status_code,
        "text": _safe_extract_text(data),
        "raw": data,
    }


def _call_ollama(provider_cfg: Dict[str, Any], prompt: str, timeout_sec: int) -> Dict[str, Any]:
    base_url = _ollama_base_url(provider_cfg)
    model = str(provider_cfg.get("model", "llama3.1:8b"))
    ensure = _start_ollama_server(provider_cfg)
    if not bool(ensure.get("ok", False)):
        return {
            "ok": False,
            "status_code": 503,
            "text": "",
            "raw": ensure,
        }
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(provider_cfg.get("temperature", 0.2)),
        },
    }
    r = requests.post(url, json=payload, timeout=timeout_sec)
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    return {
        "ok": r.status_code < 400,
        "status_code": r.status_code,
        "text": _safe_extract_text(data),
        "raw": data,
    }


def _call_local_transformers(provider_cfg: Dict[str, Any], prompt: str, timeout_sec: int) -> Dict[str, Any]:
    model = str(provider_cfg.get("model", "google/flan-t5-small")).strip()
    task = str(provider_cfg.get("task", "text2text-generation")).strip() or "text2text-generation"
    temperature = float(provider_cfg.get("temperature", 0.2))
    max_new_tokens = int(provider_cfg.get("max_new_tokens", 256) or 256)

    cache_key = f"{task}::{model}"
    pipe = _PIPELINE_CACHE.get(cache_key)
    if pipe is None:
        from transformers import pipeline  # type: ignore

        pipe = pipeline(task=task, model=model, device=-1)
        _PIPELINE_CACHE[cache_key] = pipe

    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
    }

    # Avoid invalid generation arg combinations for deterministic mode.
    if temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
    else:
        kwargs["do_sample"] = False

    out = pipe(prompt, **kwargs)
    text = _safe_extract_text(out)
    if not text and isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            text = str(first.get("generated_text") or first.get("summary_text") or "")

    return {
        "ok": True,
        "status_code": 200,
        "text": text,
        "raw": out,
    }


def _call_github_models(provider_cfg: Dict[str, Any], prompt: str, timeout_sec: int) -> Dict[str, Any]:
    # GitHub Models is OpenAI-compatible over Azure inference endpoint.
    cfg = dict(provider_cfg or {})
    cfg.setdefault("base_url", "https://models.inference.ai.azure.com")
    cfg.setdefault("chat_endpoint", "/chat/completions")
    cfg.setdefault("auth_scheme", "Bearer")
    return _call_openai_compatible(cfg, prompt, timeout_sec)


def call_llm(provider_name: str, prompt: str, providers_cfg: Dict[str, Any], timeout_sec: int = 30) -> Dict[str, Any]:
    providers = (providers_cfg.get("providers") or {})
    cfg = (providers.get(provider_name) or {})
    if not cfg:
        return {"ok": False, "error": f"Provider not found: {provider_name}"}

    if not bool(cfg.get("enabled", False)):
        return {"ok": False, "error": f"Provider disabled: {provider_name}"}

    ptype = str(cfg.get("type", "openai_compatible")).strip().lower()
    try:
        if ptype == "openai_compatible":
            out = _call_openai_compatible(cfg, prompt, timeout_sec)
        elif ptype == "anthropic":
            out = _call_anthropic(cfg, prompt, timeout_sec)
        elif ptype == "huggingface":
            out = _call_huggingface(cfg, prompt, timeout_sec)
        elif ptype == "ollama":
            out = _call_ollama(cfg, prompt, timeout_sec)
        elif ptype == "local_transformers":
            out = _call_local_transformers(cfg, prompt, timeout_sec)
        elif ptype == "github_models":
            out = _call_github_models(cfg, prompt, timeout_sec)
        else:
            return {"ok": False, "error": f"Unsupported provider type: {ptype}"}

        out["provider"] = provider_name
        out["provider_type"] = ptype
        return out
    except Exception as e:
        return {"ok": False, "provider": provider_name, "provider_type": ptype, "error": str(e)}

"""Always-on Telegram command listener for TSMM backend control.

Supported commands (prefix configurable, default /tsmm):
- /tsmm help|commands|?
- /tsmm status
- /tsmm deploy [--refresh|--no-refresh] [--dry-run] [--no-start-job]
- /tsmm deploy stop
- /tsmm trading start [--plan-model MODEL]
- /tsmm trading resume
- /tsmm trading status
- /tsmm trading stop
- /tsmm endpoint restart
- /tsmm ui start|stop
- /tsmm resource status|relieve

Natural-language chat is also supported (for example: "start trading", "please stop deploy", "how is trading doing?").
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import psutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.notification_telegram import _resolve_secret, send_telegram_notification  # noqa: E402
from utils.resource_guard import check_and_relieve, read_status as read_resource_status  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _listener_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(trading_cfg.get("telegram_listener") or {})


def _tg_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(trading_cfg.get("telegram_notifications") or {})


def _token_and_chat_ids(trading_cfg: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    tcfg = _tg_cfg(trading_cfg)
    lcfg = _listener_cfg(trading_cfg)

    token = _resolve_secret(str(tcfg.get("bot_token") or "")).strip()
    chat_id_default = _resolve_secret(str(tcfg.get("chat_id") or "")).strip()

    allowed = [str(c).strip() for c in (lcfg.get("allowed_chat_ids") or []) if str(c).strip()]
    if chat_id_default and chat_id_default not in allowed:
        allowed.append(chat_id_default)

    return token, chat_id_default, allowed


def _api_get(token: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = f"https://api.telegram.org/bot{token}/{method}"
    r = requests.get(base, params=params, timeout=30)
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"ok": False, "raw": r.text}


def _run_cmd(args: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "args": list(args),
        "stdout": (proc.stdout or "")[-3000:],
        "stderr": (proc.stderr or "")[-3000:],
    }


def _run_cmd_async(args: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]

    p = subprocess.Popen(
        args,
        cwd=str(ROOT),
        env=env,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "ok": True,
        "returncode": 0,
        "args": list(args),
        "pid": int(p.pid),
        "stdout": "",
        "stderr": "",
    }


def _audit_path() -> Path:
    p = ROOT / "reports" / "runtime" / "telegram_command_audit.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _conversation_path() -> Path:
    p = ROOT / "reports" / "runtime" / "telegram_conversation_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_audit(entry: Dict[str, Any]) -> None:
    with _audit_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _write_conversation(entry: Dict[str, Any]) -> None:
    with _conversation_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _console_trace(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _allowed_roots(trading_cfg: Dict[str, Any]) -> set[str]:
    lcfg = _listener_cfg(trading_cfg)
    roots = [str(x).strip().lower() for x in (lcfg.get("allowed_commands") or ["status", "deploy", "trading", "endpoint", "ui", "resource"]) if str(x).strip()]
    return set(roots)


def _help_message(prefix: str) -> str:
    p = str(prefix or "/tsmm").strip() or "/tsmm"
    return (
        "TSMM Bot Commands\n"
        f"- {p} help: Show this command list.\n"
        f"- {p} status: Show trading state, endpoint status, and active LLM provider.\n"
        f"- {p} deploy [--refresh|--no-refresh|--dry-run|--no-start-job]: Run deployment pipeline.\n"
        f"- {p} deploy stop: Request an active deployment pipeline to stop safely.\n"
        f"- {p} trading start [--plan-model MODEL]: Start trading job.\n"
        f"- {p} trading resume: Resume Agent B trading loop.\n"
        f"- {p} trading status: Show latest trading job status and decision.\n"
        f"- {p} trading stop: Stop trading job.\n"
        f"- {p} endpoint restart: Restart local signal endpoint service.\n"
        f"- {p} ui start: Start UI apps.\n"
        f"- {p} ui stop: Stop UI apps.\n"
        f"- {p} resource status: Show CPU/RAM guard status.\n"
        f"- {p} resource relieve: Run immediate resource-relief action.\n"
        "Latest async request status is posted automatically every 2 minutes and once when completed.\n"
        "You can also send natural language requests (for example: 'start trading', 'deploy with refresh', 'show trading status')."
    )


def _contains_any(text: str, phrases: List[str]) -> bool:
    t = str(text or "").strip().lower()
    return any(p in t for p in phrases)


def _infer_natural_language_tail(text: str) -> Tuple[str | None, str | None]:
    t = str(text or "").strip().lower()
    if not t:
        return None, None

    if _contains_any(t, ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        return None, "I am online and ready. Tell me what you want in plain language, or type '/tsmm help'."

    if _contains_any(t, ["help", "what can you do", "commands", "how do i", "usage", "manual"]):
        return "help", None

    if _contains_any(t, ["stop deploy", "cancel deploy", "abort deploy", "halt deploy"]):
        return "deploy stop", None

    if _contains_any(t, ["deploy"]):
        flags: List[str] = []
        if _contains_any(t, ["dry run", "dry-run", "simulate"]):
            flags.append("--dry-run")
        if _contains_any(t, ["no refresh", "without refresh", "skip refresh"]):
            flags.append("--no-refresh")
        elif _contains_any(t, ["refresh", "force refresh", "update data"]):
            flags.append("--refresh")
        if _contains_any(t, ["no start", "don't start", "dont start", "without trading", "no-start-job"]):
            flags.append("--no-start-job")
        tail = "deploy" + (" " + " ".join(flags) if flags else "")
        return tail, None

    if _contains_any(t, ["trading status", "status trading", "how is trading", "trading doing", "trading progress"]):
        return "trading status", None

    if _contains_any(t, ["start trading", "begin trading", "run trading", "launch trading"]):
        return "trading start", None

    if _contains_any(t, ["resume trading", "continue trading", "restart trading loop"]):
        return "trading resume", None

    if _contains_any(t, ["stop trading", "halt trading", "end trading"]):
        return "trading stop", None

    if _contains_any(t, ["restart endpoint", "endpoint restart", "reset endpoint", "reboot endpoint"]):
        return "endpoint restart", None

    if _contains_any(t, ["start ui", "open ui", "launch ui", "start dashboard"]):
        return "ui start", None

    if _contains_any(t, ["stop ui", "close ui", "shutdown ui", "hide ui"]):
        return "ui stop", None

    if _contains_any(t, ["resource status", "cpu", "ram", "memory status", "resource usage"]):
        return "resource status", None

    if _contains_any(t, ["resource relieve", "relieve resources", "free resources", "relieve pressure"]):
        return "resource relieve", None

    if _contains_any(t, ["status", "health", "are you online", "system status"]):
        return "status", None

    return None, (
        "I understood that as chat, but not as an action yet. "
        "Try saying things like 'start trading', 'deploy with refresh', 'show trading status', or '/tsmm help'."
    )


def _is_pid_alive(pid: int) -> bool:
    try:
        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        return False


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_last_jsonl(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return {}
        return json.loads(lines[-1])
    except Exception:
        return {}


def _latest_request_status_message(request_info: Dict[str, Any]) -> str:
    req_type = str(request_info.get("type") or "request")
    pid = int(request_info.get("pid") or 0)
    alive = _is_pid_alive(pid)
    base = f"request_status: type={req_type}; pid={pid}; running={'yes' if alive else 'no'}"

    if req_type == "deploy":
        stage_tail = _read_last_jsonl(ROOT / "reports" / "runtime" / "deployment_pipeline_stage_log.jsonl")
        stage = str(stage_tail.get("stage") or "n/a")
        stage_status = str(stage_tail.get("status") or "n/a")
        return f"{base}; stage={stage}; stage_status={stage_status}"

    if req_type in {"trading start", "trading resume"}:
        state = _read_json(ROOT / "reports" / "runtime" / "trading_job_state.json")
        plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
        return (
            f"{base}; status={state.get('status', 'unknown')}; stage={state.get('stage', 'unknown')}; "
            f"decision={plan.get('decision', 'n/a')}; closed_reason={state.get('closed_reason', 'n/a')}"
        )

    return base


def _latest_request_done_message(request_info: Dict[str, Any]) -> str:
    req_type = str(request_info.get("type") or "request")
    pid = int(request_info.get("pid") or 0)
    base = f"request_done: type={req_type}; pid={pid}; running=no"

    if req_type == "deploy":
        summary = _read_json(ROOT / "reports" / "runtime" / "deployment_pipeline_last.json")
        return (
            f"{base}; llm_provider={((summary.get('llm') or {}).get('chosen_provider', 'n/a'))}; "
            f"endpoint_ok={((summary.get('endpoint') or {}).get('ok', 'n/a'))}; "
            f"trading_started={((summary.get('trading') or {}).get('started', False))}"
        )

    if req_type in {"trading start", "trading resume"}:
        state = _read_json(ROOT / "reports" / "runtime" / "trading_job_state.json")
        plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
        return (
            f"{base}; status={state.get('status', 'unknown')}; stage={state.get('stage', 'unknown')}; "
            f"decision={plan.get('decision', 'n/a')}; closed_reason={state.get('closed_reason', 'n/a')}"
        )

    return base


def _restart_endpoint_service(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    lcfg = _listener_cfg(trading_cfg)
    p_cfg = _load_yaml(ROOT / str(lcfg.get("default_pipeline_config") or "config/agent_pipeline.yaml"))
    ep_cfg = (p_cfg.get("endpoints") or {})

    script = str(ep_cfg.get("service_script", "scripts/local_signal_endpoint_service.py"))
    host = str(ep_cfg.get("host", "127.0.0.1"))
    port = int(ep_cfg.get("port", 8000) or 8000)

    env = os.environ.copy()
    env["TSMM_SIGNAL_HOST"] = host
    env["TSMM_SIGNAL_PORT"] = str(port)

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]

    p = subprocess.Popen(
        [sys.executable, script],
        cwd=str(ROOT),
        env=env,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    health_url = f"http://{host}:{port}/health"
    ok = False
    payload: Dict[str, Any] = {}
    for _ in range(20):
        time.sleep(1)
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                payload = r.json()
                ok = True
                break
        except Exception:
            pass

    return {
        "ok": ok,
        "pid": int(p.pid),
        "health_url": health_url,
        "health": payload,
    }


def _handle_command(text: str, trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    lcfg = _listener_cfg(trading_cfg)
    prefix = str(lcfg.get("command_prefix", "/tsmm")).strip() or "/tsmm"
    body = text.strip()
    if not body.lower().startswith(prefix.lower()):
        if bool(lcfg.get("allow_natural_language", True)):
            tail, chat_msg = _infer_natural_language_tail(body)
            if tail:
                routed = _handle_command(f"{prefix} {tail}", trading_cfg)
                if routed.get("handled", False):
                    routed["message"] = (
                        f"I interpreted your request as: '{tail}'.\n"
                        f"{str(routed.get('message') or '')}"
                    )
                    routed["parsed_from_natural_language"] = True
                return routed
            if chat_msg:
                return {
                    "handled": True,
                    "ok": True,
                    "message": chat_msg,
                    "parsed_from_natural_language": True,
                }

        if bool(lcfg.get("reply_on_non_command", True)):
            return {
                "handled": True,
                "ok": True,
                "message": (
                    f"I am online. You can use commands starting with {prefix}, "
                    "or plain language like 'show trading status' or 'start trading'."
                ),
            }
        return {"handled": False, "reason": "wrong_prefix"}

    tail = body[len(prefix) :].strip()
    if not tail:
        return {
            "handled": True,
            "ok": True,
            "message": _help_message(prefix),
        }

    parts = tail.split()
    cmd = parts[0].lower()
    rest = parts[1:]

    if cmd in {"help", "commands", "?"}:
        return {
            "handled": True,
            "ok": True,
            "message": _help_message(prefix),
        }

    if cmd not in _allowed_roots(trading_cfg):
        return {"handled": True, "ok": False, "message": f"command_not_allowed:{cmd}"}

    env = os.environ.copy()

    if cmd == "status":
        state_path = ROOT / "reports" / "runtime" / "trading_job_state.json"
        endpoint_path = ROOT / "reports" / "runtime" / "local_signal_endpoint_service.pid"
        last_summary = ROOT / "reports" / "runtime" / "deployment_pipeline_last.json"

        state = _load_yaml(state_path) if state_path.suffix in {".yaml", ".yml"} else None
        if state is None and state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {}

        summary = {}
        if last_summary.exists():
            try:
                summary = json.loads(last_summary.read_text(encoding="utf-8"))
            except Exception:
                summary = {}

        return {
            "handled": True,
            "ok": True,
            "message": (
                f"status: trading_state={str((state or {}).get('status', 'unknown'))}; "
                f"endpoint_pid_file={'yes' if endpoint_path.exists() else 'no'}; "
                f"llm_provider={str((summary.get('llm') or {}).get('chosen_provider', 'n/a'))}"
            ),
        }

    if cmd == "deploy":
        script = str(lcfg.get("run_deploy_script", "scripts/deploy_agent_pipeline.py"))
        cfg = str(lcfg.get("default_pipeline_config", "config/agent_pipeline.yaml"))
        if rest and rest[0].lower() == "stop":
            args = [sys.executable, script, "--stop"]
            out = _run_cmd(args, env)
            return {
                "handled": True,
                "ok": out.get("ok", False),
                "message": f"deploy stop rc={out.get('returncode')}",
                "exec_args": out.get("args"),
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "returncode": out.get("returncode"),
            }

        args = [sys.executable, script, "--pipeline-config", cfg]

        allowed = {"--refresh", "--no-refresh", "--dry-run", "--no-start-job"}
        for tok in rest:
            if tok in allowed:
                args.append(tok)

        # Deploy can run for several minutes; run detached to keep listener responsive.
        out = _run_cmd_async(args, env)
        return {
            "handled": True,
            "ok": out.get("ok", False),
            "message": f"deploy started pid={out.get('pid')}",
            "track_request": {
                "type": "deploy",
                "pid": out.get("pid"),
            },
            "exec_args": out.get("args"),
            "stdout": out.get("stdout", ""),
            "stderr": out.get("stderr", ""),
            "returncode": out.get("returncode"),
        }

    if cmd == "trading":
        action = rest[0].lower() if rest else ""
        if action == "status":
            state_path = ROOT / "reports" / "runtime" / "trading_job_state.json"
            state = {}
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    state = {}

            plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
            return {
                "handled": True,
                "ok": True,
                "message": (
                    "trading_status: "
                    f"status={state.get('status', 'unknown')}; "
                    f"stage={state.get('stage', 'unknown')}; "
                    f"decision={plan.get('decision', 'n/a')}; "
                    f"model={plan.get('model', 'n/a')}; "
                    f"confidence={plan.get('confidence', 'n/a')}; "
                    f"closed_reason={state.get('closed_reason', 'n/a')}"
                ),
            }

        if action not in {"start", "resume", "stop"}:
            return {"handled": True, "ok": False, "message": "usage: trading start|resume|stop|status"}

        args = [sys.executable, "app.py", "trading-job", action]
        if action == "start" and len(rest) >= 3 and rest[1] == "--plan-model":
            args.extend(["--plan-model", rest[2]])

        # trading start/resume can block; launch detached so listener keeps polling.
        if action in {"start", "resume"}:
            out = _run_cmd_async(args, env)
            return {
                "handled": True,
                "ok": out.get("ok", False),
                "message": f"trading {action} started pid={out.get('pid')}",
                "track_request": {
                    "type": f"trading {action}",
                    "pid": out.get("pid"),
                },
                "exec_args": out.get("args"),
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "returncode": out.get("returncode"),
            }

        out = _run_cmd(args, env)
        return {
            "handled": True,
            "ok": out.get("ok", False),
            "message": f"trading {action} rc={out.get('returncode')}",
            "exec_args": out.get("args"),
            "stdout": out.get("stdout", ""),
            "stderr": out.get("stderr", ""),
            "returncode": out.get("returncode"),
        }

    if cmd == "endpoint" and rest and rest[0].lower() == "restart":
        out = _restart_endpoint_service(trading_cfg)
        return {
            "handled": True,
            "ok": out.get("ok", False),
            "message": f"endpoint restart ok={out.get('ok')} pid={out.get('pid')}",
        }

    if cmd == "ui":
        action = rest[0].lower() if rest else ""
        if action == "start":
            out = _run_cmd([sys.executable, "scripts/start_all_uis.py"], env)
            return {
                "handled": True,
                "ok": out.get("ok", False),
                "message": f"ui start rc={out.get('returncode')}",
                "exec_args": out.get("args"),
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "returncode": out.get("returncode"),
            }
        if action == "stop":
            out = _run_cmd([sys.executable, "scripts/stop_all_uis.py"], env)
            return {
                "handled": True,
                "ok": out.get("ok", False),
                "message": f"ui stop rc={out.get('returncode')}",
                "exec_args": out.get("args"),
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "returncode": out.get("returncode"),
            }
        return {"handled": True, "ok": False, "message": "usage: ui start|stop"}

    if cmd == "resource":
        action = rest[0].lower() if rest else "status"
        if action == "status":
            st = read_resource_status(ROOT)
            return {
                "handled": True,
                "ok": True,
                "message": (
                    f"resource_status: cpu={st.get('cpu'):.1f}% ram={st.get('ram'):.1f}% "
                    f"breach_since={st.get('breach_since')} last_relieved_at={st.get('last_relieved_at')}"
                ),
            }
        if action == "relieve":
            out = check_and_relieve(ROOT, trading_cfg)
            return {
                "handled": True,
                "ok": True,
                "message": f"resource_relief: {json.dumps(out, default=str)}",
            }
        return {"handled": True, "ok": False, "message": "usage: resource status|relieve"}

    return {"handled": True, "ok": False, "message": "unknown command"}


def run_listener(trading_config_path: Path) -> int:
    trading_cfg = _load_yaml(trading_config_path)
    lcfg = _listener_cfg(trading_cfg)
    if not bool(lcfg.get("enabled", False)):
        print("telegram listener disabled in config")
        return 1

    token, default_chat_id, allowed_chat_ids = _token_and_chat_ids(trading_cfg)
    if not token:
        print("telegram bot token missing")
        return 2

    poll_seconds = max(int(lcfg.get("poll_seconds", 3) or 3), 1)
    progress_interval_sec = max(int(lcfg.get("latest_request_status_interval_seconds", 120) or 120), 30)
    offset = 0
    latest_request: Dict[str, Any] = {}

    print(f"telegram listener started: cfg={trading_config_path} prefix={lcfg.get('command_prefix', '/tsmm')}")

    while True:
        try:
            res = _api_get(token, "getUpdates", {"timeout": 25, "offset": offset})
            updates = res.get("result") or []
            for upd in updates:
                try:
                    uid = int(upd.get("update_id", 0))
                    offset = max(offset, uid + 1)
                    msg = upd.get("message") or {}
                    chat = msg.get("chat") or {}
                    chat_id = str(chat.get("id", "")).strip()
                    text = str(msg.get("text") or "").strip()
                    if not chat_id or not text:
                        continue
                    if allowed_chat_ids and chat_id not in allowed_chat_ids:
                        continue

                    if bool(_listener_cfg(trading_cfg).get("log_conversations", True)):
                        _write_conversation(
                            {
                                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "direction": "inbound",
                                "chat_id": chat_id,
                                "text": text,
                            }
                        )

                    # Reload cfg each command so changes are applied without restart.
                    trading_cfg = _load_yaml(trading_config_path)
                    lcfg = _listener_cfg(trading_cfg)
                    raw_text = text
                    _console_trace(f"inbound chat_id={chat_id} text={raw_text}")
                    if bool(lcfg.get("require_secret", False)):
                        secret_env = str(lcfg.get("secret_env", "TSMM_TELEGRAM_COMMAND_SECRET")).strip() or "TSMM_TELEGRAM_COMMAND_SECRET"
                        secret = os.environ.get(secret_env, "").strip()
                        marker = f"#{secret}" if secret else ""
                        if not secret or marker not in raw_text:
                            out = {"handled": True, "ok": False, "message": "authentication_failed"}
                        else:
                            text = raw_text.replace(marker, "").strip()
                            out = _handle_command(text, trading_cfg)
                    else:
                        out = _handle_command(text, trading_cfg)
                    if not out.get("handled", False):
                        _console_trace("command ignored: not handled")
                        continue

                    if out.get("exec_args"):
                        _console_trace(f"exec args={out.get('exec_args')}")
                    if out.get("returncode") is not None:
                        _console_trace(f"exec returncode={out.get('returncode')}")
                    _console_trace(f"command result ok={bool(out.get('ok', False))} message={out.get('message')}")

                    tr = out.get("track_request") if isinstance(out, dict) else None
                    if isinstance(tr, dict) and tr.get("pid"):
                        latest_request = {
                            "chat_id": chat_id,
                            "type": str(tr.get("type") or "request"),
                            "pid": int(tr.get("pid") or 0),
                            "next_status_at": time.time() + float(progress_interval_sec),
                            "done_notified": False,
                        }

                    _write_audit(
                        {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "chat_id": chat_id,
                            "text": raw_text,
                            "parsed_text": text,
                            "ok": bool(out.get("ok", False)),
                            "message": out.get("message"),
                            "exec_args": out.get("exec_args"),
                            "returncode": out.get("returncode"),
                            "stdout": out.get("stdout"),
                            "stderr": out.get("stderr"),
                        }
                    )

                    body = str(out.get("message") or "")
                    if out.get("stdout"):
                        body += "\nstdout:\n" + str(out.get("stdout"))
                    if out.get("stderr"):
                        body += "\nstderr:\n" + str(out.get("stderr"))

                    tcfg = _tg_cfg(trading_cfg)
                    tcfg = dict(tcfg)
                    tcfg["chat_id"] = chat_id or default_chat_id
                    send_res = send_telegram_notification(tcfg, body[:3500])
                    _console_trace(
                        "telegram send "
                        f"ok={bool(send_res.get('ok', False))} "
                        f"status_code={send_res.get('status_code')} "
                        f"message_id={send_res.get('message_id')}"
                    )

                    _write_audit(
                        {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "chat_id": chat_id,
                            "event": "telegram_send_result",
                            "ok": bool(send_res.get("ok", False)),
                            "status_code": send_res.get("status_code"),
                            "error": send_res.get("error"),
                            "message_id": send_res.get("message_id"),
                        }
                    )

                    if bool(_listener_cfg(trading_cfg).get("log_conversations", True)):
                        _write_conversation(
                            {
                                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "direction": "outbound",
                                "chat_id": chat_id,
                                "text": body[:3500],
                            }
                        )
                except Exception:
                    continue

            # Resource guard runs continuously while listener is live.
            try:
                check_and_relieve(ROOT, trading_cfg)
            except Exception:
                pass

            if latest_request and latest_request.get("chat_id"):
                req_pid = int(latest_request.get("pid") or 0)
                req_chat_id = str(latest_request.get("chat_id") or "")
                req_done_notified = bool(latest_request.get("done_notified", False))
                req_alive = _is_pid_alive(req_pid)

                if req_alive and time.time() >= float(latest_request.get("next_status_at") or 0):
                    status_msg = _latest_request_status_message(latest_request)
                    tcfg = dict(_tg_cfg(trading_cfg))
                    tcfg["chat_id"] = req_chat_id
                    send_telegram_notification(tcfg, status_msg[:3500])
                    latest_request["next_status_at"] = time.time() + float(progress_interval_sec)

                if (not req_alive) and (not req_done_notified):
                    done_msg = _latest_request_done_message(latest_request)
                    tcfg = dict(_tg_cfg(trading_cfg))
                    tcfg["chat_id"] = req_chat_id
                    send_telegram_notification(tcfg, done_msg[:3500])
                    latest_request["done_notified"] = True

            time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print("telegram listener interrupted")
            return 0
        except Exception:
            time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="TSMM Telegram command listener")
    parser.add_argument("--trading-config", default="config/trading_agent.yaml", help="Path to trading config")
    args = parser.parse_args()

    return run_listener(ROOT / args.trading_config)


if __name__ == "__main__":
    raise SystemExit(main())

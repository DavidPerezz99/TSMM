"""Email notification utility for trading-job approvals and emergencies."""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict


def _resolve_secret(value: str) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value.startswith("env:"):
        return os.environ.get(value.split(":", 1)[1], "")
    return value


def send_email_notification(email_cfg: Dict[str, Any], subject: str, body: str) -> Dict[str, Any]:
    cfg = email_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return {"ok": False, "skipped": True, "reason": "email disabled"}

    smtp_host = _resolve_secret(str(cfg.get("smtp_host", "smtp.gmail.com"))).strip()
    smtp_port = int(cfg.get("smtp_port", 587) or 587)
    use_tls = bool(cfg.get("use_tls", True))

    sender = _resolve_secret(str(cfg.get("sender_email", ""))).strip()
    receiver = _resolve_secret(str(cfg.get("receiver_email", ""))).strip()
    username = _resolve_secret(str(cfg.get("username", sender))).strip()
    password = _resolve_secret(str(cfg.get("password", "")))

    if not sender or not receiver or not username or not password:
        return {"ok": False, "error": "missing_email_credentials_or_addresses"}

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            if use_tls:
                server.starttls()
            server.login(username, password)
            server.sendmail(sender, [receiver], msg.as_string())
        return {"ok": True, "smtp_host": smtp_host, "receiver": receiver}
    except Exception as e:
        return {"ok": False, "error": str(e), "smtp_host": smtp_host, "receiver": receiver}

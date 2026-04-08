"""Local case server for SupportBench evaluation.

Serves case pages at /case/<case_id> matching the exact supportbot.info style.
Includes problem, solution, tags, and evidence messages with media.

Usage:
    from eval_case_server import start_case_server, stop_case_server

    server = start_case_server(cases, dataset_msgs, dataset_dir, port=8099)
    # ... run eval ...
    stop_case_server(server)
"""
import mimetypes
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

_cases: list[dict] = []
_msgs_by_id: dict[str, dict] = {}
_dataset_dir: Path | None = None
_static_dir: Path = Path(__file__).resolve().parent.parent / "signal-web" / "public"

# ── CSS: exact copy from signal-web/pages/case/[id].tsx ──────────────────────
_CSS = """
@import url("https://rsms.me/inter/inter.css");
:root{--signal-blue:#2c6bed;--page-bg:#f6f7f9;--card-bg:#ffffff;--text:#0d0d0d;--text-sec:#5c5c5c;--border:#d8d8d8;--radius:12px;--green:#16a34a;--yellow:#ca8a04}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Inter",-apple-system,BlinkMacSystemFont,system-ui,sans-serif;background:var(--page-bg);color:var(--text);min-height:100vh;padding:48px 20px;-webkit-font-smoothing:antialiased}
@media(max-width:520px){body{padding:24px 12px}}
.shell{max-width:640px;margin:0 auto}
.card{background:var(--card-bg);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:16px}
header{display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--border)}
.header-left{display:flex;align-items:center;gap:10px;text-decoration:none;color:inherit}
.logo{width:28px;height:28px}
.brand{font-size:15px;font-weight:600;letter-spacing:-.02em}
.status-badge{padding:4px 10px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;border-radius:4px}
.status-solved{background:#dcfce7;color:var(--green)}
.status-recommendation{background:#fef3c7;color:#b45309}
main{padding:24px 20px}
@media(max-width:520px){main{padding:20px 16px}}
h1{font-size:20px;font-weight:700;letter-spacing:-.025em;margin-bottom:12px;line-height:1.3}
.meta{font-size:12px;color:var(--text-sec);margin-bottom:16px}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:20px}
.tag{background:rgba(44,107,237,.08);color:var(--signal-blue);padding:4px 10px;border-radius:4px;font-size:12px;font-weight:500}
.section-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--text-sec);margin-bottom:10px;display:flex;align-items:center;gap:6px}
.section-title svg{width:14px;height:14px}
.section-content{font-size:15px;line-height:1.6;color:var(--text);white-space:pre-wrap}
.problem-section{padding-bottom:20px;border-bottom:1px solid var(--border);margin-bottom:20px}
.solution-section{background:#f0fdf4;margin:-24px -20px -24px -20px;padding:20px;border-top:1px solid #bbf7d0}
.solution-section .section-title{color:var(--green)}
.recommendation-solution{background:#fffbeb;border-top-color:#fde68a}
.recommendation-solution .section-title{color:#b45309}
@media(max-width:520px){.solution-section{margin:-20px -16px -20px -16px;padding:16px}}
.chat-header{padding:14px 20px;border-bottom:1px solid var(--border);background:var(--page-bg)}
.chat-header h2{font-size:14px;font-weight:600;color:var(--text);display:flex;align-items:center;gap:8px}
.chat-header h2 svg{width:16px;height:16px;color:var(--signal-blue)}
.messages{padding:0}
.message{padding:16px 20px;border-bottom:1px solid var(--border)}
.message:last-child{border-bottom:none}
.message-header{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.avatar{width:28px;height:28px;border-radius:50%;background:var(--signal-blue);color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;flex-shrink:0}
.sender-name{font-size:13px;font-weight:600;color:var(--text)}
.message-time{font-size:11px;color:var(--text-sec)}
.message-text{font-size:15px;line-height:1.55;color:var(--text);white-space:pre-wrap;margin-left:38px}
.message-images{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;margin-left:38px}
.message-images img{display:block;max-width:320px;max-height:240px;border-radius:8px;border:1px solid var(--border);object-fit:cover}
.message-images video{display:block;max-width:320px;max-height:240px;border-radius:8px;border:1px solid var(--border)}
.empty-chat{padding:32px 20px;text-align:center;color:var(--text-sec);font-size:14px}
footer{padding:14px 20px;border-top:1px solid var(--border);color:var(--text-sec);font-size:12px;text-align:center}
@media(max-width:520px){.message{padding:14px 16px}.message-text{margin-left:0;margin-top:8px}.message-images{margin-left:0}}
/* Index */
.case-list{list-style:none}
.case-list li{padding:10px 0;border-bottom:1px solid var(--border);font-size:15px;line-height:1.55;display:flex;align-items:baseline;gap:8px}
.case-list li:last-child{border-bottom:none}
.case-list a{color:var(--text);text-decoration:none}
.case-list a:hover{color:var(--signal-blue)}
.case-list .status{font-size:10px;font-weight:600;text-transform:uppercase;padding:2px 6px;border-radius:3px;flex-shrink:0}
"""

_SVG_PROBLEM = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
_SVG_SOLVED = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>'
_SVG_CHAT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'

# Avatar colors per participant index
_AVATAR_COLORS = ['#2c6bed', '#16a34a', '#dc2626', '#9333ea', '#ea580c', '#0891b2', '#4f46e5', '#be185d']


_all_msgs: list[dict] = []  # all dataset messages for date lookup


def _case_date(idx: int) -> str:
    """Derive a display date for case idx from message timestamps."""
    from datetime import datetime
    if not _all_msgs:
        return ""
    # Cases are extracted from ~75-msg chunks of history
    # Rough position: spread cases evenly across the message timeline
    n = len(_all_msgs)
    pos = min(int(idx / max(len(_cases), 1) * n), n - 1)
    ts = _all_msgs[pos].get("ts", 0)
    if not ts:
        return ""
    try:
        dt = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts)
        return dt.strftime("%-d %b %Y, %H:%M")
    except Exception:
        return ""


def _e(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_e(title)} | SupportBot</title>
<link rel="icon" type="image/png" href="/static/favicon.png">
<style>{_CSS}</style>
</head><body>
<div class="shell">
{body}
</div>
</body></html>"""


def _render_case(case_id: str) -> str | None:
    try:
        idx = int(case_id.replace("eval_", ""))
    except (ValueError, AttributeError):
        return None
    if idx < 0 or idx >= len(_cases):
        return None

    c = _cases[idx]
    status = c.get("status", "recommendation")
    is_solved = status == "solved"

    status_cls = "status-solved" if is_solved else "status-recommendation"
    status_text = "Solved" if is_solved else "Recommendation"

    tags_html = "".join(f'<span class="tag">#{_e(t)}</span>' for t in c.get("tags", []))

    solution_cls = "solution-section" if is_solved else "solution-section recommendation-solution"
    solution_icon = _SVG_SOLVED if is_solved else _SVG_PROBLEM
    solution_label = "Solution" if is_solved else "Recommendation (unconfirmed)"

    # Evidence card
    evidence_ids = c.get("evidence_ids", [])
    evidence_card = ""
    if evidence_ids:
        sender_order = []
        msgs_data = []
        for eid in evidence_ids:
            msg = _msgs_by_id.get(str(eid))
            if not msg:
                continue
            sender = msg.get("sender", "?")
            if sender not in sender_order:
                sender_order.append(sender)
            msgs_data.append(msg)

        msg_items = []
        for msg in msgs_data:
            sender = msg.get("sender", "?")
            num = sender_order.index(sender) + 1
            initials = f"U{num}"
            color = _AVATAR_COLORS[(num - 1) % len(_AVATAR_COLORS)]
            body = _e(msg.get("body") or msg.get("text") or "")

            media_html = ""
            mp = msg.get("media_path")
            if mp and _dataset_dir:
                full = _dataset_dir / mp
                if full.exists():
                    mime = mimetypes.guess_type(str(full))[0] or ""
                    if mime.startswith("image/"):
                        media_html = f'<div class="message-images"><img src="/media/{mp}" alt=""></div>'
                    elif mime.startswith("video/"):
                        media_html = f'<div class="message-images"><video controls preload="metadata"><source src="/media/{mp}"></video></div>'

            msg_items.append(f"""<div class="message">
<div class="message-header">
<div class="avatar" style="background:{color}">{initials}</div>
<div><span class="sender-name">Participant {num}</span></div>
</div>
{f'<p class="message-text">{body[:600]}</p>' if body else ''}
{media_html}
</div>""")

        evidence_card = f"""<div class="card">
<div class="chat-header"><h2>{_SVG_CHAT} Evidence Messages</h2></div>
<div class="messages">{"".join(msg_items)}</div>
<footer>Academia Tech &copy; 2026</footer>
</div>"""
    else:
        evidence_card = f"""<div class="card">
<div class="chat-header"><h2>{_SVG_CHAT} Evidence Messages</h2></div>
<div class="empty-chat">Evidence messages not available for this case</div>
<footer>Academia Tech &copy; 2026</footer>
</div>"""

    body = f"""<div class="card">
<header>
<a href="/" class="header-left">
<img src="/static/supportbot-logo-128.png" alt="SupportBot" class="logo"/>
<span class="brand">SupportBot</span>
</a>
<span class="{status_cls} status-badge">{status_text}</span>
</header>
<main>
<h1>{_e(c.get('problem_title', 'Untitled'))}</h1>
<p class="meta">{_case_date(idx)}</p>
{f'<div class="tags">{tags_html}</div>' if tags_html else ''}
<div class="problem-section">
<h2 class="section-title">{_SVG_PROBLEM} Problem</h2>
<p class="section-content">{_e(c.get('problem_summary', ''))}</p>
</div>
<div class="{solution_cls}">
<h2 class="section-title">{solution_icon} {solution_label}</h2>
<p class="section-content">{_e(c.get('solution_summary', ''))}</p>
</div>
</main>
</div>
{evidence_card}"""

    return _page(c.get("problem_title", "Case"), body)


def _render_index() -> str:
    solved = sum(1 for c in _cases if c.get("status") == "solved")
    reco = len(_cases) - solved
    items = []
    for i, c in enumerate(_cases):
        status = c.get("status", "recommendation")
        s_cls = "status-solved" if status == "solved" else "status-recommendation"
        s_text = "S" if status == "solved" else "R"
        title = _e(c.get("problem_title", "?"))
        items.append(
            f'<li><span class="status {s_cls}">{s_text}</span>'
            f'<a href="/case/eval_{i}">{title}</a></li>'
        )
    body = f"""<div class="card">
<header>
<a href="/" class="header-left">
<img src="/static/supportbot-logo-128.png" alt="SupportBot" class="logo"/>
<span class="brand">SupportBot</span>
</a>
</header>
<main>
<h1>Knowledge Base</h1>
<p class="meta">{len(_cases)} cases &mdash; {solved} solved, {reco} recommendations</p>
<ul class="case-list">{"".join(items)}</ul>
</main>
<footer>Academia Tech &copy; 2026</footer>
</div>"""
    return _page("Knowledge Base", body)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path)

        if path.startswith("/case/"):
            case_id = path.split("/case/")[1].rstrip("/")
            html = _render_case(case_id)
            if html:
                self._send(200, html, "text/html; charset=utf-8")
            else:
                self._send(404, "Case not found")
            return

        if path.startswith("/media/") and _dataset_dir:
            rel = path[len("/media/"):]
            full = _dataset_dir / rel
            if full.exists() and full.is_file():
                mime = mimetypes.guess_type(str(full))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(full.read_bytes())
            else:
                self._send(404, "Not found")
            return

        if path.startswith("/static/"):
            fname = path[len("/static/"):]
            full = _static_dir / fname
            if full.exists() and full.is_file():
                mime = mimetypes.guess_type(str(full))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(full.read_bytes())
            else:
                self._send(404, "Not found")
            return

        if path in ("/", ""):
            self._send(200, _render_index(), "text/html; charset=utf-8")
            return

        self._send(404, "Not found")

    def do_HEAD(self):
        self.do_GET()

    def _send(self, code: int, body: str, ct: str = "text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)

    def log_message(self, format, *args):
        pass


def start_case_server(cases: list[dict], dataset_msgs: list[dict],
                       dataset_dir: Path | None = None, port: int = 8099) -> HTTPServer:
    global _cases, _msgs_by_id, _dataset_dir, _all_msgs
    _cases = cases
    _dataset_dir = dataset_dir
    _all_msgs = dataset_msgs
    _msgs_by_id.clear()
    for m in dataset_msgs:
        _msgs_by_id[str(m.get("id", ""))] = m
    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def stop_case_server(server: HTTPServer) -> None:
    server.shutdown()

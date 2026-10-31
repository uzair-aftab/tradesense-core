"""
TradeSense Verified 1-Click Clipboard Copy Cards
=================================================
Renders interactive frosted-glass cards that copy order values (Limit Entry,
Take Profit, Stop Loss, Quantity) directly to the system clipboard via HTML5
navigator.clipboard with instant visual feedback.
"""

from __future__ import annotations

import streamlit.components.v1 as components


def render_copy_card(
    label: str,
    value_str: str,
    subtext: str = "",
    accent: str = "#38bdf8",
    copy_raw: str = "",
    height: int = 76,
) -> None:
    """
    Renders an interactive 1-click clipboard card using an embedded HTML5 snippet.
    """
    raw_to_copy = copy_raw if copy_raw else value_str.replace("$", "").replace(",", "").strip()

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            user-select: none;
        }}
        body {{
            background: transparent;
            overflow: hidden;
        }}
        .glass-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 8px 12px;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: {height - 4}px;
        }}
        .glass-card:hover {{
            background: rgba(255, 255, 255, 0.06);
            border-color: {accent}88;
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        }}
        .glass-card:active {{
            transform: translateY(0);
        }}
        .header-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2px;
        }}
        .label {{
            font-size: 0.68rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
        }}
        .copy-badge {{
            font-size: 0.62rem;
            color: #64748b;
            background: rgba(255, 255, 255, 0.05);
            padding: 2px 5px;
            border-radius: 4px;
            transition: color 0.2s;
        }}
        .value {{
            font-family: "JetBrains Mono", "SF Mono", Consolas, monospace;
            font-size: 1.05rem;
            font-weight: 700;
            color: {accent};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .subtext {{
            font-size: 0.68rem;
            color: #64748b;
            margin-top: 1px;
        }}
        .copied-toast {{
            position: absolute;
            right: 10px;
            top: 8px;
            background: rgba(52, 211, 153, 0.2);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.4);
            font-size: 0.68rem;
            padding: 2px 7px;
            border-radius: 5px;
            font-weight: 600;
            opacity: 0;
            transition: opacity 0.2s ease;
            pointer-events: none;
        }}
        .copied-toast.show {{
            opacity: 1;
        }}
    </style>
    </head>
    <body>
        <div class="glass-card" onclick="copyValue()">
            <div class="header-row">
                <span class="label">{label}</span>
                <span class="copy-badge" id="badge">Click to Copy</span>
            </div>
            <div class="value">{value_str}</div>
            {f'<div class="subtext">{subtext}</div>' if subtext else ''}
            <div class="copied-toast" id="toast">✓ Copied</div>
        </div>

        <script>
            function copyValue() {{
                const text = "{raw_to_copy}";
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(text).then(showToast);
                }} else {{
                    const ta = document.createElement("textarea");
                    ta.value = text;
                    ta.style.position = "fixed";
                    ta.style.opacity = "0";
                    document.body.appendChild(ta);
                    ta.select();
                    try {{
                        document.execCommand("copy");
                        showToast();
                    }} catch (e) {{}}
                    document.body.removeChild(ta);
                }}
            }}

            function showToast() {{
                const toast = document.getElementById("toast");
                const badge = document.getElementById("badge");
                toast.classList.add("show");
                badge.style.color = "#34d399";
                badge.innerText = "Copied!";
                setTimeout(() => {{
                    toast.classList.remove("show");
                    badge.style.color = "#64748b";
                    badge.innerText = "Click to Copy";
                }}, 1400);
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)

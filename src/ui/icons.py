"""
TradeSense Institutional SVG Icon Library
==========================================
Clean, scalable, stroke-based vector icons (Lucide / Feather standard).
Eliminates amateur cartoon emojis across headers, tabs, metric cards, and buttons.
"""

from typing import Dict


def svg_icon(
    name: str,
    size: int = 18,
    color: str = "currentColor",
    stroke_width: float = 1.75,
    class_name: str = "",
) -> str:
    """
    Return inline SVG markup for the requested icon name.
    
    Parameters
    ----------
    name : str
        Icon identifier (e.g. 'activity', 'trending_up', 'trophy', 'cpu', 'radar').
    size : int, default 18
        Width and height in pixels.
    color : str, default 'currentColor'
        SVG stroke color (e.g. '#10b981', '#38bdf8', '#f59e0b', 'currentColor').
    stroke_width : float, default 1.75
        Stroke thickness.
    class_name : str, optional
        Extra CSS class to append.
    """
    cls = f' class="ts-svg-icon {class_name}"' if class_name else ' class="ts-svg-icon"'
    style = f'style="vertical-align: -0.18em; display: inline-block; flex-shrink: 0;"'

    icons: Dict[str, str] = {
        # Navigation & Core Systems
        "cockpit": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/>'
            f'<line x1="6" y1="20" x2="6" y2="16"/></svg>'
        ),
        "tournament": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>'
            f'<path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>'
            f'<path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>'
            f'<path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/></svg>'
        ),
        "trophy": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>'
            f'<path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>'
            f'<path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>'
            f'<path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/></svg>'
        ),
        "award": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/></svg>'
        ),
        "neural": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M12 2a4 4 0 0 1 4 4c0 1.5-.8 2.8-2 3.5V11a2 2 0 0 1-2 2h-1v2.1a4 4 0 1 1-2 0V13H8a2 2 0 0 1-2-2V9.5C4.8 8.8 4 7.5 4 6a4 4 0 0 1 8-4Z"/>'
            f'<path d="M19 12a3 3 0 0 1 3 3c0 1.1-.6 2.1-1.5 2.6L20 20a2 2 0 0 1-2 2h-3v-2h3l.3-1.8A3 3 0 0 1 19 12Z"/>'
            f'</svg>'
        ),
        "radar": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M19.07 4.93A10 10 0 0 0 6.99 3.34"/><path d="M4 6h.01"/>'
            f'<path d="M2.29 9.62A10 10 0 0 0 21.42 14.4"/><path d="M16.24 7.76A6 6 0 1 0 8.23 16.24"/>'
            f'<path d="M12 18h.01"/><path d="M17.99 11.66A6 6 0 0 1 15.77 15.77"/><circle cx="12" cy="12" r="2"/>'
            f'<path d="m13.41 10.59 5.66-5.66"/></svg>'
        ),
        "telemetry": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
        ),

        # Trading & Microstructure
        "activity": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
        ),
        "trending_up": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>'
        ),
        "trending_down": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>'
        ),
        "target": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>'
        ),
        "crosshair": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/>'
            f'<line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>'
        ),
        "shield": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
        ),
        "shield_alert": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
            f'<line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
        ),
        "whale": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M2 12c3-4 7-6 11-4 3 1.5 5 5 7 5 1.5 0 2-2 2-2s-1 4-4 5c-3 1-8 0-11-2-2-1.3-3.5-2-5-2Z"/>'
            f'<circle cx="8" cy="11" r="1"/></svg>'
        ),
        "zap": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
        ),
        "check": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<polyline points="20 6 9 17 4 12"/></svg>'
        ),
        "check_circle": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
        ),
        "alert_triangle": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
            f'<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
        ),
        "bell": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>'
        ),
        "copy": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>'
            f'<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>'
        ),

        # Hardware & Platform
        "cpu": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
            f'<path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/>'
            f'<path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>'
        ),
        "database": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
            f'<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>'
        ),
        "terminal": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>'
        ),
        "server": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<rect width="20" height="8" x="2" y="2" rx="2" ry="2"/>'
            f'<rect width="20" height="8" x="2" y="14" rx="2" ry="2"/>'
            f'<line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/></svg>'
        ),
        "user": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
        ),
        "log_out": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/>'
            f'<line x1="21" y1="12" x2="9" y2="12"/></svg>'
        ),
        "refresh_cw": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>'
            f'<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>'
        ),
        "search": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'
        ),
        "filter": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>'
        ),
        "download": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/>'
            f'<line x1="12" y1="15" x2="12" y2="3"/></svg>'
        ),
        "layers": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/>'
            f'<polyline points="2 12 12 17 22 12"/></svg>'
        ),
        "sliders": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/>'
            f'<line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/>'
            f'<line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
            f'<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/>'
            f'<line x1="17" y1="16" x2="23" y2="16"/></svg>'
        ),
        "phone": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<rect width="14" height="20" x="5" y="2" rx="2" ry="2"/>'
            f'<path d="M12 18h.01"/></svg>'
        ),
        "map_pin": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>'
        ),
        "leaf": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>'
            f'<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>'
        ),
        "help_circle": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<circle cx="12" cy="12" r="10"/>'
            f'<path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
            f'<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
        ),
        "info": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"{cls} {style}>'
            f'<circle cx="12" cy="12" r="10"/>'
            f'<line x1="12" y1="16" x2="12" y2="12"/>'
            f'<line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
        ),
    }

    return icons.get(name, "")


def render_info_badge(label: str, tooltip_text: str, color: str = "#38bdf8") -> str:
    """
    Renders an inline educational info badge with hover tooltip for beginner traders.
    """
    safe_tip = tooltip_text.replace('"', '&quot;')
    icon_svg = svg_icon("help_circle", size=13, color=color)
    return (
        f'<span style="display: inline-flex; align-items: center; gap: 4px; cursor: help; border-bottom: 1px dotted {color}88;" title="{safe_tip}">'
        f'<span>{label}</span>'
        f'{icon_svg}'
        f'</span>'
    )


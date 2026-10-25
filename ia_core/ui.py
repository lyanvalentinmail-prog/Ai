"""
Interfaz de consola (CLI/TUI) optimizada para pantallas verticales de Termux (Android) y Linux.
Sin dependencias obligatorias: usa ANSI TrueColor (24-bit) compatible nativamente con Termux.
Incluye renderizador de imágenes y fotogramas de video en terminal mediante bloques '▀'.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
import time
from typing import Iterable, List, Optional, Sequence, Tuple

# Detectar soporte de color
SUPPORTS_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class C:
    """Códigos ANSI elegantes de alta legibilidad en fondos oscuros de Termux."""

    RESET = "\033[0m" if SUPPORTS_COLOR else ""
    BOLD = "\033[1m" if SUPPORTS_COLOR else ""
    DIM = "\033[2m" if SUPPORTS_COLOR else ""
    ITALIC = "\033[3m" if SUPPORTS_COLOR else ""

    CYAN = "\033[38;2;56;189;248m" if SUPPORTS_COLOR else ""
    BLUE = "\033[38;2;96;165;250m" if SUPPORTS_COLOR else ""
    PURPLE = "\033[38;2;192;132;252m" if SUPPORTS_COLOR else ""
    GREEN = "\033[38;2;74;222;128m" if SUPPORTS_COLOR else ""
    YELLOW = "\033[38;2;250;204;21m" if SUPPORTS_COLOR else ""
    ORANGE = "\033[38;2;251;146;60m" if SUPPORTS_COLOR else ""
    RED = "\033[38;2;248;113;113m" if SUPPORTS_COLOR else ""
    GRAY = "\033[38;2;148;163;184m" if SUPPORTS_COLOR else ""
    WHITE = "\033[38;2;241;245;249m" if SUPPORTS_COLOR else ""
    BG_CODE = "\033[48;2;30;41;59m" if SUPPORTS_COLOR else ""


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(text: str) -> int:
    """Longitud real visible ignorando secuencias ANSI."""
    return len(ANSI_RE.sub("", text))


def term_width(default: int = 68, max_width: int = 96) -> int:
    """Obtiene el ancho actual de la consola de Termux con límites cómodos de lectura."""
    try:
        cols = shutil.get_terminal_size((default, 24)).columns
        return max(38, min(cols, max_width))
    except Exception:
        return default


def print_banner(
    ai_name: str,
    engine_label: str,
    memory_count: int,
    safe_mode: bool,
    is_termux_env: bool,
) -> None:
    """Imprime cabecera compacta adaptada a pantallas móviles de Termux."""
    w = term_width()
    line = "─" * (w - 2)
    env_tag = "Android/Termux" if is_termux_env else "Linux/Local"
    safe_tag = f"{C.GREEN}Seguro ON{C.RESET}" if safe_mode else f"{C.ORANGE}Auto{C.RESET}"

    print(f"{C.CYAN}╭{line}╮{C.RESET}")
    title = f"⚡ {C.BOLD}{ai_name}{C.RESET} {C.PURPLE}IA Propia 100% Local{C.RESET}"
    pad1 = max(0, (w - 2) - visible_len(title) - 2)
    print(f"{C.CYAN}│{C.RESET} {title}{' ' * pad1} {C.CYAN}│{C.RESET}")

    meta = f"{C.GRAY}Entorno:{C.RESET} {env_tag}  {C.GRAY}│{C.RESET}  {C.GRAY}Modo:{C.RESET} {safe_tag}"
    pad2 = max(0, (w - 2) - visible_len(meta) - 2)
    print(f"{C.CYAN}│{C.RESET} {meta}{' ' * pad2} {C.CYAN}│{C.RESET}")

    brain = f"{C.GRAY}Cerebro:{C.RESET} {C.GREEN}{engine_label}{C.RESET} ({memory_count} recuerdos)"
    if visible_len(brain) > w - 4:
        brain = f"{C.GREEN}{engine_label}{C.RESET} ({memory_count} mem)"
    pad3 = max(0, (w - 2) - visible_len(brain) - 2)
    print(f"{C.CYAN}│{C.RESET} {brain}{' ' * pad3} {C.CYAN}│{C.RESET}")

    hint = f"{C.DIM}Escribe un mensaje o /ayuda · /imagen · /video · /buscar · /agente{C.RESET}"
    if visible_len(hint) > w - 4:
        hint = f"{C.DIM}/ayuda · /imagen · /video · /buscar · /agente{C.RESET}"
    pad4 = max(0, (w - 2) - visible_len(hint) - 2)
    print(f"{C.CYAN}│{C.RESET} {hint}{' ' * pad4} {C.CYAN}│{C.RESET}")
    print(f"{C.CYAN}╰{line}╯{C.RESET}")


def format_markdown(text: str) -> str:
    """Formatea Markdown ligero (bloques de código, negritas, títulos, listas) para terminal."""
    w = term_width()
    lines = text.splitlines()
    out: List[str] = []
    in_code = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            lang = stripped[3:].strip()
            if in_code:
                label = f" {lang} " if lang else " código "
                bar = "─" * max(4, w - len(label) - 6)
                out.append(f"{C.GRAY}┌─{C.YELLOW}{label}{C.GRAY}{bar}{C.RESET}")
            else:
                out.append(f"{C.GRAY}└{'─' * max(6, w - 4)}{C.RESET}")
            continue

        if in_code:
            out.append(f"{C.GRAY}│{C.RESET} {C.CYAN}{raw_line}{C.RESET}")
            continue

        # Encabezados
        if stripped.startswith("### "):
            out.append(f"{C.BOLD}{C.PURPLE}▸ {stripped[4:]}{C.RESET}")
            continue
        if stripped.startswith("## "):
            out.append(f"\n{C.BOLD}{C.CYAN}◆ {stripped[3:]}{C.RESET}")
            continue
        if stripped.startswith("# "):
            out.append(f"\n{C.BOLD}{C.BLUE}══ {stripped[2:]} ══{C.RESET}")
            continue

        # Negritas y código inline
        formatted = re.sub(
            r"\*\*(.+?)\*\*",
            lambda m: f"{C.BOLD}{C.WHITE}{m.group(1)}{C.RESET}",
            raw_line,
        )
        formatted = re.sub(
            r"`([^`]+)`",
            lambda m: f"{C.YELLOW}{m.group(1)}{C.RESET}",
            formatted,
        )

        # Ajuste suave de líneas largas respetando listas
        if visible_len(formatted) > w and not stripped.startswith(("http://", "https://", "│", "╭", "╰")):
            indent = ""
            if stripped.startswith(("- ", "* ", "• ")):
                indent = "  "
            wrapped = textwrap.wrap(
                formatted,
                width=w,
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
            out.extend(wrapped if wrapped else [formatted])
        else:
            out.append(formatted)

    return "\n".join(out)


def print_ai_response(ai_name: str, text: str, tag: str = "") -> None:
    """Muestra la respuesta principal de la IA con formato limpio."""
    badge = f" {C.DIM}[{tag}]{C.RESET}" if tag else ""
    print(f"\n{C.PURPLE}● {C.BOLD}{ai_name}{C.RESET}{badge}")
    print(format_markdown(text))
    print()


def print_step(icon: str, title: str, detail: str = "", color: str = C.CYAN) -> None:
    """Imprime un paso intermedio (usado por Agente, Búsqueda o Generadores)."""
    suffix = f" {C.GRAY}— {detail}{C.RESET}" if detail else ""
    print(f"  {color}{icon}{C.RESET} {C.BOLD}{title}{C.RESET}{suffix}")


def print_success(msg: str) -> None:
    print(f"{C.GREEN}✔{C.RESET} {msg}")


def print_warning(msg: str) -> None:
    print(f"{C.YELLOW}⚠{C.RESET} {msg}")


def print_error(msg: str) -> None:
    print(f"{C.RED}✖{C.RESET} {msg}")


def progress_bar(current: int, total: int, prefix: str = "", width: int = 24) -> None:
    """Barra de progreso ligera compatible con pantallas pequeñas de Termux."""
    if total <= 0:
        return
    ratio = min(1.0, max(0.0, current / total))
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(ratio * 100)
    sys.stdout.write(f"\r  {C.CYAN}{prefix}{C.RESET} [{C.GREEN}{bar}{C.RESET}] {pct:3d}%")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def confirm_prompt(question: str, default: bool = False) -> bool:
    """Solicita confirmación interactiva [s/N] en consola."""
    if not sys.stdin.isatty():
        return default
    hint = "[S/n]" if default else "[s/N]"
    try:
        ans = input(f"  {C.YELLOW}?{C.RESET} {question} {C.GRAY}{hint}:{C.RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not ans:
        return default
    return ans in ("s", "si", "sí", "y", "yes")


def render_rgb_buffer_ansi(
    pixels: Sequence[Sequence[Tuple[int, int, int]]],
    max_cols: Optional[int] = None,
) -> str:
    """
    Convierte una matriz RGB [alto][ancho][(r,g,b)] en arte ANSI TrueColor de 24 bits
    usando el carácter '▀' (dos píxeles verticales por celda de terminal).
    Se ve nítido y a todo color directamente en Termux.
    """
    src_h = len(pixels)
    if src_h == 0:
        return ""
    src_w = len(pixels[0])
    if src_w == 0:
        return ""

    cols = max_cols or min(term_width() - 4, 64)
    cols = max(16, cols)
    # Mantener proporción real: cada fila de texto representa 2 píxeles verticales
    scale = src_w / cols
    target_h_pixels = max(2, int(round(src_h / scale)))
    if target_h_pixels % 2 == 1:
        target_h_pixels += 1

    lines: List[str] = []
    use_ansi = SUPPORTS_COLOR or os.environ.get("FORCE_ANSI_PREVIEW") == "1"
    ascii_ramp = " .:-=+*#%@"

    for py in range(0, target_h_pixels, 2):
        row_chars: List[str] = ["  "]
        sy_top = min(src_h - 1, int(py * src_h / target_h_pixels))
        sy_bot = min(src_h - 1, int((py + 1) * src_h / target_h_pixels))
        for px in range(cols):
            sx = min(src_w - 1, int(px * src_w / cols))
            r1, g1, b1 = pixels[sy_top][sx]
            r2, g2, b2 = pixels[sy_bot][sx]
            if use_ansi:
                row_chars.append(f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀")
            else:
                lum = int(0.299 * r1 + 0.587 * g1 + 0.114 * b1)
                idx = min(len(ascii_ramp) - 1, lum * len(ascii_ramp) // 256)
                row_chars.append(ascii_ramp[idx])
        if use_ansi:
            row_chars.append("\033[0m")
        lines.append("".join(row_chars))

    return "\n".join(lines)


def play_frames_in_terminal(
    frames: Iterable[Sequence[Sequence[Tuple[int, int, int]]]],
    fps: int = 10,
    max_cols: Optional[int] = None,
) -> None:
    """Reproduce una animación de fotogramas RGB directamente en la consola de Termux."""
    frame_list = list(frames)
    if not frame_list:
        return
    delay = 1.0 / max(1, fps)
    rendered = [render_rgb_buffer_ansi(f, max_cols=max_cols) for f in frame_list]
    height_lines = rendered[0].count("\n") + 1

    if not sys.stdout.isatty():
        # En salida no interactiva mostramos el primer y último fotograma
        print(rendered[0])
        return

    try:
        sys.stdout.write("\033[?25l")  # Ocultar cursor
        for idx, art in enumerate(rendered):
            t0 = time.perf_counter()
            if idx > 0:
                sys.stdout.write(f"\033[{height_lines}A")
            sys.stdout.write(art + "\n")
            sys.stdout.flush()
            elapsed = time.perf_counter() - t0
            if elapsed < delay:
                time.sleep(delay - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\033[0m")
        sys.stdout.flush()

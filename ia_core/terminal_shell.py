"""
NEXUS-TERM — Terminal y Shell Propia Completa con IA Integrada para Termux y Linux.

Convierte la ejecución de `./ia` en un entorno de terminal propio e independiente:
- Pantalla aislada (Alternate Screen Buffer \\033[?1049h) que preserva intacta tu consola anterior
- HUD superior estilo Sistema Operativo con telemetría de RAM, CPU, Batería, Reloj y Cerebro IA
- Temas visuales propios cambiables en vivo (`cyber`, `matrix`, `ambar`, `oceano`)
- Prompt propio de 2 líneas con ruta inteligente, estado Git ultrarrápido y métricas de ejecución
- Motor Híbrido: ejecuta comandos Linux/Termux reales (`cd`, `ls`, `git`, `pkg`, `vim`, alias, pipes)
  Y peticiones a los 5 módulos de la IA (Texto, Buscar, Imágenes, Video, Agente) en el mismo lugar
- Doctor de errores (`reparar`): analiza automáticamente el último comando fallido de consola
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .agent_engine import SafetyGuard
from .config import get_system_resources, is_termux
from .memory import strip_accents
from .ui import (
    SUPPORTS_COLOR,
    THEMES,
    C,
    apply_theme,
    confirm_prompt,
    enter_alt_screen,
    exit_alt_screen,
    format_markdown,
    print_ai_response,
    print_error,
    print_nexus_term_hud,
    print_step,
    print_success,
    print_warning,
    set_terminal_title,
    term_width,
)

if TYPE_CHECKING:
    from .cli import NexusApplication


DEFAULT_ALIASES: Dict[str, str] = {
    "ll": "ls -lah --color=auto",
    "la": "ls -A --color=auto",
    "l": "ls -CF --color=auto",
    "espacio": "df -h",
    "ram": "free -m",
    "actualizar": "pkg update -y && pkg upgrade -y" if is_termux() else "sudo apt update",
}

SHELL_BUILTINS = {
    "cd", "..", "...", "pwd", "export", "unset", "alias", "unalias", "aliases",
    "clear", "cls", "hud", "panel", "tema", "modo", "reparar",
    "historial", "history", "ayuda", "help", "salir", "exit", "logout",
}

INTERACTIVE_PROGRAMS = {
    "vim", "vi", "nvim", "nano", "micro", "emacs",
    "htop", "top", "btop", "less", "more", "man",
    "tmux", "screen", "ssh", "bash", "zsh", "sh", "fish",
}

NATURAL_AI_STARTERS = (
    "hola", "buenas", "buenos dias", "buenas tardes", "buenas noches",
    "dime", "explica", "explicame", "calcula", "resuelve", "cuanto ", "cuánto ",
    "resume", "resumir", "traduce", "recuerda ", "aprende ", "me llamo ",
    "como me llamo", "quien eres", "que eres", "que es ", "qué es ", "que puedes",
    "por que ", "por qué ", "como funciona", "cómo funciona",
    "dibuja", "pinta", "genera una imagen", "crea una imagen", "haz una imagen",
    "genera un video", "crea un video", "haz un video", "anima ",
    "crea un script", "crea una calculadora", "crea un programa", "haz un programa",
    "escribe un codigo", "escribe un script", "busca ", "investiga ", "modo agente",
)

# Binarios cortos que también podrían aparecer al inicio de una oración en español
AMBIGUOUS_SHORT_CMDS = {"id", "who", "cal", "man", "yes", "date", "factor"}


def _rl_wrap(ansi_code: str) -> str:
    """Envuelve secuencias ANSI con \\001..\\002 para que readline calcule bien el ancho del cursor."""
    if not ansi_code or not sys.stdin.isatty():
        return ansi_code
    return f"\001{ansi_code}\002"


def detect_git_status_fast(cwd: Path) -> Optional[str]:
    """Lee la rama Git actual directamente desde .git/HEAD en <2 ms sin lanzar procesos pesados."""
    cur = cwd
    git_dir: Optional[Path] = None
    for _ in range(8):
        candidate = cur / ".git"
        if candidate.exists():
            git_dir = candidate
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    if git_dir is None:
        return None

    try:
        if git_dir.is_file():
            # Submódulo o worktree: "gitdir: ..."
            ref_txt = git_dir.read_text(encoding="utf-8", errors="ignore").strip()
            if ref_txt.startswith("gitdir:"):
                git_dir = (cur / ref_txt.split(":", 1)[1].strip()).resolve()
        head_file = git_dir / "HEAD"
        if not head_file.exists():
            return None
        head = head_file.read_text(encoding="utf-8", errors="ignore").strip()
        if head.startswith("ref: refs/heads/"):
            return head[len("ref: refs/heads/") :]
        return head[:7]
    except Exception:
        return None


def shorten_path_mobile(cwd: Path, max_chars: int = 24) -> str:
    """Abrevia rutas largas para que el prompt luzca limpio en pantallas verticales de Android."""
    home = Path.home()
    try:
        rel = "~/" + str(cwd.relative_to(home))
        if rel == "~/.":
            rel = "~"
    except ValueError:
        rel = str(cwd)

    if len(rel) <= max_chars:
        return rel
    parts = Path(rel).parts
    if len(parts) <= 2:
        return rel[-max_chars:]
    return f"{parts[0]}/…/{parts[-1]}"


def launch_in_new_session(exe_path: Path) -> bool:
    """
    Intenta abrir NEXUS-TERM en una sesión/ventana completamente aparte usando:
    1. `tmux` (nueva ventana si ya estás en tmux, o nueva sesión dedicada `nexus-term`)
    2. Android `am startservice` hacia `RunCommandService` de Termux si está configurado
    """
    cmd_str = f"{shlex.quote(sys.executable)} {shlex.quote(str(exe_path))}"

    if shutil.which("tmux"):
        if os.environ.get("TMUX"):
            res = subprocess.run(["tmux", "new-window", "-n", "NEXUS-TERM", cmd_str])
            return res.returncode == 0
        res = subprocess.run(["tmux", "new-session", "-s", "nexus-term", "-n", "NEXUS-TERM", cmd_str])
        return res.returncode == 0

    if is_termux() and shutil.which("am"):
        res = subprocess.run(
            [
                "am",
                "startservice",
                "--user",
                "0",
                "-n",
                "com.termux/com.termux.app.RunCommandService",
                "-a",
                "com.termux.RUN_COMMAND",
                "--es",
                "com.termux.RUN_COMMAND_PATH",
                sys.executable,
                "--esa",
                "com.termux.RUN_COMMAND_ARGUMENTS",
                str(exe_path),
                "--es",
                "com.termux.RUN_COMMAND_WORKDIR",
                str(Path.cwd()),
                "--ez",
                "com.termux.RUN_COMMAND_BACKGROUND",
                "false",
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            return True

    return False


class NexusTerminalShell:
    """Entorno interactivo propio que fusiona un Shell Linux/Termux real con los 5 módulos de NEXUS."""

    def __init__(self, app: "NexusApplication", isolate_screen: Optional[bool] = None):
        self.app = app
        self.config = app.config
        self.use_alt_screen = self.config.use_alt_screen if isolate_screen is None else isolate_screen

        self.theme_name = apply_theme(self.config.terminal_theme)
        self.shell_mode = self.config.shell_mode if self.config.shell_mode in ("hibrido", "ia", "shell", "agente") else "hibrido"

        self.old_pwd: Path = Path.cwd()
        self.aliases: Dict[str, str] = dict(DEFAULT_ALIASES)
        self.custom_env: Dict[str, str] = {}
        self.cmd_history: List[str] = []

        self.last_exit_code: int = 0
        self.last_duration: float = 0.0
        self.last_failed_cmd: str = ""
        self.last_failed_stderr: str = ""

        self._path_executables_cache: Optional[List[str]] = None
        self._load_shell_rc()

    # ------------------------------------------------------------------
    # Persistencia de alias y variables de entorno propias (shell_rc.json)
    # ------------------------------------------------------------------

    def _load_shell_rc(self) -> None:
        rc = self.config.shell_rc_path
        if not rc.exists():
            return
        try:
            data = json.loads(rc.read_text(encoding="utf-8"))
            if isinstance(data.get("aliases"), dict):
                self.aliases.update({str(k): str(v) for k, v in data["aliases"].items()})
            if isinstance(data.get("env"), dict):
                for k, v in data["env"].items():
                    self.custom_env[str(k)] = str(v)
                    os.environ[str(k)] = str(v)
        except Exception:
            pass

    def _save_shell_rc(self) -> None:
        user_only_aliases = {
            k: v for k, v in self.aliases.items() if DEFAULT_ALIASES.get(k) != v
        }
        payload = {"aliases": user_only_aliases, "env": self.custom_env}
        self.config.shell_rc_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Clasificador Inteligente Híbrido (Consola Linux vs IA Propia)
    # ------------------------------------------------------------------

    def _expand_alias(self, raw: str) -> str:
        parts = raw.strip().split(maxsplit=1)
        if not parts:
            return raw
        first = parts[0]
        if first in self.aliases:
            rest = f" {parts[1]}" if len(parts) > 1 else ""
            return f"{self.aliases[first]}{rest}"
        return raw

    def classify_input(self, raw: str) -> Tuple[str, str]:
        """
        Determina cómo ejecutar la línea escrita por el usuario en NEXUS-TERM.
        Retorna (tipo, contenido_limpio) donde tipo es:
        - 'builtin' : comando interno de NEXUS-TERM (cd, alias, tema, modo, clear, reparar, salir...)
        - 'slash'   : comando explícito '/...' de los módulos de IA
        - 'shell'   : comando real del sistema operativo (Termux / Linux)
        - 'agent'   : misión para el Agente Autónomo
        - 'ai'      : conversación, cálculo, búsqueda, imagen o video en lenguaje natural
        """
        s = raw.strip()
        if not s:
            return ("noop", "")

        # 1. Prefijos explícitos de 1 carácter
        if s.startswith("!") and not s[1:].isdigit() and s != "!!":
            return ("shell", s[1:].strip())
        if s.startswith("?"):
            return ("ai", s[1:].strip())
        if s.startswith("@"):
            return ("agent", s[1:].strip())

        # 2. Distinguir '/comando_ia' de rutas ejecutables absolutas '/bin/ls'
        if s.startswith("/"):
            first_tok = s.split()[0]
            if first_tok.count("/") >= 2 and Path(first_tok).exists():
                return ("shell", s)
            return ("slash", s)

        first_word = s.split()[0]
        first_low = first_word.lower()

        # 3. Built-ins propios de NEXUS-TERM (funcionan siempre en cualquier modo)
        if first_low in SHELL_BUILTINS:
            # Evitar capturar frases largas tipo "ayuda con mi tarea de matemáticas"
            if first_low in ("ayuda", "help") and len(s.split()) > 2:
                return ("ai", s)
            return ("builtin", s)

        # 4. Modos forzados por el usuario (`modo shell`, `modo ia`, `modo agente`)
        if self.shell_mode == "shell":
            return ("shell", self._expand_alias(s))
        if self.shell_mode == "ia":
            return ("ai", s)
        if self.shell_mode == "agente":
            return ("agent", s)

        # 5. Modo HÍBRIDO inteligente (por defecto)
        # Si el usuario definió explícitamente un alias para la primera palabra, tiene prioridad
        if first_word in self.aliases:
            return ("shell", self._expand_alias(s))

        low_norm = strip_accents(s.lower())

        # ¿Es claramente una consulta/orden en lenguaje natural a la IA?
        has_shell_ops = any(op in s for op in (" | ", " && ", " || ", " > ", " >> ", " 2>", "$("))
        if not has_shell_ops:
            if s.startswith(("¿", "¡")) or (s.endswith("?") and not Path(first_word).exists()):
                return ("ai", s)
            for starter in NATURAL_AI_STARTERS:
                st = starter.strip()
                if low_norm == st or low_norm.startswith(st + " ") or low_norm.startswith(st + ",") or low_norm.startswith(st + ":"):
                    return ("ai", s)

        # ¿Es una ruta ejecutable directa o asignación VAR=val?
        if s.startswith(("./", "../", "~/")):
            return ("shell", s)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", first_word):
            return ("shell", s)

        # ¿Es un alias o binario instalado en $PATH?
        expanded = self._expand_alias(s)
        exp_first = expanded.split()[0]
        if shutil.which(exp_first) is not None:
            words = s.split()
            if (
                first_low in AMBIGUOUS_SHORT_CMDS
                and len(words) >= 4
                and not any(w.startswith("-") for w in words[1:])
                and not has_shell_ops
            ):
                return ("ai", s)
            return ("shell", expanded)

        # Si NO existe como comando del sistema, en NEXUS-TERM nunca da "command not found":
        # pasa directamente al cerebro de la IA Propia.
        return ("ai", s)

    # ------------------------------------------------------------------
    # Prompt Propio de 2 Líneas Estilo OS (NEXUS-PROMPT)
    # ------------------------------------------------------------------

    def _print_prompt_header_and_get_line(self) -> str:
        cwd = Path.cwd()
        short_cwd = shorten_path_mobile(cwd)
        git_branch = detect_git_status_fast(cwd)
        git_tag = f" {C.PURPLE}⎇ {git_branch}{C.RESET}" if git_branch else ""

        if self.last_exit_code == 0:
            status_tag = f"{C.GREEN}✔ {self.last_duration:.2f}s{C.RESET}"
        else:
            status_tag = f"{C.RED}✖ {self.last_exit_code}{C.RESET}"

        mode_icons = {
            "hibrido": "⚡",
            "ia": "💬",
            "shell": "🐚",
            "agente": "🤖",
        }
        m_icon = mode_icons.get(self.shell_mode, "⚡")

        # Línea 1 del Prompt Propio (impresa directamente para no afectar el cálculo de ancho de readline)
        top_line = (
            f"\n{C.CYAN}┏━[{C.BOLD}{m_icon} {self.config.ai_name}{C.RESET}{C.CYAN}]"
            f"━[{C.YELLOW}📁 {short_cwd}{C.RESET}{git_tag}{C.CYAN}]"
            f"━[{status_tag}{C.CYAN}]{C.RESET}"
        )
        print(top_line)

        # Línea 2 del Prompt con códigos envueltos en \001..\002 para edición perfecta con flechas/Tab
        c_cyan = _rl_wrap(C.CYAN)
        c_bold = _rl_wrap(C.BOLD)
        c_yel = _rl_wrap(C.YELLOW)
        c_rst = _rl_wrap(C.RESET)
        prompt_line2 = f"{c_cyan}┗━{c_bold}{c_yel}❯{c_rst} "
        return input(prompt_line2)

    # ------------------------------------------------------------------
    # Ejecutor de Comandos Linux / Termux con Guardia y Doctor IA
    # ------------------------------------------------------------------

    def execute_shell_command(self, cmd: str) -> None:
        level, reason = SafetyGuard.check_bash(cmd)
        if level == "blocked":
            print_error(f"NEXUS SafetyGuard bloqueó el comando: {reason}")
            self.last_exit_code = 126
            return

        # Detectar si es un programa interactivo de pantalla completa (vim, nano, top, python interactivo)
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        prog_name = Path(tokens[0]).name if tokens else ""
        is_repl_call = prog_name in ("python", "python3", "node", "irb") and len(tokens) == 1
        is_interactive_tui = (prog_name in INTERACTIVE_PROGRAMS or is_repl_call) and not any(
            op in cmd for op in ("|", ">", "<")
        )

        t0 = time.perf_counter()
        if is_interactive_tui:
            proc = subprocess.run(cmd, shell=True)
            self.last_duration = time.perf_counter() - t0
            self.last_exit_code = proc.returncode
            set_terminal_title(f"⚡ {self.config.ai_name}-TERM · {shorten_path_mobile(Path.cwd())}")
            return

        # Ejecución estándar con streaming en vivo de stderr para poder alimentar 'reparar' si falla
        proc_p = subprocess.Popen(
            cmd,
            shell=True,
            stderr=subprocess.PIPE,
            text=True,
        )
        stderr_chunks: List[str] = []

        def _drain_stderr() -> None:
            assert proc_p.stderr is not None
            for line in proc_p.stderr:
                stderr_chunks.append(line)
                sys.stderr.write(line)
                sys.stderr.flush()

        reader = threading.Thread(target=_drain_stderr, daemon=True)
        reader.start()
        try:
            proc_p.wait()
        except KeyboardInterrupt:
            proc_p.terminate()
            print()
        reader.join(timeout=1.0)

        self.last_duration = time.perf_counter() - t0
        self.last_exit_code = proc_p.returncode if proc_p.returncode is not None else 1

        if self.last_exit_code != 0:
            self.last_failed_cmd = cmd
            self.last_failed_stderr = "".join(stderr_chunks).strip()[-1200:]
            print(
                f"  {C.DIM}💡 Salió con código {self.last_exit_code}. "
                f"Escribe {C.RESET}{C.YELLOW}reparar{C.RESET}{C.DIM} para diagnóstico IA automático.{C.RESET}"
            )

    def diagnose_last_error(self) -> None:
        """Analiza el último comando fallido en la terminal y sugiere o aplica la solución."""
        if not self.last_failed_cmd:
            print_success("No hay errores recientes de consola registrados en esta sesión.")
            return

        cmd = self.last_failed_cmd
        err = self.last_failed_stderr
        print_step("🩺", "Analizando fallo de terminal", f"`{cmd}`")

        suggestions: List[str] = [
            f"### Diagnóstico para `{cmd}`",
        ]
        if err:
            suggestions.append(f"**Salida de error capturada:**\n```text\n{err[:500]}\n```")

        err_low = err.lower()
        first_bin = cmd.split()[0] if cmd.split() else cmd

        if "not found" in err_low or "no se encontró" in err_low:
            pkg_hint = f"pkg install {first_bin}" if is_termux() else f"sudo apt install {first_bin}"
            suggestions.append(
                f"- **Causa:** El ejecutable `{first_bin}` no está instalado o no figura en `$PATH`.\n"
                f"- **Solución sugerida en tu entorno:** `{pkg_hint}`"
            )
        elif "modulenotfounderror" in err_low or "no module named" in err_low:
            m = re.search(r"No module named ['\"]([^'\"]+)['\"]", err)
            mod = m.group(1) if m else "paquete"
            suggestions.append(
                f"- **Causa:** Falta el módulo de Python `{mod}`.\n"
                f"- **Solución en Termux/Python:** `pip install {mod}` (o `pkg install python-{mod}` si tiene extensión C)."
            )
        elif "permission denied" in err_low or "permiso denegado" in err_low:
            suggestions.append(
                f"- **Causa:** Falta permiso de ejecución o acceso sobre el archivo.\n"
                f"- **Solución sugerida:** `chmod +x <archivo>` (o en Termux verifica `termux-setup-storage` para `/sdcard`)."
            )
        elif "syntaxerror" in err_low or "nameerror" in err_low or "traceback" in err_low:
            suggestions.append(
                "- **Causa:** Excepción en tiempo de ejecución del script Python.\n"
                f"- **Consejo:** Usa `@corrige el error en {cmd}` para que el Agente inspeccione el archivo."
            )
        else:
            ai_ans = self.app.text_engine.reply(
                f"Explica cómo solucionar este error al ejecutar '{cmd}': {err[:300]}"
            )
            suggestions.append(ai_ans["text"])

        print_ai_response(self.config.ai_name, "\n\n".join(suggestions), tag="Doctor de Terminal")

    # ------------------------------------------------------------------
    # Comandos Built-in Propios de NEXUS-TERM
    # ------------------------------------------------------------------

    def handle_builtin(self, raw: str) -> bool:
        """Ejecuta comandos internos de NEXUS-TERM. Retorna False para cerrar la terminal."""
        parts = raw.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        t0 = time.perf_counter()

        if cmd in ("salir", "exit", "logout"):
            print(f"{C.CYAN}⚡ Cerrando NEXUS-TERM y restaurando tu consola anterior...{C.RESET}")
            return False

        if cmd in ("clear", "cls"):
            if sys.stdout.isatty():
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
            self.draw_hud()
            self.last_exit_code = 0
            self.last_duration = time.perf_counter() - t0
            return True

        if cmd in ("hud", "panel"):
            self.draw_hud()
            self.last_exit_code = 0
            return True

        if cmd in ("cd", "..", "..."):
            if cmd == "..":
                target_str = ".."
            elif cmd == "...":
                target_str = "../.."
            elif not arg or arg == "~":
                target_str = str(Path.home())
            elif arg == "-":
                target_str = str(self.old_pwd)
                print(f"{C.GRAY}{target_str}{C.RESET}")
            else:
                target_str = os.path.expandvars(arg)

            target_path = Path(target_str).expanduser()
            if not target_path.is_absolute():
                target_path = (Path.cwd() / target_path).resolve()

            if not target_path.exists() or not target_path.is_dir():
                print_error(f"cd: no existe el directorio: {arg or target_str}")
                self.last_exit_code = 1
            else:
                self.old_pwd = Path.cwd()
                os.chdir(target_path)
                set_terminal_title(f"⚡ {self.config.ai_name}-TERM · {shorten_path_mobile(target_path)}")
                self.last_exit_code = 0
            self.last_duration = time.perf_counter() - t0
            return True

        if cmd == "pwd":
            print(Path.cwd())
            self.last_exit_code = 0
            return True

        if cmd == "export":
            if not arg:
                for k, v in sorted(os.environ.items()):
                    print(f"{C.CYAN}{k}{C.RESET}={v}")
            elif "=" in arg:
                k, v = arg.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                os.environ[k] = v
                self.custom_env[k] = v
                self._save_shell_rc()
                print_success(f"Variable exportada: {k}={v}")
            else:
                print_warning("Uso: `export NOMBRE=valor`")
            self.last_exit_code = 0
            return True

        if cmd == "unset":
            if arg:
                os.environ.pop(arg, None)
                self.custom_env.pop(arg, None)
                self._save_shell_rc()
                print_success(f"Variable eliminada: {arg}")
            self.last_exit_code = 0
            return True

        if cmd in ("alias", "aliases"):
            if not arg or cmd == "aliases":
                lines = ["# 🔗 Alias activos en NEXUS-TERM"]
                for k, v in sorted(self.aliases.items()):
                    lines.append(f"- `{k}` → `{v}`")
                print(format_markdown("\n".join(lines)))
            elif "=" in arg:
                k, v = arg.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                self.aliases[k] = v
                self._save_shell_rc()
                print_success(f"Alias guardado: `{k}` -> `{v}`")
            else:
                print_warning("Uso: `alias nombre='comando'` o `aliases` para listar.")
            self.last_exit_code = 0
            return True

        if cmd == "unalias":
            if arg in self.aliases:
                self.aliases.pop(arg)
                self._save_shell_rc()
                print_success(f"Alias `{arg}` eliminado.")
            self.last_exit_code = 0
            return True

        if cmd == "tema":
            if not arg or arg.lower() not in THEMES:
                opts = ", ".join(THEMES.keys())
                print_warning(f"Temas disponibles: {opts}. Ejemplo: `tema matrix`")
            else:
                self.theme_name = apply_theme(arg)
                self.config.terminal_theme = self.theme_name
                self.config.save()
                if sys.stdout.isatty():
                    sys.stdout.write("\033[2J\033[H")
                    sys.stdout.flush()
                self.draw_hud()
                print_success(f"Tema visual cambiado a `{self.theme_name}`.")
            self.last_exit_code = 0
            return True

        if cmd == "modo":
            valid = ("hibrido", "ia", "shell", "agente")
            choice = strip_accents(arg.lower())
            if choice not in valid:
                print_warning(
                    "Modos disponibles: `modo hibrido` (consola+IA), `modo ia` (solo chat), "
                    "`modo shell` (solo Linux), `modo agente` (misiones autónomas)."
                )
            else:
                self.shell_mode = choice
                self.config.shell_mode = choice
                self.config.save()
                print_success(f"Modo de NEXUS-TERM cambiado a: **{choice.upper()}**")
            self.last_exit_code = 0
            return True

        if cmd == "reparar":
            self.diagnose_last_error()
            self.last_exit_code = 0
            return True

        if cmd in ("historial", "history"):
            tail = self.cmd_history[-30:]
            start_i = max(1, len(self.cmd_history) - len(tail) + 1)
            for idx, item in enumerate(tail, start=start_i):
                print(f"  {C.GRAY}{idx:4d}{C.RESET}  {item}")
            self.last_exit_code = 0
            return True

        if cmd in ("ayuda", "help"):
            help_text = (
                "# ⚡ Guía de NEXUS-TERM (Tu Terminal Propia con IA)\n"
                "## 1. Consola Nativa de Sistema (Linux / Termux)\n"
                "- Ejecuta directamente `ls`, `cd`, `pwd`, `git status`, `python3`, `pkg install`, `vim`, tuberías `|` y redirecciones `>`.\n"
                "- **Built-ins propios:** `cd -` (volver atrás), `..`, `alias x='y'`, `export K=V`, `historial`, `!!` (repetir último), `clear`.\n"
                "- **Personalización en vivo:** `tema <cyber|matrix|ambar|oceano>` · `modo <hibrido|ia|shell|agente>` · `hud`.\n"
                "- **Autodiagnóstico:** Si un comando falla, escribe `reparar`.\n\n"
                "## 2. Prefijos Rápidos de 1 Tecla\n"
                "- `?pregunta` — Consultar al módulo de Texto/RAG\n"
                "- `@misión` — Lanzar el Agente Autónomo sobre la carpeta actual\n"
                "- `!comando` — Forzar ejecución directa en bash\n\n"
                "## 3. Los 5 Módulos de tu IA Propia\n"
                "- **Texto / Entrenamiento:** escribe en lenguaje natural o usa `/entrenar archivo.txt`, `/aprender dato`, `/indexar ruta`, `/modelos`\n"
                "- **Buscar:** `busca <tema>` o `/buscar <tema>`, `/local <texto>`, `/web <tema>`, `/leer <url>`\n"
                "- **Imágenes:** `dibuja <escena>` o `/imagen <prompt>`, `/filtro <bordes|cyberpunk|gris|sepia>`\n"
                "- **Video:** `haz un video de <escena 1 | escena 2>` o `/video <prompt>`, `/reproducir`\n"
                "- **Agente:** `@crea calc.py y pruébalo` o `/agente <misión>`, `/auto`, `/voz`, `/doctor`\n"
                "- Escribe `salir` o `exit` para cerrar NEXUS-TERM y volver a tu consola normal."
            )
            print(format_markdown(help_text))
            self.last_exit_code = 0
            return True

        return True

    # ------------------------------------------------------------------
    # Autocompletado Unificado con Tab (Comandos IA + Shell + Rutas)
    # ------------------------------------------------------------------

    def _get_path_executables(self) -> List[str]:
        if self._path_executables_cache is not None:
            return self._path_executables_cache
        exes = set(SHELL_BUILTINS) | set( self.aliases.keys() )
        for p_dir in os.environ.get("PATH", "").split(os.pathsep):
            d = Path(p_dir)
            if d.is_dir():
                try:
                    for item in d.iterdir():
                        if item.name and not item.name.startswith("."):
                            exes.add(item.name)
                except Exception:
                    continue
        self._path_executables_cache = sorted(exes)
        return self._path_executables_cache

    def _setup_readline(self) -> Optional[Path]:
        try:
            import readline
            from .cli import SLASH_COMMANDS

            hist_file = self.config.home_dir / "nexus_term_history.txt"
            if hist_file.exists():
                readline.read_history_file(str(hist_file))
                for i in range(1, readline.get_current_history_length() + 1):
                    item = readline.get_history_item(i)
                    if item:
                        self.cmd_history.append(item)
            readline.set_history_length(800)

            def completer(text: str, state: int) -> Optional[str]:
                buf = readline.get_line_buffer()
                stripped_lead = buf.lstrip()

                # 1. Comandos /... de IA
                if stripped_lead.startswith("/") and " " not in stripped_lead:
                    matches = [c for c in SLASH_COMMANDS if c.startswith(stripped_lead)]
                    return matches[state] if state < len(matches) else None

                # 2. Subopciones de `tema` y `modo`
                if stripped_lead.startswith("tema "):
                    opts = [k for k in THEMES.keys() if k.startswith(text)]
                    return opts[state] if state < len(opts) else None
                if stripped_lead.startswith("modo "):
                    opts = [m for m in ("hibrido", "ia", "shell", "agente") if m.startswith(text)]
                    return opts[state] if state < len(opts) else None

                # 3. Primer token sin ruta -> autocompletar comandos de consola, builtins y alias
                if " " not in stripped_lead and "/" not in text and not text.startswith("."):
                    cmds = [c for c in self._get_path_executables() if c.startswith(text)]
                    return cmds[state] if state < len(cmds) else None

                # 4. Autocompletar archivos y carpetas del directorio actual
                expanded_txt = os.path.expanduser(text) if text.startswith("~") else text
                p = Path(expanded_txt) if expanded_txt else Path(".")
                if expanded_txt.endswith("/") or not text:
                    parent = p if text else Path(".")
                    prefix = ""
                else:
                    parent = p.parent
                    prefix = p.name

                try:
                    candidates: List[str] = []
                    for item in sorted(parent.iterdir()):
                        if not item.name.startswith(prefix):
                            continue
                        if item.name.startswith(".") and not prefix.startswith("."):
                            continue
                        rel_str = str(item) if parent != Path(".") else item.name
                        if text.startswith("~"):
                            try:
                                rel_str = "~/" + str(item.resolve().relative_to(Path.home()))
                            except Exception:
                                pass
                        candidates.append(rel_str + ("/" if item.is_dir() else ""))
                    return candidates[state] if state < len(candidates) else None
                except Exception:
                    return None

            readline.set_completer_delims(" \t\n;|>&")
            readline.parse_and_bind("tab: complete")
            readline.set_completer(completer)
            return hist_file
        except Exception:
            return None

    # ------------------------------------------------------------------
    # HUD y Bucle Principal de NEXUS-TERM
    # ------------------------------------------------------------------

    def draw_hud(self) -> None:
        res = get_system_resources()
        bat_info = self.app.termux.battery_status()
        bat_pct: Optional[int] = None
        if bat_info.get("ok") and isinstance(bat_info.get("data"), dict):
            raw_pct = bat_info["data"].get("percentage")
            if isinstance(raw_pct, int):
                bat_pct = raw_pct

        print_nexus_term_hud(
            ai_name=self.config.ai_name,
            engine_label=self.app.text_engine.active_engine_label(),
            memory_count=self.app.memory.count_knowledge(),
            safe_mode=self.config.safe_mode,
            shell_mode=self.shell_mode,
            theme_name=self.theme_name,
            avail_ram_mb=res["avail_ram_mb"],
            total_ram_mb=res["total_ram_mb"],
            arch=res["arch"],
            is_termux_env=res["termux"],
            battery_pct=bat_pct,
        )

    def run(self) -> int:
        """Inicia la sesión dedicada de NEXUS-TERM en pantalla aislada y restaura al salir."""
        hist_file = self._setup_readline()
        entered_alt = False
        if self.use_alt_screen:
            entered_alt = enter_alt_screen(f"⚡ {self.config.ai_name}-TERM · {shorten_path_mobile(Path.cwd())}")

        try:
            self.draw_hud()
            while True:
                try:
                    raw_in = self._print_prompt_header_and_get_line().strip()
                except KeyboardInterrupt:
                    print(f"\n{C.DIM}(Ctrl+C · escribe 'salir' o Ctrl+D para cerrar NEXUS-TERM){C.RESET}")
                    self.last_exit_code = 130
                    continue
                except EOFError:
                    print(f"\n{C.CYAN}⚡ Cerrando NEXUS-TERM...{C.RESET}")
                    break

                if not raw_in:
                    continue

                # Expansión rápida de historial (!! y !N)
                if raw_in == "!!":
                    if not self.cmd_history:
                        print_warning("El historial aún está vacío.")
                        continue
                    raw_in = self.cmd_history[-1]
                    print(f"  {C.GRAY}↻ {raw_in}{C.RESET}")
                elif raw_in.startswith("!") and raw_in[1:].isdigit():
                    idx = int(raw_in[1:]) - 1
                    if 0 <= idx < len(self.cmd_history):
                        raw_in = self.cmd_history[idx]
                        print(f"  {C.GRAY}↻ {raw_in}{C.RESET}")
                    else:
                        print_warning(f"No existe la entrada #{idx + 1} en el historial.")
                        continue

                self.cmd_history.append(raw_in)
                kind, content = self.classify_input(raw_in)

                if kind == "builtin":
                    if not self.handle_builtin(content):
                        break
                elif kind == "slash":
                    t0 = time.perf_counter()
                    if not self.app._handle_slash_command(content):
                        break
                    self.last_duration = time.perf_counter() - t0
                    self.last_exit_code = 0
                elif kind == "shell":
                    self.execute_shell_command(content)
                elif kind == "agent":
                    t0 = time.perf_counter()
                    self.app.do_agent(content)
                    self.last_duration = time.perf_counter() - t0
                    self.last_exit_code = 0
                elif kind == "ai":
                    t0 = time.perf_counter()
                    self.app.dispatch_natural_input(content)
                    self.last_duration = time.perf_counter() - t0
                    self.last_exit_code = 0
        finally:
            if hist_file:
                try:
                    import readline

                    readline.write_history_file(str(hist_file))
                except Exception:
                    pass
            if entered_alt:
                exit_alt_screen()

        return 0

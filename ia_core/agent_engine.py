"""
Agente Autónomo Completo (Ciclo ReAct: Planificar -> Pensar -> Herramienta -> Observar -> Verificar).
Diseñado para Termux y programación:
- 14 herramientas nativas (archivos, edición de código, bash, Python, búsqueda, imagen, video, memoria y Termux:API)
- Guardia de seguridad de 3 niveles (Verde: seguro, Amarillo: confirmación en Modo Seguro, Rojo: bloqueado siempre)
- Doble planificador: ReAct con LLM GGUF local (llama.cpp) + Planificador Simbólico-Autónomo Propio 100% offline
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import IAConfig
from .image_engine import ImageEngine
from .memory import LocalMemory, strip_accents
from .search_engine import SearchEngine
from .termux_bridge import TermuxBridge
from .text_engine import TextEngine, safe_eval_math, summarize_text_extractive
from .video_engine import VideoEngine


# ---------------------------------------------------------------------------
# Guardia de Seguridad para Termux y Sistema de Archivos
# ---------------------------------------------------------------------------

class SafetyGuard:
    """Clasifica acciones del agente en VERDE (seguro), AMARILLO (confirmar) o ROJO (bloqueado)."""

    BLOCKED_PATTERNS = [
        r"\brm\s+(-[a-zA-Z]+\s+)*(/|/\*|~|~/\*|/sdcard\S*|/storage\S*|/data\S*)(?:\s|$|;|&|\|)",
        r"\brm\b.*--no-preserve-root",
        r"\bmkfs(?:\.[a-z0-9]+)?\b",
        r"\bdd\s+if=.*of=/dev/",
        r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;:",
        r">\s*/dev/(?:sd[a-z]|block/)",
        r"\bchmod\s+-R\s+777\s+/(?:\s|$)",
    ]

    READONLY_CMDS = {
        "ls", "pwd", "whoami", "uname", "date", "uptime", "df", "du", "free",
        "wc", "head", "tail", "cat", "stat", "file", "which", "echo", "env",
    }

    @classmethod
    def check_bash(cls, command: str) -> Tuple[str, str]:
        """Retorna ('blocked' | 'safe' | 'confirm', motivo)."""
        cmd = command.strip()
        for pat in cls.BLOCKED_PATTERNS:
            if re.search(pat, cmd, flags=re.I):
                return ("blocked", "Comando destructivo bloqueado por protección del dispositivo.")

        # Si no encadena operadores peligrosos y empieza con comando de solo lectura
        if not any(sep in cmd for sep in ("&&", "||", ";", "|", ">", "<", "`", "$(")):
            try:
                parts = shlex.split(cmd)
            except ValueError:
                parts = cmd.split()
            if parts:
                prog = Path(parts[0]).name
                if prog in cls.READONLY_CMDS:
                    return ("safe", "Comando de lectura/diagnóstico.")
                if prog == "git" and len(parts) > 1 and parts[1] in ("status", "log", "diff", "branch"):
                    return ("safe", "Consulta git de solo lectura.")

        return ("confirm", f"Ejecutar comando de terminal: `{cmd}`")

    @classmethod
    def check_write_path(cls, target: Path, workspace: Path) -> Tuple[str, str]:
        resolved = target.expanduser().resolve()
        ws = workspace.expanduser().resolve()
        blocked_roots = (Path("/etc"), Path("/bin"), Path("/usr/bin"), Path("/system"), Path("/proc"))
        for br in blocked_roots:
            try:
                resolved.relative_to(br)
                return ("blocked", f"Ruta crítica del sistema protegida: {resolved}")
            except ValueError:
                pass
        try:
            resolved.relative_to(ws)
            return ("safe", "Escritura dentro del directorio de trabajo.")
        except ValueError:
            return ("confirm", f"Escribir archivo fuera del directorio actual: `{resolved}`")


# ---------------------------------------------------------------------------
# Motor Principal del Agente
# ---------------------------------------------------------------------------

StepCallback = Callable[[str, str, str], None]
ConfirmCallback = Callable[[str], bool]


class AgentEngine:
    """Agente autónomo con herramientas integradas de código, sistema, multimedia y Termux:API."""

    def __init__(
        self,
        config: IAConfig,
        memory: LocalMemory,
        text_engine: TextEngine,
        search_engine: SearchEngine,
        image_engine: ImageEngine,
        video_engine: VideoEngine,
        termux_bridge: TermuxBridge,
    ):
        self.config = config
        self.memory = memory
        self.text_engine = text_engine
        self.search_engine = search_engine
        self.image_engine = image_engine
        self.video_engine = video_engine
        self.termux = termux_bridge

    # --- Herramientas ejecutables del Agente ---

    def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        workspace: Path,
        safe_mode: bool = True,
        confirm_cb: Optional[ConfirmCallback] = None,
    ) -> Dict[str, Any]:
        """Ejecuta una herramienta del agente aplicando validaciones de seguridad."""
        t = tool_name.strip().lower()

        if t == "listar_archivos":
            ruta = Path(args.get("ruta", ".")).expanduser()
            if not ruta.is_absolute():
                ruta = (workspace / ruta).resolve()
            if not ruta.exists():
                return {"ok": False, "error": f"No existe la ruta: {ruta}"}
            items = []
            for p in sorted(ruta.iterdir()):
                if p.name.startswith(".") and not args.get("ocultos", False):
                    continue
                kind = "DIR" if p.is_dir() else f"{p.stat().st_size} B"
                items.append(f"{p.name}/ ({kind})" if p.is_dir() else f"{p.name} ({kind})")
            return {"ok": True, "ruta": str(ruta), "total": len(items), "archivos": items[:60]}

        if t == "leer_archivo":
            ruta = Path(args.get("ruta", "")).expanduser()
            if not ruta.is_absolute():
                ruta = (workspace / ruta).resolve()
            if not ruta.exists() or not ruta.is_file():
                return {"ok": False, "error": f"No existe el archivo: {ruta}"}
            lines = ruta.read_text(encoding="utf-8", errors="ignore").splitlines()
            limit = int(args.get("lineas", 120))
            numbered = [f"{i+1:4d} | {ln}" for i, ln in enumerate(lines[:limit])]
            return {"ok": True, "ruta": str(ruta), "total_lineas": len(lines), "contenido": "\n".join(numbered)}

        if t == "escribir_archivo":
            raw_path = args.get("ruta", "")
            if not raw_path:
                return {"ok": False, "error": "Falta argumento 'ruta'."}
            ruta = Path(raw_path).expanduser()
            if not ruta.is_absolute():
                ruta = (workspace / ruta).resolve()
            level, reason = SafetyGuard.check_write_path(ruta, workspace)
            if level == "blocked":
                return {"ok": False, "error": reason}
            if level == "confirm" and safe_mode:
                if not confirm_cb or not confirm_cb(reason):
                    return {"ok": False, "error": "Escritura cancelada por el usuario en modo seguro."}
            ruta.parent.mkdir(parents=True, exist_ok=True)
            contenido = str(args.get("contenido", ""))
            ruta.write_text(contenido, encoding="utf-8")
            return {"ok": True, "ruta": str(ruta), "bytes": len(contenido.encode("utf-8"))}

        if t == "editar_archivo":
            ruta = Path(args.get("ruta", "")).expanduser()
            if not ruta.is_absolute():
                ruta = (workspace / ruta).resolve()
            if not ruta.exists():
                return {"ok": False, "error": f"No existe el archivo: {ruta}"}
            level, reason = SafetyGuard.check_write_path(ruta, workspace)
            if level == "blocked":
                return {"ok": False, "error": reason}
            if level == "confirm" and safe_mode and (not confirm_cb or not confirm_cb(reason)):
                return {"ok": False, "error": "Edición cancelada por el usuario."}
            old_str = str(args.get("buscar", ""))
            new_str = str(args.get("reemplazar", ""))
            current = ruta.read_text(encoding="utf-8", errors="ignore")
            if old_str not in current:
                return {"ok": False, "error": "No se encontró el fragmento exacto en el archivo."}
            updated = current.replace(old_str, new_str, 1)
            ruta.write_text(updated, encoding="utf-8")
            return {"ok": True, "ruta": str(ruta)}

        if t == "bash_exec":
            cmd = str(args.get("comando", "")).strip()
            if not cmd:
                return {"ok": False, "error": "Comando vacío."}
            level, reason = SafetyGuard.check_bash(cmd)
            if level == "blocked":
                return {"ok": False, "error": reason}
            # Permitir ejecutar python3 sobre un archivo recién creado dentro del workspace sin fricción
            is_local_py = False
            try:
                parts = shlex.split(cmd)
                if len(parts) >= 2 and Path(parts[0]).name.startswith("python"):
                    script_p = (workspace / parts[1]).resolve()
                    script_p.relative_to(workspace.resolve())
                    if script_p.exists():
                        is_local_py = True
            except Exception:
                pass

            if level == "confirm" and safe_mode and not is_local_py:
                if not confirm_cb or not confirm_cb(reason):
                    return {"ok": False, "error": "Ejecución cancelada por el usuario en modo seguro."}

            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=int(args.get("timeout", 30)),
                )
                return {
                    "ok": proc.returncode == 0,
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout.strip()[:2500],
                    "stderr": proc.stderr.strip()[:1200],
                }
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": "Tiempo límite excedido ejecutando el comando."}

        if t == "python_eval":
            code = str(args.get("codigo", ""))
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                return {
                    "ok": proc.returncode == 0,
                    "stdout": proc.stdout.strip()[:2000],
                    "stderr": proc.stderr.strip()[:1000],
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}

        if t == "buscar_info":
            q = str(args.get("consulta", ""))
            modo = str(args.get("modo", "todo"))
            res = self.search_engine.search(q, mode=modo, root_dir=workspace)
            return {
                "ok": True,
                "memoria": res["memory_hits"][:2],
                "archivos_locales": res["local_files"][:4],
                "web": res["web_hits"][:3],
                "web_status": res["web_status"],
            }

        if t == "generar_imagen":
            prompt_img = str(args.get("prompt", "Paisaje futurista"))
            out_file = args.get("salida")
            out_p = (workspace / out_file).resolve() if out_file else None
            info = self.image_engine.generate(prompt_img, out_path=out_p)
            return {
                "ok": True,
                "ruta": info["path"],
                "resolucion": f"{info['width']}x{info['height']}",
                "motor": info["engine"],
                "pixels": info["pixels"],
            }

        if t == "generar_video":
            prompt_vid = str(args.get("prompt", "Viaje espacial"))
            out_file = args.get("salida")
            out_p = (workspace / out_file).resolve() if out_file else None
            info = self.video_engine.generate(
                prompt_vid,
                duration=float(args.get("duracion", 3.0)),
                fps=int(args.get("fps", 10)),
                out_path=out_p,
            )
            return {
                "ok": True,
                "ruta": info["path"],
                "fotogramas": info["frames_count"],
                "encoder": info["encoder"],
            }

        if t == "memoria_guardar":
            titulo = str(args.get("titulo", "Nota de Agente"))
            contenido = str(args.get("contenido", ""))
            ok = self.memory.add_knowledge(title=titulo, content=contenido, source="agente")
            return {"ok": ok, "titulo": titulo}

        if t == "termux_api":
            accion = str(args.get("accion", "sistema")).lower()
            if accion == "bateria":
                return self.termux.battery_status()
            if accion == "notificacion":
                return self.termux.notify(
                    str(args.get("titulo", self.config.ai_name)),
                    str(args.get("mensaje", "Tarea completada")),
                )
            if accion == "hablar":
                return self.termux.speak(str(args.get("texto", "Listo")))
            if accion == "vibrar":
                return self.termux.vibrate(int(args.get("ms", 250)))
            if accion == "abrir":
                return self.termux.open_file(Path(str(args.get("ruta", ""))))
            return {"ok": True, "resumen": self.termux.device_summary()}

        return {"ok": False, "error": f"Herramienta desconocida: {tool_name}"}

    # --- Ejecución principal del Agente (ReAct LLM + Planificador Propio) ---

    def run(
        self,
        goal: str,
        workspace: Optional[Path] = None,
        safe_mode: Optional[bool] = None,
        step_cb: Optional[StepCallback] = None,
        confirm_cb: Optional[ConfirmCallback] = None,
    ) -> Dict[str, Any]:
        """Ejecuta una misión autónoma completa de múltiples pasos."""
        ws = (workspace or Path.cwd()).resolve()
        use_safe = self.config.safe_mode if safe_mode is None else safe_mode
        trace: List[Dict[str, Any]] = []

        def emit(kind: str, title: str, detail: str = "") -> None:
            if step_cb:
                step_cb(kind, title, detail)

        self.memory.log_message("user", f"[AGENTE] {goal}", module="agente")

        # 1. Si hay un modelo GGUF local activo con llama.cpp, ejecutamos bucle ReAct generativo
        if self.text_engine.has_local_gguf_ready():
            res = self._run_llm_react_loop(goal, ws, use_safe, emit, confirm_cb, trace)
            if res:
                self.memory.log_message("assistant", res["summary"], module="agente")
                return res

        # 2. Planificador Autónomo Propio (100% determinista/heurístico sin requerir pesos GGUF)
        res = self._run_autonomous_propio_planner(goal, ws, use_safe, emit, confirm_cb, trace)
        self.memory.log_message("assistant", res["summary"], module="agente")
        return res

    def _run_llm_react_loop(
        self,
        goal: str,
        ws: Path,
        safe_mode: bool,
        emit: StepCallback,
        confirm_cb: Optional[ConfirmCallback],
        trace: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Ciclo ReAct estándar para modelos GGUF locales."""
        sys_prompt = (
            "Eres un agente autónomo local en Termux. Responde SIEMPRE con UN solo objeto JSON por turno.\n"
            "Herramientas disponibles:\n"
            "- listar_archivos: {\"ruta\": \".\"}\n"
            "- leer_archivo: {\"ruta\": \"archivo.py\"}\n"
            "- escribir_archivo: {\"ruta\": \"script.py\", \"contenido\": \"...\"}\n"
            "- editar_archivo: {\"ruta\": \"script.py\", \"buscar\": \"...\", \"reemplazar\": \"...\"}\n"
            "- bash_exec: {\"comando\": \"python3 script.py\"}\n"
            "- buscar_info: {\"consulta\": \"tema\", \"modo\": \"todo\"}\n"
            "- generar_imagen: {\"prompt\": \"descripcion\"}\n"
            "- generar_video: {\"prompt\": \"descripcion\", \"duracion\": 3}\n"
            "- termux_api: {\"accion\": \"bateria|notificacion|hablar|sistema\", \"mensaje\": \"...\"}\n\n"
            "Formato para usar herramienta:\n"
            "{\"pensamiento\": \"...\", \"herramienta\": \"nombre\", \"args\": {...}}\n"
            "Formato cuando termines la tarea:\n"
            "{\"pensamiento\": \"...\", \"respuesta_final\": \"explicación completa en español\"}"
        )

        history = f"Objetivo del usuario: {goal}\nDirectorio actual: {ws}\n"
        created_media: List[Dict[str, Any]] = []

        for step_num in range(1, self.config.max_agent_steps + 1):
            raw = self.text_engine.generate_raw_llm(history, system_prompt=sys_prompt, max_tokens=420)
            if not raw:
                return None
            m_json = re.search(r"\{.*\}", raw, flags=re.S)
            if not m_json:
                if step_num == 1:
                    return None
                return {"summary": raw, "steps": trace, "media": created_media}

            try:
                action = json.loads(m_json.group(0))
            except Exception:
                return None

            thought = str(action.get("pensamiento", "Analizando siguiente paso..."))
            emit("thought", f"Paso {step_num}", thought)

            if "respuesta_final" in action:
                return {
                    "summary": str(action["respuesta_final"]),
                    "steps": trace,
                    "media": created_media,
                }

            tool = str(action.get("herramienta", ""))
            args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}
            emit("tool", f"Herramienta: {tool}", json.dumps(args, ensure_ascii=False)[:100])

            obs = self.execute_tool(tool, args, ws, safe_mode=safe_mode, confirm_cb=confirm_cb)
            obs_copy = {k: v for k, v in obs.items() if k != "pixels"}
            if tool in ("generar_imagen", "generar_video") and obs.get("ok"):
                created_media.append({"type": tool, **obs})

            trace.append({"step": step_num, "thought": thought, "tool": tool, "args": args, "result": obs_copy})
            emit("obs", "Resultado", "OK" if obs.get("ok") else str(obs.get("error", "Fallo")))
            history += f"\nTurno {step_num} -> Acción: {json.dumps(action, ensure_ascii=False)}\nObservación: {json.dumps(obs_copy, ensure_ascii=False)[:900]}\n"

        return {
            "summary": "Se alcanzó el número máximo de pasos del agente. Revisa los resultados ejecutados arriba.",
            "steps": trace,
            "media": created_media,
        }

    def _synthesize_python_program(self, goal: str) -> Tuple[str, str]:
        """Construye código Python funcional y verificable adaptado a la tarea pedida."""
        low = strip_accents(goal.lower())

        # Extraer nombre de archivo si el usuario lo especificó (ej. calc.py, primos.py)
        m_file = re.search(r"\b([a-zA-Z0-9_\-]+\.py)\b", goal)
        filename = m_file.group(1) if m_file else "programa_generado.py"

        if "primo" in low:
            code = (
                "#!/usr/bin/env python3\n"
                '"""Generador y verificador de números primos creado por NEXUS Agente."""\n\n'
                "import math\n\n"
                "def es_primo(n: int) -> bool:\n"
                "    if n < 2:\n"
                "        return False\n"
                "    if n in (2, 3):\n"
                "        return True\n"
                "    if n % 2 == 0:\n"
                "        return False\n"
                "    for d in range(3, int(math.isqrt(n)) + 1, 2):\n"
                "        if n % d == 0:\n"
                "            return False\n"
                "    return True\n\n"
                "def primeros_primos(hasta: int = 100) -> list[int]:\n"
                "    return [k for k in range(2, hasta + 1) if es_primo(k)]\n\n"
                "if __name__ == '__main__':\n"
                "    lista = primeros_primos(100)\n"
                "    print(f'Total de primos hasta 100: {len(lista)}')\n"
                "    print('Primos:', lista)\n"
            )
            return filename, code

        if "fibonacci" in low:
            code = (
                "#!/usr/bin/env python3\n"
                '"""Secuencia de Fibonacci optimizada creada por NEXUS Agente."""\n\n'
                "def fibonacci(n: int) -> list[int]:\n"
                "    seq = []\n"
                "    a, b = 0, 1\n"
                "    for _ in range(max(0, n)):\n"
                "        seq.append(a)\n"
                "        a, b = b, a + b\n"
                "    return seq\n\n"
                "if __name__ == '__main__':\n"
                "    datos = fibonacci(15)\n"
                "    print('Primeros 15 números de Fibonacci:', datos)\n"
            )
            return filename, code

        if "calculadora" in low:
            code = (
                "#!/usr/bin/env python3\n"
                '"""Calculadora científica CLI lista para Termux."""\n\n'
                "import math\n"
                "import sys\n\n"
                "ENTORNO = {\n"
                "    'sqrt': math.sqrt,\n"
                "    'sin': math.sin,\n"
                "    'cos': math.cos,\n"
                "    'tan': math.tan,\n"
                "    'log': math.log,\n"
                "    'round': round,\n"
                "    'abs': abs,\n"
                "    'pi': math.pi,\n"
                "    'e': math.e,\n"
                "}\n\n"
                "def evaluar(expresion: str) -> float:\n"
                "    limpia = expresion.replace('^', '**')\n"
                "    return float(eval(limpia, {'__builtins__': {}}, ENTORNO))\n\n"
                "if __name__ == '__main__':\n"
                "    if len(sys.argv) > 1:\n"
                "        expr = ' '.join(sys.argv[1:])\n"
                "        print(f'{expr} = {evaluar(expr)}')\n"
                "    else:\n"
                "        pruebas = ['2 + 2 * 10', 'sqrt(144) + 8', 'round(sin(pi / 2), 4)']\n"
                "        for p in pruebas:\n"
                "            print(f'{p} = {evaluar(p)}')\n"
            )
            return filename, code

        if "contraseña" in low or "password" in low or "clave" in low:
            code = (
                "#!/usr/bin/env python3\n"
                '"""Generador criptográficamente seguro de contraseñas para Termux."""\n\n'
                "import secrets\n"
                "import string\n\n"
                "def generar_password(longitud: int = 16) -> str:\n"
                "    alfabeto = string.ascii_letters + string.digits + '!@#$%&*-_'\n"
                "    return ''.join(secrets.choice(alfabeto) for _ in range(longitud))\n\n"
                "if __name__ == '__main__':\n"
                "    for i in range(3):\n"
                "        print(f'Clave {i+1}: {generar_password(18)}')\n"
            )
            return filename, code

        # Script multipropósito adaptable a la instrucción
        safe_doc = goal.replace('"""', '').strip()
        code = (
            "#!/usr/bin/env python3\n"
            f'"""Script generado automáticamente por NEXUS Agente: {safe_doc}"""\n\n'
            "import datetime\n"
            "import platform\n"
            "from pathlib import Path\n\n"
            "def main() -> None:\n"
            f"    print('Tarea ejecutada:', {safe_doc!r})\n"
            "    print('Sistema:', platform.system(), platform.machine())\n"
            "    print('Directorio actual:', Path.cwd())\n"
            "    print('Fecha y hora:', datetime.datetime.now().isoformat(timespec='seconds'))\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        return filename, code

    def _run_autonomous_propio_planner(
        self,
        goal: str,
        ws: Path,
        safe_mode: bool,
        emit: StepCallback,
        confirm_cb: Optional[ConfirmCallback],
        trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Planificador autónomo multi-paso propio (sin requerir GGUF externo).
        Descompone misiones complejas combinando código, terminal, archivos, búsqueda, multimedia y Termux:API.
        """
        low = strip_accents(goal.lower())
        summary_lines: List[str] = []
        created_media: List[Dict[str, Any]] = []
        step_idx = 0

        def run_step(thought: str, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal step_idx
            step_idx += 1
            emit("thought", f"Paso {step_idx}", thought)
            emit("tool", f"Herramienta: {tool}", json.dumps(args, ensure_ascii=False)[:95])
            obs = self.execute_tool(tool, args, ws, safe_mode=safe_mode, confirm_cb=confirm_cb)
            obs_clean = {k: v for k, v in obs.items() if k != "pixels"}
            trace.append({"step": step_idx, "thought": thought, "tool": tool, "args": args, "result": obs_clean})
            emit("obs", "Resultado", "Completado" if obs.get("ok") else str(obs.get("error", "Aviso")))
            if tool in ("generar_imagen", "generar_video") and obs.get("ok"):
                created_media.append({"type": tool, **obs})
            return obs

        # 1. ¿Pide ejecutar explícitamente un comando de consola? (ej. "ejecuta `uname -a`")
        m_cmd = re.search(r"`([^`]+)`", goal)
        if m_cmd and any(k in low for k in ("ejecuta", "corre", "comando", "terminal", "bash")):
            cmd_str = m_cmd.group(1)
            obs = run_step("Ejecutar comando solicitado con verificación de seguridad.", "bash_exec", {"comando": cmd_str})
            if obs.get("ok"):
                summary_lines.append(f"**Comando ejecutado (`{cmd_str}`):**\n```text\n{obs.get('stdout') or '(sin salida)'}\n```")
            else:
                summary_lines.append(f"**No se completó `{cmd_str}`:** {obs.get('error') or obs.get('stderr')}")

        # 2. ¿Pide crear/programar un script Python (y opcionalmente probarlo)?
        wants_code = any(
            k in low
            for k in ("script", "programa", "codigo", ".py", "funcion", "calculadora", "primos", "fibonacci", "contraseña", "password")
        ) and any(k in low for k in ("crea", "escribe", "haz", "genera", "programa", "desarrolla"))

        if wants_code:
            fname, py_code = self._synthesize_python_program(goal)
            w_obs = run_step(
                f"Diseñar y escribir el código fuente en `{fname}`.",
                "escribir_archivo",
                {"ruta": fname, "contenido": py_code},
            )
            if w_obs.get("ok"):
                summary_lines.append(f"- **Archivo creado:** `{w_obs['ruta']}` ({w_obs['bytes']} bytes)")
                t_obs = run_step(
                    f"Ejecutar `{fname}` para verificar que funciona sin errores.",
                    "bash_exec",
                    {"comando": f"{shlex.quote(sys.executable)} {shlex.quote(fname)}"},
                )
                if t_obs.get("ok"):
                    summary_lines.append(
                        f"- **Verificación exitosa (`exit 0`):**\n```text\n{t_obs.get('stdout', '')}\n```"
                    )
                else:
                    summary_lines.append(f"- **Error al probar `{fname}`:** `{t_obs.get('stderr') or t_obs.get('error')}`")

        # 3. ¿Pide inspeccionar / analizar carpeta o archivos locales?
        if any(k in low for k in ("analiza", "lista", "archivos", "carpeta", "directorio", "proyecto", "estructura")):
            l_obs = run_step("Inspeccionar el contenido del directorio de trabajo.", "listar_archivos", {"ruta": "."})
            if l_obs.get("ok"):
                files_preview = "\n".join(f"  - {item}" for item in l_obs["archivos"][:18])
                report_body = (
                    f"### Análisis del directorio `{l_obs['ruta']}`\n"
                    f"Total de elementos visibles: **{l_obs['total']}**\n\n"
                    f"{files_preview}"
                )
                summary_lines.append(report_body)

                # Si pide guardar reporte en un archivo .md o .txt
                m_rep = re.search(r"\b([a-zA-Z0-9_\-]+\.(?:md|txt))\b", goal)
                if m_rep:
                    rep_name = m_rep.group(1)
                    r_obs = run_step(
                        f"Guardar el informe generado en `{rep_name}`.",
                        "escribir_archivo",
                        {"ruta": rep_name, "contenido": report_body + "\n"},
                    )
                    if r_obs.get("ok"):
                        summary_lines.append(f"- **Reporte guardado en:** `{r_obs['ruta']}`")

        # 4. ¿Pide investigar / buscar un tema (y opcionalmente guardarlo en notas)?
        if any(k in low for k in ("busca", "investiga", "averigua", "informacion sobre")):
            q_match = re.search(
                r"(?:busca|investiga|averigua|informacion sobre)\s+(?:informacion sobre\s+)?(.+?)(?:\s+y\s+(?:guarda|crea|haz|genera)|$)",
                goal,
                flags=re.I,
            )
            query = q_match.group(1).strip(" .,") if q_match else goal
            s_obs = run_step(f"Buscar información local y web sobre '{query}'.", "buscar_info", {"consulta": query, "modo": "todo"})
            collected: List[str] = []
            for m_hit in s_obs.get("memoria", []):
                collected.append(f"- **{m_hit['title']}** (Memoria): {summarize_text_extractive(m_hit['content'], 2)}")
            for w_hit in s_obs.get("web", []):
                collected.append(f"- **{w_hit['title']}** ({w_hit.get('url', '')}): {w_hit.get('snippet', '')}")
            for f_hit in s_obs.get("archivos_locales", [])[:2]:
                collected.append(f"- **Archivo `{f_hit['title']}`**: {f_hit['snippet']}")

            digest_txt = "\n".join(collected) if collected else f"No se hallaron coincidencias adicionales para '{query}'."
            summary_lines.append(f"### Hallazgos de búsqueda (`{query}`)\n{digest_txt}")

            m_doc = re.search(r"\b([a-zA-Z0-9_\-]+\.(?:md|txt))\b", goal)
            if m_doc and any(k in low for k in ("guarda", "resumen", "archivo", "escribe", "notas")):
                doc_name = m_doc.group(1)
                run_step(
                    f"Guardar resumen de investigación en `{doc_name}`.",
                    "escribir_archivo",
                    {"ruta": doc_name, "contenido": f"# Investigación: {query}\n\n{digest_txt}\n"},
                )
                summary_lines.append(f"- **Resumen guardado en:** `{ws / doc_name}`")

        # 5. ¿Pide generar una imagen dentro del flujo del agente?
        if any(k in low for k in ("imagen", "dibuja", "ilustracion", "arte", "foto")):
            m_img = re.search(r"(?:imagen|dibuja|ilustracion|arte)\s+(?:de|sobre)?\s*(.+?)(?:\s+y\s+|$)", goal, flags=re.I)
            img_prompt = m_img.group(1).strip() if m_img else goal
            i_obs = run_step(f"Generar imagen local para '{img_prompt}'.", "generar_imagen", {"prompt": img_prompt})
            if i_obs.get("ok"):
                summary_lines.append(f"- **Imagen generada:** `{i_obs['ruta']}` ({i_obs['resolucion']}, {i_obs['motor']})")

        # 6. ¿Pide generar un video dentro del flujo del agente?
        if any(k in low for k in ("video", "animacion", "pelicula", "clip")):
            m_vid = re.search(r"(?:video|animacion|clip)\s+(?:de|sobre)?\s*(.+?)(?:\s+y\s+|$)", goal, flags=re.I)
            vid_prompt = m_vid.group(1).strip() if m_vid else goal
            v_obs = run_step(f"Renderizar video local para '{vid_prompt}'.", "generar_video", {"prompt": vid_prompt, "duracion": 3.0})
            if v_obs.get("ok"):
                summary_lines.append(f"- **Video generado:** `{v_obs['ruta']}` ({v_obs['fotogramas']} fotogramas, {v_obs['encoder']})")

        # 7. ¿Pide funciones de Termux:API (batería, notificación, hablar, vibrar)?
        if "bateria" in low:
            b_obs = run_step("Consultar estado de batería del dispositivo.", "termux_api", {"accion": "bateria"})
            if b_obs.get("ok"):
                summary_lines.append(f"- **Batería:** `{json.dumps(b_obs.get('data'), ensure_ascii=False)}`")
            else:
                summary_lines.append(f"- **Batería:** {b_obs.get('error')}")

        if "notificacion" in low or "notifica" in low:
            n_obs = run_step(
                "Enviar notificación nativa en Termux/Android.",
                "termux_api",
                {"accion": "notificacion", "titulo": self.config.ai_name, "mensaje": goal[:100]},
            )
            summary_lines.append(
                "- **Notificación enviada.**" if n_obs.get("ok") else f"- **Notificación:** {n_obs.get('error')}"
            )

        if any(k in low for k in ("habla", "di en voz alta", "pronuncia", "tts")):
            sp_obs = run_step("Sintetizar voz mediante Termux TTS.", "termux_api", {"accion": "hablar", "texto": goal})
            summary_lines.append("- **Voz reproducida.**" if sp_obs.get("ok") else f"- **TTS:** {sp_obs.get('error')}")

        # 8. Si ninguna regla específica se activó, resolvemos combinando cálculo/búsqueda/diagnóstico
        if not summary_lines:
            val = safe_eval_math(goal)
            if val is not None:
                summary_lines.append(f"Resultado calculado: **`{val}`**")
            else:
                sys_obs = run_step("Analizar contexto local y memoria para responder la misión.", "buscar_info", {"consulta": goal, "modo": "local"})
                reply_data = self.text_engine.reply(goal)
                summary_lines.append(reply_data["text"])
                if sys_obs.get("archivos_locales"):
                    arch_list = ", ".join(f"`{a['title']}`" for a in sys_obs["archivos_locales"][:3])
                    summary_lines.append(f"\nArchivos relacionados detectados en la carpeta: {arch_list}")

        return {
            "summary": "\n\n".join(summary_lines),
            "steps": trace,
            "media": created_media,
        }

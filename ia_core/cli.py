"""
Interfaz de Línea de Comandos (CLI) y Consola Interactiva (REPL) para Termux y Linux.
Soporta tanto subcomandos directos (`ia imagen "..."`, `ia agente "..."`)
como sesión interactiva con autocompletado Tab, historial persistente y enrutamiento natural.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

from . import __version__
from .agent_engine import AgentEngine
from .config import IAConfig, get_system_resources, is_termux
from .image_engine import ImageEngine, read_png_rgb
from .memory import LocalMemory, strip_accents
from .model_manager import GGUF_CATALOG, ModelManager
from .search_engine import SearchEngine
from .termux_bridge import TermuxBridge
from .terminal_shell import NexusTerminalShell, launch_in_new_session
from .text_engine import TextEngine
from .ui import (
    THEMES,
    C,
    apply_theme,
    confirm_prompt,
    format_markdown,
    play_frames_in_terminal,
    print_ai_response,
    print_banner,
    print_error,
    print_step,
    print_success,
    print_warning,
    progress_bar,
    render_rgb_buffer_ansi,
)
from .video_engine import VideoEngine

SLASH_COMMANDS = [
    "/ayuda",
    "/texto ",
    "/buscar ",
    "/web ",
    "/local ",
    "/leer ",
    "/imagen ",
    "/filtro ",
    "/video ",
    "/reproducir",
    "/agente ",
    "/aprender ",
    "/entrenar ",
    "/indexar ",
    "/memoria",
    "/modelos",
    "/tema ",
    "/auto",
    "/voz",
    "/abrir",
    "/doctor",
    "/limpiar",
    "/salir",
]


class NexusApplication:
    """Contenedor unificado de todos los motores de la IA Propia."""

    def __init__(self, config: Optional[IAConfig] = None):
        self.config = config or IAConfig.load()
        apply_theme(self.config.terminal_theme)
        self.memory = LocalMemory(self.config.memory_db_path)
        self.model_manager = ModelManager(self.config)
        self.text_engine = TextEngine(self.config, self.memory, self.model_manager)
        self.search_engine = SearchEngine(self.memory)
        self.image_engine = ImageEngine(self.config, self.model_manager)
        self.video_engine = VideoEngine(self.config, self.image_engine)
        self.termux = TermuxBridge()
        self.agent_engine = AgentEngine(
            config=self.config,
            memory=self.memory,
            text_engine=self.text_engine,
            search_engine=self.search_engine,
            image_engine=self.image_engine,
            video_engine=self.video_engine,
            termux_bridge=self.termux,
        )
        self.last_media_path: Optional[Path] = None
        self.last_video_frames: List[Any] = []

    # ------------------------------------------------------------------
    # Acciones de alto nivel reutilizadas por CLI directo y REPL
    # ------------------------------------------------------------------

    def do_text(self, prompt: str) -> None:
        res = self.text_engine.reply(prompt)
        print_ai_response(self.config.ai_name, res["text"], tag=res["engine"])
        if self.config.speak_responses:
            self.termux.speak(res["text"])

    def do_search(self, query: str, mode: str = "todo", synthesize: bool = True) -> None:
        print_step("🔍", f"Buscando ({mode})", query)
        res = self.search_engine.search(query, mode=mode)

        lines: List[str] = []
        if res["memory_hits"]:
            lines.append("## 🧠 Memoria Propia")
            for h in res["memory_hits"]:
                snip = h["content"].replace("\n", " ")[:180]
                lines.append(f"- **{h['title']}** *(score {h['score']})*: {snip}")

        if res["local_files"]:
            lines.append("## 📁 Archivos Locales")
            for f in res["local_files"]:
                loc = f":{f['line']}" if f["line"] else ""
                lines.append(f"- `{f['title']}{loc}` — {f['snippet']}")

        if res["web_hits"]:
            badge = "Caché Offline" if res["web_status"] == "cache_offline" else "Web en vivo"
            lines.append(f"## 🌐 Resultados Web ({badge})")
            for w in res["web_hits"]:
                lines.append(f"- **{w['title']}**\n  {w.get('url', '')}\n  {w.get('snippet', '')}")
        elif mode in ("web", "todo") and res["web_status"] == "sin_conexion":
            lines.append(
                "\n*Sin conexión web externa disponible en este momento; se mostraron resultados 100% locales.*"
            )

        if not lines:
            print_warning(f"No se encontraron resultados para '{query}'.")
            return

        print(format_markdown("\n".join(lines)))

        if synthesize and (res["web_hits"] or res["memory_hits"]):
            ctx_chunks = [h["snippet"] for h in res["web_hits"] if h.get("snippet")] + [
                h["content"][:300] for h in res["memory_hits"][:2]
            ]
            if ctx_chunks:
                ans = self.text_engine.reply(query, extra_context="\n".join(ctx_chunks))
                print_ai_response(self.config.ai_name, ans["text"], tag="Síntesis de Búsqueda")

    def do_read_url(self, url: str) -> None:
        print_step("🌐", "Extrayendo página", url)
        try:
            page = self.search_engine.fetch_clean_page(url, learn=True)
            ans = self.text_engine.reply(f"Resume: {page['content']}", extra_context=page["content"])
            print_success(f"Título: {page['title']} (indexado en memoria local)")
            print_ai_response(self.config.ai_name, ans["text"], tag="Lector Web")
        except Exception as e:
            print_error(f"No se pudo descargar la URL: {e}")

    def do_image(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        mode: Optional[str] = None,
        out_path: Optional[Path] = None,
        show_preview: Optional[bool] = None,
    ) -> Path:
        print_step("🎨", "Generando imagen local", prompt)
        info = self.image_engine.generate(
            prompt=prompt,
            width=width,
            height=height,
            out_path=out_path,
            force_mode=mode,
        )
        saved = Path(info["path"])
        self.last_media_path = saved

        preview_flag = self.config.preview_in_terminal if show_preview is None else show_preview
        if preview_flag:
            print()
            print(render_rgb_buffer_ansi(info["pixels"]))
            print()

        print_success(
            f"Imagen guardada: {C.BOLD}{saved}{C.RESET} "
            f"({info['width']}x{info['height']} px · {info['engine']} · {info['elapsed_sec']}s)"
        )
        if info.get("android_gallery_path"):
            print_success(f"Copiada a Galería de Android: {info['android_gallery_path']}")
        return saved

    def do_image_filter(self, filter_name: str, input_path: Path, out_path: Optional[Path] = None) -> None:
        print_step("🪄", f"Aplicando filtro '{filter_name}'", str(input_path))
        res = self.image_engine.apply_filter(input_path, filter_name, out_path=out_path)
        self.last_media_path = Path(res["path"])
        if self.config.preview_in_terminal:
            print()
            print(render_rgb_buffer_ansi(res["pixels"]))
            print()
        print_success(f"Imagen filtrada guardada en: {res['path']}")

    def do_video(
        self,
        prompt: str,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        play_in_term: bool = False,
        out_path: Optional[Path] = None,
    ) -> Path:
        print_step("🎬", "Sintetizando video local", prompt)
        info = self.video_engine.generate(
            prompt=prompt,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            out_path=out_path,
            progress_cb=lambda cur, tot: progress_bar(cur, tot, prefix="Fotogramas"),
        )
        saved = Path(info["path"])
        self.last_media_path = saved
        self.last_video_frames = info["frames"]

        if play_in_term:
            print_step("▶", "Reproduciendo vista previa en consola", f"{info['fps']} FPS")
            play_frames_in_terminal(info["frames"], fps=info["fps"])
        elif self.config.preview_in_terminal and info["frames"]:
            # Mostrar fotograma clave central como miniatura rápida
            mid = info["frames"][len(info["frames"]) // 2]
            print()
            print(render_rgb_buffer_ansi(mid))
            print()

        print_success(
            f"Video guardado: {C.BOLD}{saved}{C.RESET} "
            f"({info['frames_count']} frames @ {info['fps']} FPS · {info['width']}x{info['height']} · {info['encoder']} · {info['elapsed_sec']}s)"
        )
        if info.get("audio_path"):
            print_success(f"Pista de audio sintetizada: {info['audio_path']}")
        if info.get("android_gallery_path"):
            print_success(f"Copiado a Películas de Android: {info['android_gallery_path']}")
        return saved

    def do_agent(self, goal: str, auto_approve: bool = False) -> None:
        safe = False if auto_approve else self.config.safe_mode
        mode_str = "Modo Seguro (pide permiso en acciones sensibles)" if safe else "Modo Automático"
        print_step("🤖", "Iniciando Agente Autónomo", mode_str, color=C.PURPLE)

        def on_step(kind: str, title: str, detail: str) -> None:
            if kind == "thought":
                print_step("🧠", title, detail, color=C.PURPLE)
            elif kind == "tool":
                print_step("🔧", title, detail, color=C.YELLOW)
            elif kind == "obs":
                print_step("👁", title, detail, color=C.GREEN)

        res = self.agent_engine.run(
            goal=goal,
            safe_mode=safe,
            step_cb=on_step,
            confirm_cb=lambda q: confirm_prompt(q, default=False),
        )

        # Si el agente creó alguna imagen, mostrar vista previa en consola
        for m in res.get("media", []):
            if m.get("type") == "generar_imagen" and m.get("pixels") and self.config.preview_in_terminal:
                print()
                print(render_rgb_buffer_ansi(m["pixels"]))
                self.last_media_path = Path(m["ruta"])

        print_ai_response(self.config.ai_name, res["summary"], tag=f"Agente · {len(res['steps'])} pasos")

    def do_doctor(self) -> None:
        rep = self.model_manager.status_report()
        res = rep["resources"]
        lines = [
            "# 🩺 Diagnóstico del Sistema IA Propia",
            f"- **Versión:** `{__version__}`",
            f"- **Entorno:** `{'Termux (Android)' if res['termux'] else res['os']}` (`{res['arch']}`, Python `{res['python']}`)",
            f"- **CPU / Hilos IA:** `{res['cpu_cores']}` núcleos (`{res['optimal_threads']}` hilos asignados)",
            f"- **Memoria RAM:** `{res['avail_ram_mb']} MB` libres de `{res['total_ram_mb']} MB` totales",
            f"- **Almacenamiento libre:** `{res['free_disk_mb']} MB`",
            f"- **Carpeta de datos:** `{self.config.home_dir}`",
            f"- **Recuerdos en SQLite RAG:** `{self.memory.count_knowledge()}` documentos",
            f"- **Red Neuronal Propia:** `{len(self.text_engine.neural_lm.vocab)}` tokens entrenados (`pasos: {self.text_engine.neural_lm.trained_steps}`)",
            f"- **FFmpeg para MP4:** `{'Instalado ✔' if res['has_ffmpeg'] else 'No detectado (usando codificador GIF/AVI propio)'}`",
            f"- **Termux:API:** `{'Instalado ✔' if res['has_termux_api'] else 'No detectado (opcional: pkg install termux-api)'}`",
            f"- **Motor llama.cpp:** `{rep['llama_binary'] or 'No compilado aún (usa: ia modelos compilar-llama)'}`",
            f"- **Modelos GGUF locales:** `{', '.join(rep['installed_gguf']) if rep['installed_gguf'] else 'Ninguno (usando Motor Neuronal Propio)'}`",
            f"- **Modelo GGUF ideal para tu RAM:** `{rep['recommended_key']}` — *{rep['recommended_model']['name']} ({rep['recommended_model']['size_mb']} MB)*",
        ]
        print(format_markdown("\n".join(lines)))

    def do_models_menu(self, sub_args: List[str]) -> None:
        if not sub_args or sub_args[0] == "estado":
            rep = self.model_manager.status_report()
            lines = [
                "# 🧩 Gestor de Motores y Modelos Locales (Termux)",
                f"- **Cerebro actual:** `{self.text_engine.active_engine_label()}`",
                f"- **Binario llama.cpp:** `{rep['llama_binary'] or 'No instalado'}`",
                f"- **Modelos GGUF instalados:** `{', '.join(rep['installed_gguf']) or 'Ninguno'}`",
                f"- **Recomendado para tu RAM ({rep['resources']['total_ram_mb']} MB):** `{rep['recommended_key']}`",
                "\n## Catálogo GGUF Optimizado para Celulares ARM64",
            ]
            for k, meta in GGUF_CATALOG.items():
                star = " ⭐ *(Recomendado)*" if k == rep["recommended_key"] else ""
                lines.append(
                    f"- **`{k}`** — {meta['name']} (`{meta['size_mb']} MB`, RAM mín `{meta['min_ram_mb']} MB`){star}\n"
                    f"  {meta['desc']}"
                )
            lines.extend(
                [
                    "\n## Comandos disponibles",
                    "- `ia modelos compilar-llama` — Instala o compila `llama.cpp` nativo en Termux",
                    "- `ia modelos descargar <id>` — Descarga un modelo GGUF con soporte de reanudación",
                    "- `ia modelos importar <ruta.gguf>` — Usa un archivo `.gguf` que ya tengas descargado",
                    "- `ia modelos motor <auto|propio|gguf>` — Cambia el motor activo",
                ]
            )
            print(format_markdown("\n".join(lines)))
            return

        cmd = sub_args[0].lower()
        if cmd == "compilar-llama":
            print_step("⚙", "Instalando/compilando llama.cpp para tu CPU", "Esto puede tardar unos minutos...")
            try:
                bin_path = self.model_manager.build_llama_cpp()
                print_success(f"llama.cpp listo en: {bin_path}")
            except Exception as e:
                print_error(str(e))
        elif cmd == "descargar" and len(sub_args) >= 2:
            key = sub_args[1].lower()
            try:
                print_step("⬇", f"Descargando modelo {key}", "Soporta reanudación automática si se corta la red")
                dest = self.model_manager.download_gguf(
                    key,
                    progress_cb=lambda d, t: progress_bar(d, t, prefix=key),
                )
                print_success(f"Modelo instalado y activado: {dest}")
            except Exception as e:
                print_error(f"Error en descarga: {e}")
        elif cmd == "importar" and len(sub_args) >= 2:
            try:
                dest = self.model_manager.import_local_gguf(Path(sub_args[1]))
                print_success(f"Modelo GGUF vinculado y activado: {dest}")
            except Exception as e:
                print_error(str(e))
        elif cmd == "motor" and len(sub_args) >= 2:
            choice = sub_args[1].lower()
            if choice in ("auto", "propio", "gguf"):
                self.config.text_backend = choice
                self.config.save()
                print_success(f"Modo de motor de texto cambiado a: {choice} ({self.text_engine.active_engine_label()})")
            else:
                print_error("Usa: auto, propio o gguf")
        else:
            print_warning("Subcomando no reconocido. Ejecuta `ia modelos` para ver las opciones.")

    # ------------------------------------------------------------------
    # Enrutador de Lenguaje Natural + Slash Commands en Consola Interactiva
    # ------------------------------------------------------------------

    def _setup_readline(self) -> Optional[Path]:
        try:
            import readline

            hist_file = self.config.home_dir / "historial_cli.txt"
            if hist_file.exists():
                readline.read_history_file(str(hist_file))
            readline.set_history_length(500)

            def completer(text: str, state: int) -> Optional[str]:
                buf = readline.get_line_buffer()
                if buf.startswith("/"):
                    matches = [c for c in SLASH_COMMANDS if c.startswith(buf)]
                    if state < len(matches):
                        return matches[state] if text == buf else matches[state][len(buf) - len(text) :]
                # Autocompletar nombres de archivos
                p = Path(text) if text else Path(".")
                parent = p.parent if text and not text.endswith("/") else p
                prefix = p.name if text and not text.endswith("/") else ""
                try:
                    candidates = [
                        str(item) + ("/" if item.is_dir() else "")
                        for item in sorted(parent.iterdir())
                        if item.name.startswith(prefix)
                    ]
                    return candidates[state] if state < len(candidates) else None
                except Exception:
                    return None

            readline.set_completer_delims(" \t\n")
            readline.parse_and_bind("tab: complete")
            readline.set_completer(completer)
            return hist_file
        except Exception:
            return None

    def dispatch_natural_input(self, raw_in: str) -> None:
        """Enruta entradas en lenguaje natural hacia Imagen, Video, Búsqueda, Agente o Texto."""
        low = strip_accents(raw_in.lower())
        if re.match(r"^(?:dibuja|genera una imagen|crea una imagen|haz una imagen|pinta)\b", low):
            cleaned = re.sub(
                r"^(?:dibuja|genera una imagen|crea una imagen|haz una imagen|pinta)\s*(?:de|sobre|un|una)?\s*",
                "",
                raw_in,
                flags=re.I,
            ).strip()
            self.do_image(cleaned or raw_in)
        elif re.match(r"^(?:genera un video|crea un video|haz un video|anima)\b", low):
            cleaned = re.sub(
                r"^(?:genera un video|crea un video|haz un video|anima)\s*(?:de|sobre)?\s*",
                "",
                raw_in,
                flags=re.I,
            ).strip()
            self.do_video(cleaned or raw_in)
        elif re.match(r"^(?:busca en internet|busca en la web|busca|investiga)\b", low):
            cleaned = re.sub(
                r"^(?:busca en internet|busca en la web|busca|investiga)\s*(?:sobre|que es)?\s*",
                "",
                raw_in,
                flags=re.I,
            ).strip()
            self.do_search(cleaned or raw_in, mode="todo")
        elif re.match(r"^(?:modo agente|agente\s*:)", low):
            cleaned = re.sub(r"^(?:modo agente|agente)\s*[:\-]?\s*", "", raw_in, flags=re.I).strip()
            self.do_agent(cleaned or raw_in)
        elif re.match(r"^(?:crea un script|crea una calculadora|crea un programa)\b", low) and any(
            k in low for k in ("prueb", "ejecut", ".py")
        ):
            self.do_agent(raw_in)
        else:
            self.do_text(raw_in)

    def launch_terminal_shell(self, isolate_screen: Optional[bool] = None) -> int:
        """Lanza NEXUS-TERM: la Terminal/Shell propia híbrida (Comandos + IA) en pantalla aislada."""
        shell = NexusTerminalShell(self, isolate_screen=isolate_screen)
        return shell.run()

    def interactive_repl(self) -> None:
        """Consola de chat simple (sin shell híbrido)."""
        hist_file = self._setup_readline()
        print_banner(
            ai_name=self.config.ai_name,
            engine_label=self.text_engine.active_engine_label(),
            memory_count=self.memory.count_knowledge(),
            safe_mode=self.config.safe_mode,
            is_termux_env=is_termux(),
        )

        while True:
            try:
                prompt_str = f"{C.CYAN}tú{C.RESET} {C.BOLD}❯{C.RESET} "
                raw_in = input(prompt_str).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{C.GRAY}Cerrando {self.config.ai_name}. ¡Hasta pronto!{C.RESET}")
                break

            if not raw_in:
                continue

            if raw_in.startswith("/"):
                if not self._handle_slash_command(raw_in):
                    break
                continue

            self.dispatch_natural_input(raw_in)

        if hist_file:
            try:
                import readline

                readline.write_history_file(str(hist_file))
            except Exception:
                pass

    def _handle_slash_command(self, raw: str) -> bool:
        """Procesa comandos '/' en consola. Retorna False si debe salir."""
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/salir", "/exit", "/quit", "/q"):
            print(f"{C.GRAY}Sesión guardada en {self.config.home_dir}. ¡Hasta pronto!{C.RESET}")
            return False

        if cmd in ("/ayuda", "/help", "/?"):
            help_md = (
                "# ⚡ Comandos de tu IA Propia en Termux\n"
                "- `/texto <mensaje>` — Conversar, calcular, resumir o pedir código\n"
                "- `/buscar <tema>` — Búsqueda híbrida (Memoria + Archivos + Web)\n"
                "- `/web <tema>` / `/local <texto>` — Buscar solo en Web o solo en tus archivos\n"
                "- `/leer <https://...>` — Leer, resumir y aprender una página web\n"
                "- `/imagen <descripción>` — Generar imagen PNG local y verla a color en consola\n"
                "- `/filtro <bordes|cyberpunk|gris|sepia|invertir> [ruta.png]` — Editar imagen\n"
                "- `/video <escena 1 | escena 2>` — Generar video animado MP4/GIF/AVI con audio\n"
                "- `/reproducir` — Reproducir en vivo el último video en la terminal\n"
                "- `/agente <misión>` — Agente autónomo (crea código, usa bash, investiga y Termux:API)\n"
                "- `/aprender <dato>` — Guardar conocimiento nuevo y entrenar la red neuronal\n"
                "- `/entrenar <archivo.txt>` — Entrenar tu red neuronal propia con un archivo\n"
                "- `/indexar <carpeta_o_archivo>` — Indexar documentos o código en la memoria RAG\n"
                "- `/memoria` — Ver recuerdos y estadísticas de memoria\n"
                "- `/modelos` — Gestionar `llama.cpp` y modelos GGUF locales según tu RAM\n"
                "- `/auto` — Alternar Modo Seguro / Modo Automático del Agente\n"
                "- `/voz` — Activar/desactivar respuestas habladas (Termux TTS)\n"
                "- `/abrir` — Abrir la última imagen o video en Android (`termux-open`)\n"
                "- `/doctor` — Diagnóstico completo de Termux, CPU y RAM\n"
                "- `/limpiar` — Limpiar pantalla · `/salir` — Cerrar"
            )
            print(format_markdown(help_md))
            return True

        if cmd == "/limpiar":
            os.system("clear")
            return True

        if cmd == "/texto":
            self.do_text(arg)
        elif cmd == "/buscar":
            if not arg:
                print_warning("Uso: `/buscar <consulta>`")
            else:
                self.do_search(arg, mode="todo")
        elif cmd == "/web":
            if not arg:
                print_warning("Uso: `/web <consulta>`")
            else:
                self.do_search(arg, mode="web")
        elif cmd == "/local":
            if not arg:
                print_warning("Uso: `/local <consulta>`")
            else:
                self.do_search(arg, mode="local", synthesize=False)
        elif cmd == "/leer":
            if not arg.startswith("http"):
                print_warning("Uso: `/leer https://ejemplo.com`")
            else:
                self.do_read_url(arg)
        elif cmd == "/imagen":
            if not arg:
                print_warning("Uso: `/imagen Atardecer cyberpunk sobre montañas y océano`")
            else:
                self.do_image(arg)
        elif cmd == "/filtro":
            fparts = arg.split(maxsplit=1)
            if not fparts:
                print_warning("Uso: `/filtro <bordes|cyberpunk|gris|sepia|invertir> [archivo.png]`")
            else:
                fname = fparts[0]
                target = Path(fparts[1]) if len(fparts) > 1 else self.last_media_path
                if not target or not Path(target).exists():
                    print_error("Indica una ruta PNG válida o genera primero una imagen con `/imagen`.")
                else:
                    self.do_image_filter(fname, Path(target))
        elif cmd == "/video":
            if not arg:
                print_warning("Uso: `/video Amanecer en montañas | Ciudad cyberpunk | Planeta espacial`")
            else:
                self.do_video(arg)
        elif cmd == "/reproducir":
            if self.last_video_frames:
                play_frames_in_terminal(self.last_video_frames, fps=self.config.video_fps)
            else:
                print_warning("Aún no has generado ningún video en esta sesión. Usa `/video <descripción>`.")
        elif cmd == "/agente":
            if not arg:
                print_warning("Uso: `/agente Crea un script primos.py y pruébalo`")
            else:
                self.do_agent(arg)
        elif cmd == "/aprender":
            if not arg:
                print_warning("Uso: `/aprender <conocimiento que quieres enseñarle>`")
            else:
                info = self.text_engine.learn_fact_or_text(arg, title="Enseñanza directa")
                print_success(
                    f"Aprendido en RAG y Red Neuronal actualizada "
                    f"(pérdida: {info['neural_stats']['final_loss']}, vocabulario: {info['neural_stats']['vocab_size']} tokens)."
                )
        elif cmd == "/entrenar":
            if not arg:
                print_warning("Uso: `/entrenar ruta/al/archivo.txt`")
            else:
                try:
                    stats = self.text_engine.train_from_file(
                        Path(arg),
                        epochs=6,
                        progress_cb=lambda ep, tot, loss: progress_bar(ep, tot, prefix=f"Época {ep}/{tot} (loss {loss:.3f})"),
                    )
                    print_success(
                        f"Entrenamiento completado: pérdida {stats['initial_loss']} -> {stats['final_loss']} "
                        f"(perplejidad {stats['perplexity']}, {stats['indexed_chunks']} fragmentos RAG)."
                    )
                except Exception as e:
                    print_error(str(e))
        elif cmd == "/indexar":
            if not arg:
                print_warning("Uso: `/indexar <archivo_o_carpeta>`")
            else:
                fn, cn = self.memory.index_path(Path(arg))
                print_success(f"Indexados {fn} archivos ({cn} bloques de conocimiento) en la memoria vectorial.")
        elif cmd == "/memoria":
            facts = self.memory.get_facts()
            total_k = self.memory.count_knowledge()
            lines = [
                f"# 🧠 Estado de Memoria Local (`{total_k}` bloques RAG)",
                f"- **Base de datos:** `{self.config.memory_db_path}`",
                f"- **Datos recordados del usuario:** `{facts if facts else 'Ninguno aún'}`",
            ]
            print(format_markdown("\n".join(lines)))
        elif cmd == "/modelos":
            self.do_models_menu(arg.split() if arg else [])
        elif cmd == "/tema":
            if not arg or arg.lower() not in THEMES:
                print_warning(f"Temas disponibles: {', '.join(THEMES.keys())}. Ejemplo: `/tema matrix`")
            else:
                chosen = apply_theme(arg)
                self.config.terminal_theme = chosen
                self.config.save()
                print_success(f"Tema visual cambiado a: {chosen}")
        elif cmd == "/auto":
            self.config.safe_mode = not self.config.safe_mode
            self.config.save()
            state = "Modo Seguro ACTIVADO (pide confirmación)" if self.config.safe_mode else "Modo Automático ACTIVADO"
            print_success(state)
        elif cmd == "/voz":
            self.config.speak_responses = not self.config.speak_responses
            self.config.save()
            print_success(f"Respuestas por voz TTS: {'ACTIVADAS' if self.config.speak_responses else 'DESACTIVADAS'}")
        elif cmd == "/abrir":
            target = Path(arg) if arg else self.last_media_path
            if not target or not target.exists():
                print_warning("No hay archivo reciente para abrir. Especifica `/abrir ruta`.")
            else:
                r = self.termux.open_file(target)
                if r["ok"]:
                    print_success(f"Abriendo {target}...")
                else:
                    print_warning(r["error"])
        elif cmd == "/doctor":
            self.do_doctor()
        else:
            print_warning(f"Comando `{cmd}` desconocido. Escribe `/ayuda` para ver la lista completa.")

        return True


# ---------------------------------------------------------------------------
# Parser de Subcomandos CLI Directos y Lanzador de NEXUS-TERM
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia",
        description="NEXUS — Terminal Propia e Inteligencia Artificial 100% Local para Termux y Linux.",
    )
    parser.add_argument("-v", "--version", action="version", version=f"NEXUS-TERM IA Propia v{__version__}")
    parser.add_argument(
        "--tema",
        choices=list(THEMES.keys()),
        default=None,
        help="Tema visual de NEXUS-TERM (cyber, matrix, ambar, oceano)",
    )
    parser.add_argument(
        "--modo-shell",
        choices=["hibrido", "ia", "shell", "agente"],
        default=None,
        help="Modo inicial de NEXUS-TERM",
    )
    parser.add_argument(
        "--sin-aislar",
        action="store_true",
        help="No cambiar al Alternate Screen Buffer al abrir NEXUS-TERM",
    )
    parser.add_argument(
        "--nueva-sesion",
        action="store_true",
        help="Lanzar NEXUS-TERM en una sesión/ventana separada de tmux o Termux",
    )
    parser.add_argument(
        "--chat-simple",
        action="store_true",
        help="Usar el modo chat básico en lugar de la terminal completa NEXUS-TERM",
    )

    sub = parser.add_subparsers(dest="subcommand")

    # 0. Terminal propia dedicada (también es el modo por defecto al ejecutar `ia` sin subcomando)
    sub.add_parser("terminal", help="Abrir NEXUS-TERM (Terminal/Shell propia completa con IA integrada)")
    sub.add_parser("shell", help="Alias de 'terminal': abre el entorno propio NEXUS-TERM")

    # 1. Texto
    p_txt = sub.add_parser("texto", help="Generar texto, responder preguntas, calcular o resumir")
    p_txt.add_argument("mensaje", nargs="+", help="Pregunta o instrucción")

    # 2. Buscar
    p_src = sub.add_parser("buscar", help="Buscar en memoria propia, archivos locales y web sin API keys")
    p_src.add_argument("consulta", nargs="*", help="Texto a buscar")
    p_src.add_argument("--modo", choices=["todo", "local", "web"], default="todo", help="Ámbito de búsqueda")
    p_src.add_argument("--local", action="store_true", help="Atajo para buscar solo offline en archivos y memoria")
    p_src.add_argument("--web", action="store_true", help="Atajo para buscar en Wikipedia y DuckDuckGo")
    p_src.add_argument("--url", type=str, default=None, help="Extraer, resumir e indexar una URL específica")

    # 3. Imagen
    p_img = sub.add_parser("imagen", help="Generar o filtrar imágenes PNG 100% locales con vista en consola")
    p_img.add_argument("prompt", nargs="*", help="Descripción visual de la imagen")
    p_img.add_argument("--ancho", type=int, default=None, help="Ancho en píxeles (ej. 512)")
    p_img.add_argument("--alto", type=int, default=None, help="Alto en píxeles (ej. 384)")
    p_img.add_argument(
        "--modo",
        choices=["landscape", "space", "cppn", "fractal"],
        default=None,
        help="Forzar arquitectura generativa (ej. cppn para Red Neuronal CPPN)",
    )
    p_img.add_argument("--filtro", choices=["bordes", "cyberpunk", "gris", "sepia", "invertir"], default=None)
    p_img.add_argument("--entrada", type=str, default=None, help="Imagen PNG de entrada al usar --filtro")
    p_img.add_argument("-o", "--salida", type=str, default=None, help="Ruta de salida PNG")
    p_img.add_argument("--sin-vista", action="store_true", help="No imprimir vista previa ANSI en consola")

    # 4. Video
    p_vid = sub.add_parser("video", help="Generar video animado local (MP4/GIF/AVI) con física temporal y audio")
    p_vid.add_argument("prompt", nargs="+", help="Descripción o storyboard separado por '|'")
    p_vid.add_argument("--duracion", type=float, default=None, help="Duración en segundos (ej. 4.0)")
    p_vid.add_argument("--fps", type=int, default=None, help="Fotogramas por segundo (ej. 12)")
    p_vid.add_argument("--ancho", type=int, default=None, help="Ancho en píxeles")
    p_vid.add_argument("--alto", type=int, default=None, help="Alto en píxeles")
    p_vid.add_argument("--reproducir", action="store_true", help="Reproducir animación en vivo en Termux")
    p_vid.add_argument("-o", "--salida", type=str, default=None, help="Ruta de salida")

    # 5. Agente
    p_agt = sub.add_parser("agente", help="Ejecutar misión autónoma multi-paso (código, bash, web, Termux:API)")
    p_agt.add_argument("objetivo", nargs="+", help="Descripción de la misión a realizar")
    p_agt.add_argument("--auto", action="store_true", help="Desactivar confirmación manual en acciones amarillas")

    # 6. Entrenar red neuronal propia
    p_trn = sub.add_parser("entrenar", help="Entrenar la Red Neuronal Propia e indexar un archivo de texto")
    p_trn.add_argument("archivo", type=str, help="Ruta al archivo .txt / .md de entrenamiento")
    p_trn.add_argument("--epocas", type=int, default=6, help="Número de épocas de retropropagación")

    # 7. Indexar carpeta o código en memoria RAG
    p_idx = sub.add_parser("indexar", help="Indexar archivos o carpetas en la memoria vectorial local")
    p_idx.add_argument("ruta", type=str, help="Archivo o directorio a indexar")

    # 8. Gestor de modelos GGUF / llama.cpp
    p_mod = sub.add_parser("modelos", help="Gestionar motores C++ locales (llama.cpp) y modelos GGUF")
    p_mod.add_argument("accion", nargs="*", help="estado | compilar-llama | descargar <id> | importar <ruta.gguf>")

    # 9. Modo Voz Termux
    sub.add_parser("voz", help="Conversar por voz usando micrófono y altavoz de Android (Termux:API)")

    # 10. Diagnóstico
    sub.add_parser("doctor", help="Diagnóstico completo de recursos, RAM, CPU y módulos en Termux")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    app = NexusApplication()

    if not args_list:
        return app.launch_terminal_shell()

    known_subs_and_flags = {
        "terminal", "shell",
        "texto", "buscar", "imagen", "video", "agente",
        "entrenar", "indexar", "modelos", "voz", "doctor",
        "-h", "--help", "-v", "--version",
        "--tema", "--modo-shell", "--sin-aislar", "--nueva-sesion", "--chat-simple",
    }
    # Si el primer argumento no es un subcomando o flag conocido (ej. `./ia "Hola, cuánto es 25*16"`), tratarlo como texto directo
    if not args_list[0].startswith("-") and args_list[0] not in known_subs_and_flags:
        app.do_text(" ".join(args_list))
        return 0

    parser = build_parser()
    ns = parser.parse_args(args_list)

    if ns.tema:
        app.config.terminal_theme = apply_theme(ns.tema)
        app.config.save()

    if ns.modo_shell:
        app.config.shell_mode = ns.modo_shell
        app.config.save()

    if ns.nueva_sesion:
        from .config import get_repo_root

        exe_file = get_repo_root() / "ia"
        if launch_in_new_session(exe_file):
            print_success("NEXUS-TERM abierto en una nueva sesión/ventana independiente.")
            return 0
        print_warning("tmux / RunCommandService no detectado; abriendo en pantalla aislada (Alternate Screen Buffer)...")

    if ns.subcommand in (None, "terminal", "shell"):
        if ns.chat_simple:
            app.interactive_repl()
            return 0
        isolate = False if ns.sin_aislar else None
        return app.launch_terminal_shell(isolate_screen=isolate)

    if ns.subcommand == "texto":
        app.do_text(" ".join(ns.mensaje))
    elif ns.subcommand == "buscar":
        if ns.url:
            app.do_read_url(ns.url)
        else:
            q = " ".join(ns.consulta).strip()
            if not q:
                print_error("Especifica una consulta o usa `--url https://...`")
                return 1
            mode = "local" if ns.local else ("web" if ns.web else ns.modo)
            app.do_search(q, mode=mode)
    elif ns.subcommand == "imagen":
        out_p = Path(ns.salida) if ns.salida else None
        if ns.filtro:
            if not ns.entrada:
                print_error("Con `--filtro` debes indicar `--entrada ruta.png`.")
                return 1
            app.do_image_filter(ns.filtro, Path(ns.entrada), out_path=out_p)
        else:
            prompt_str = " ".join(ns.prompt).strip()
            if not prompt_str:
                print_error("Especifica una descripción de la imagen. Ejemplo: `ia imagen 'Ciudad cyberpunk de noche'`")
                return 1
            app.do_image(
                prompt=prompt_str,
                width=ns.ancho,
                height=ns.alto,
                mode=ns.modo,
                out_path=out_p,
                show_preview=not ns.sin_vista,
            )
    elif ns.subcommand == "video":
        out_p = Path(ns.salida) if ns.salida else None
        app.do_video(
            prompt=" ".join(ns.prompt),
            duration=ns.duracion,
            fps=ns.fps,
            width=ns.ancho,
            height=ns.alto,
            play_in_term=ns.reproducir,
            out_path=out_p,
        )
    elif ns.subcommand == "agente":
        app.do_agent(" ".join(ns.objetivo), auto_approve=ns.auto)
    elif ns.subcommand == "entrenar":
        stats = app.text_engine.train_from_file(
            Path(ns.archivo),
            epochs=ns.epocas,
            progress_cb=lambda ep, tot, loss: progress_bar(ep, tot, prefix=f"Época {ep}/{tot} (loss {loss:.3f})"),
        )
        print_success(
            f"Red Neuronal Propia entrenada: pérdida {stats['initial_loss']} -> {stats['final_loss']} "
            f"(perplejidad {stats['perplexity']}, vocabulario {stats['vocab_size']} tokens)."
        )
    elif ns.subcommand == "indexar":
        fn, cn = app.memory.index_path(Path(ns.ruta))
        print_success(f"Indexados {fn} archivos ({cn} fragmentos) en memoria vectorial SQLite.")
    elif ns.subcommand == "modelos":
        app.do_models_menu(ns.accion)
    elif ns.subcommand == "voz":
        print_step("🎙", "Escuchando micrófono (Termux:API)", "Habla ahora...")
        stt = app.termux.listen()
        if not stt["ok"] or not stt.get("stdout"):
            print_error(stt.get("error") or "No se detectó voz.")
            return 1
        user_spoken = stt["stdout"]
        print_step("🗣", "Dijiste", user_spoken)
        res = app.text_engine.reply(user_spoken)
        print_ai_response(app.config.ai_name, res["text"], tag=res["engine"])
        app.termux.speak(res["text"])
    elif ns.subcommand == "doctor":
        app.do_doctor()

    return 0

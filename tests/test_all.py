"""
Suite completa de pruebas automatizadas para NEXUS (IA Propia 100% Local).
Verifica:
1. Texto: cálculo AST, memoria del usuario, RAG, resumen extractivo y entrenamiento de Red Neuronal Propia
2. Buscar: búsqueda en archivos locales y memoria vectorial
3. Imágenes: generación semántica, espacio 3D, fractal, red neuronal CPPN, lectura/escritura PNG y filtros
4. Video: animación multi-escena con transiciones, audio WAV sintetizado y codificadores GIF89a LZW + RIFF AVI
5. Agente: guardia de seguridad (bloqueo de comandos destructivos) + creación, ejecución y verificación autónoma de código
6. Ejecutable CLI `./ia`: subcomandos directos desde terminal
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ia_core.agent_engine import AgentEngine, SafetyGuard
from ia_core.config import IAConfig
from ia_core.image_engine import ImageEngine, read_png_rgb
from ia_core.memory import LocalMemory
from ia_core.model_manager import ModelManager
from ia_core.search_engine import SearchEngine
from ia_core.termux_bridge import TermuxBridge
from ia_core.text_engine import TextEngine
from ia_core.video_engine import VideoEngine


class TestNexusIAPropia(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.home_path = Path(self.tmp_dir.name) / "ia_home"
        os.environ["IA_HOME"] = str(self.home_path)
        self.cfg = IAConfig.load()
        self.cfg.preview_in_terminal = False
        self.cfg.save()

        self.memory = LocalMemory(self.cfg.memory_db_path)
        self.models = ModelManager(self.cfg)
        self.text_eng = TextEngine(self.cfg, self.memory, self.models)
        self.search_eng = SearchEngine(self.memory)
        self.img_eng = ImageEngine(self.cfg, self.models)
        self.vid_eng = VideoEngine(self.cfg, self.img_eng)
        self.termux = TermuxBridge()
        self.agent = AgentEngine(
            self.cfg,
            self.memory,
            self.text_eng,
            self.search_eng,
            self.img_eng,
            self.vid_eng,
            self.termux,
        )

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_text_math_memory_and_neural_training(self) -> None:
        # Cálculo seguro AST
        r_math = self.text_eng.reply("¿Cuánto es sqrt(144) * 5 + 10?")
        self.assertIn("70", r_math["text"])

        # Memoria de identidad del usuario
        self.text_eng.reply("Me llamo Valentin")
        r_who = self.text_eng.reply("¿Cómo me llamo?")
        self.assertIn("Valentin", r_who["text"])

        # Aprendizaje nuevo + entrenamiento neuronal con descenso de pérdida
        corpus_custom = Path(self.tmp_dir.name) / "notas_propias.txt"
        corpus_custom.write_text(
            "El proyecto Quasar es un motor espacial autónomo creado en Maldonado.\n"
            "El proyecto Quasar utiliza propulsión iónica eficiente y paneles solares.\n",
            encoding="utf-8",
        )
        stats = self.text_eng.train_from_file(corpus_custom, epochs=5)
        self.assertLessEqual(stats["final_loss"], stats["initial_loss"])
        self.assertGreaterEqual(stats["indexed_chunks"], 1)

        # Consulta RAG sobre el conocimiento recién entrenado
        r_rag = self.text_eng.reply("¿Qué utiliza el proyecto Quasar?")
        self.assertIn("Quasar", r_rag["text"])

    def test_02_search_local_and_memory(self) -> None:
        ws = Path(self.tmp_dir.name) / "proyecto"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "sensor_termux.py").write_text(
            "def leer_temperatura_bateria():\n    return 31.5\n",
            encoding="utf-8",
        )
        res = self.search_eng.search("temperatura_bateria", mode="local", root_dir=ws)
        self.assertGreaterEqual(len(res["local_files"]), 1)
        self.assertEqual(res["local_files"][0]["title"], "sensor_termux.py")

        res_mem = self.search_eng.search("velocidad de la luz", mode="local", root_dir=ws)
        self.assertGreaterEqual(len(res_mem["memory_hits"]), 1)

    def test_03_image_generation_png_codec_and_filters(self) -> None:
        for mode_prompt in (
            ("Ciudad cyberpunk de noche con reflejo en el mar", None),
            ("Planeta con anillos en una nebulosa espacial", "space"),
            ("Arte neuronal abstracto de energia", "cppn"),
            ("Caleidoscopio fractal matematico", "fractal"),
        ):
            prompt, fmode = mode_prompt
            out = self.img_eng.generate(prompt, width=96, height=64, force_mode=fmode)
            img_path = Path(out["path"])
            self.assertTrue(img_path.exists())
            self.assertGreater(img_path.stat().st_size, 200)
            decoded = read_png_rgb(img_path)
            self.assertEqual(len(decoded), 64)
            self.assertEqual(len(decoded[0]), 96)

        # Probar filtro Sobel de bordes sobre la última imagen
        filtered = self.img_eng.apply_filter(img_path, "bordes")
        self.assertTrue(Path(filtered["path"]).exists())

    def test_04_video_storyboard_gif_avi_and_wav(self) -> None:
        vid = self.vid_eng.generate(
            "Amanecer en montañas | Ciudad cyberpunk nocturna",
            duration=1.5,
            fps=6,
            width=96,
            height=64,
        )
        main_path = Path(vid["path"])
        self.assertTrue(main_path.exists())
        self.assertEqual(vid["frames_count"], 9)
        self.assertTrue(Path(vid["gif_path"]).exists())
        # Verificar cabecera GIF89a
        gif_bytes = Path(vid["gif_path"]).read_bytes()
        self.assertEqual(gif_bytes[:6], b"GIF89a")
        if vid.get("audio_path"):
            wav_p = Path(vid["audio_path"])
            self.assertTrue(wav_p.exists())
            self.assertEqual(wav_p.read_bytes()[:4], b"RIFF")

    def test_05_agent_safety_and_autonomous_coding(self) -> None:
        # 1. Verificar que comandos destructivos están bloqueados siempre
        lvl, _ = SafetyGuard.check_bash("rm -rf /")
        self.assertEqual(lvl, "blocked")
        lvl_sd, _ = SafetyGuard.check_bash("rm -rf /sdcard")
        self.assertEqual(lvl_sd, "blocked")

        # 2. Misión autónoma: crear primos.py y ejecutarlo
        work = Path(self.tmp_dir.name) / "agent_ws"
        work.mkdir(parents=True, exist_ok=True)
        res = self.agent.run(
            "Crea un script mis_primos.py que calcule números primos y pruébalo",
            workspace=work,
            safe_mode=True,
        )
        created_py = work / "mis_primos.py"
        self.assertTrue(created_py.exists())
        self.assertIn("Primos", res["summary"])
        self.assertGreaterEqual(len(res["steps"]), 2)

    def test_06_cli_executable_end_to_end(self) -> None:
        env = os.environ.copy()
        env["IA_HOME"] = str(self.home_path)
        exe = str(REPO_ROOT / "ia")

        p_doc = subprocess.run([sys.executable, exe, "doctor"], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(p_doc.returncode, 0)
        self.assertIn("Diagnóstico", p_doc.stdout)

        p_txt = subprocess.run([sys.executable, exe, "Hola"], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(p_txt.returncode, 0)
        self.assertIn("NEXUS", p_txt.stdout)

    def test_07_nexus_terminal_hybrid_shell(self) -> None:
        from ia_core.cli import NexusApplication
        from ia_core.terminal_shell import NexusTerminalShell

        app = NexusApplication(config=self.cfg)
        shell = NexusTerminalShell(app, isolate_screen=False)

        # Clasificador híbrido (Comandos OS vs IA vs Agente vs Builtins)
        self.assertEqual(shell.classify_input("ls -la")[0], "shell")
        self.assertEqual(shell.classify_input("git status")[0], "shell")
        self.assertEqual(shell.classify_input("/bin/ls")[0], "shell")
        self.assertEqual(shell.classify_input("!echo forzado")[0], "shell")
        self.assertEqual(shell.classify_input("cd /tmp")[0], "builtin")
        self.assertEqual(shell.classify_input("tema matrix")[0], "builtin")
        self.assertEqual(shell.classify_input("reparar")[0], "builtin")
        self.assertEqual(shell.classify_input("/imagen galaxia")[0], "slash")
        self.assertEqual(shell.classify_input("?cuánto es 5+5")[0], "ai")
        self.assertEqual(shell.classify_input("¿Qué es un agujero negro?")[0], "ai")
        self.assertEqual(shell.classify_input("dibuja una montaña")[0], "ai")
        self.assertEqual(shell.classify_input("@crea calc.py y pruébalo")[0], "agent")

        # Built-ins: alias + export + cambio de directorio y cd -
        orig_cwd = Path.cwd()
        try:
            sub_dir = Path(self.tmp_dir.name) / "sub_shell_dir"
            sub_dir.mkdir(parents=True, exist_ok=True)
            shell.handle_builtin(f"cd {sub_dir}")
            self.assertEqual(Path.cwd(), sub_dir.resolve())
            shell.handle_builtin("cd -")
            self.assertEqual(Path.cwd(), orig_cwd.resolve())

            shell.handle_builtin("alias hola_nx='echo NEXUS_OK'")
            kind, expanded = shell.classify_input("hola_nx")
            self.assertEqual(kind, "shell")
            self.assertEqual(expanded, "echo NEXUS_OK")
        finally:
            os.chdir(orig_cwd)

    def test_08_nexus_term_end_to_end_session(self) -> None:
        env = os.environ.copy()
        env["IA_HOME"] = str(self.home_path)
        exe = str(REPO_ROOT / "ia")

        session_script = "\n".join(
            [
                "tema matrix",
                "echo PRUEBA_SHELL_NATIVO_123",
                "¿Cuánto es 12 * 12?",
                "comando_inexistente_xyz --flag",
                "salir",
            ]
        ) + "\n"
        proc = subprocess.run(
            [sys.executable, exe],
            input=session_script,
            env=env,
            capture_output=True,
            text=True,
            timeout=25,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("NEXUS-TERM", proc.stdout)
        self.assertIn("PRUEBA_SHELL_NATIVO_123", proc.stdout)
        self.assertIn("144", proc.stdout)
        self.assertIn("Cerrando NEXUS-TERM", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

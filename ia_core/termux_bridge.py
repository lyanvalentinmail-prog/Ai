"""
Puente nativo con Termux:API y utilidades del sistema Android/Linux.
Permite controlar notificaciones, voz TTS/STT, batería, portapapeles, vibración,
linterna y apertura de archivos multimedia en la galería de Android.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_system_resources, is_termux


class TermuxBridge:
    """Interfaz segura con herramientas nativas de Termux:API y fallbacks Linux."""

    def _run_cmd(self, args: list[str], input_text: Optional[str] = None, timeout: int = 15) -> Dict[str, Any]:
        binary = shutil.which(args[0])
        if not binary:
            return {
                "ok": False,
                "error": f"Comando '{args[0]}' no disponible. En Termux instala: pkg install termux-api (y la app Termux:API desde F-Droid).",
            }
        try:
            proc = subprocess.run(
                [binary] + args[1:],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "ok": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Tiempo agotado ejecutando '{args[0]}'."}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def battery_status(self) -> Dict[str, Any]:
        """Consulta el estado de la batería en Android/Termux o portátiles Linux."""
        if shutil.which("termux-battery-status"):
            res = self._run_cmd(["termux-battery-status"])
            if res["ok"] and res.get("stdout"):
                try:
                    data = json.loads(res["stdout"])
                    return {"ok": True, "source": "termux-api", "data": data}
                except Exception:
                    pass

        # Fallback Linux /sys/class/power_supply
        ps_root = Path("/sys/class/power_supply")
        if ps_root.exists():
            for bat in ps_root.glob("BAT*"):
                try:
                    cap = (bat / "capacity").read_text().strip()
                    status = (bat / "status").read_text().strip()
                    return {
                        "ok": True,
                        "source": "sysfs",
                        "data": {"percentage": int(cap), "status": status},
                    }
                except Exception:
                    continue

        return {
            "ok": False,
            "error": "Sensor de batería no accesible (instala termux-api en Android).",
        }

    def notify(self, title: str, content: str) -> Dict[str, Any]:
        """Muestra una notificación nativa en la barra de Android."""
        if shutil.which("termux-notification"):
            return self._run_cmd(["termux-notification", "--title", title, "--content", content])
        if shutil.which("notify-send"):
            return self._run_cmd(["notify-send", title, content])
        return {"ok": False, "error": "Notificaciones nativas no disponibles (requiere termux-notification)."}

    def speak(self, text: str, lang: str = "es") -> Dict[str, Any]:
        """Convierte texto a voz hablada (TTS) usando el motor de Android o espeak."""
        clean = text[:600]
        if shutil.which("termux-tts-speak"):
            return self._run_cmd(["termux-tts-speak", "-l", lang, clean], timeout=30)
        for cmd in ("espeak-ng", "espeak"):
            if shutil.which(cmd):
                return self._run_cmd([cmd, "-v", lang, clean], timeout=30)
        return {"ok": False, "error": "TTS no disponible (en Termux instala: pkg install termux-api)."}

    def listen(self) -> Dict[str, Any]:
        """Reconocimiento de voz a texto (STT) mediante micrófono Android."""
        if shutil.which("termux-speech-to-text"):
            return self._run_cmd(["termux-speech-to-text"], timeout=25)
        return {"ok": False, "error": "Reconocimiento de voz requiere 'termux-speech-to-text' (Termux:API)."}

    def clipboard_get(self) -> Dict[str, Any]:
        if shutil.which("termux-clipboard-get"):
            return self._run_cmd(["termux-clipboard-get"])
        if shutil.which("wl-paste"):
            return self._run_cmd(["wl-paste"])
        if shutil.which("xclip"):
            return self._run_cmd(["xclip", "-o", "-selection", "clipboard"])
        return {"ok": False, "error": "Portapapeles no disponible."}

    def clipboard_set(self, text: str) -> Dict[str, Any]:
        if shutil.which("termux-clipboard-set"):
            return self._run_cmd(["termux-clipboard-set"], input_text=text)
        return {"ok": False, "error": "Escritura de portapapeles requiere termux-clipboard-set."}

    def vibrate(self, duration_ms: int = 250) -> Dict[str, Any]:
        return self._run_cmd(["termux-vibrate", "-d", str(max(50, min(3000, duration_ms)))])

    def torch(self, enable: bool = True) -> Dict[str, Any]:
        state = "on" if enable else "off"
        return self._run_cmd(["termux-torch", state])

    def open_file(self, file_path: Path) -> Dict[str, Any]:
        """Abre una imagen, video o documento con la app nativa de Android."""
        p = str(Path(file_path).expanduser().resolve())
        if shutil.which("termux-open"):
            return self._run_cmd(["termux-open", p])
        if shutil.which("xdg-open"):
            return self._run_cmd(["xdg-open", p])
        return {"ok": False, "error": f"Archivo guardado en {p} (instala termux-tools para abrirlo con termux-open)."}

    def device_summary(self) -> Dict[str, Any]:
        res = get_system_resources()
        bat = self.battery_status()
        return {
            "termux": is_termux(),
            "resources": res,
            "battery": bat.get("data") if bat.get("ok") else None,
        }

"""
Gestión de configuración, detección de hardware y rutas para Termux y Linux.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def is_termux() -> bool:
    """Detecta si el proceso corre dentro de Termux en Android."""
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix:
        return True
    if os.environ.get("TERMUX_VERSION"):
        return True
    return Path("/data/data/com.termux/files/usr").exists()


def get_system_resources() -> Dict[str, Any]:
    """Inspecciona CPU, arquitectura, RAM total/disponible y espacio libre."""
    cpu_count = os.cpu_count() or 2
    arch = platform.machine() or "unknown"
    total_ram_mb = 2048
    avail_ram_mb = 1024

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            raw = meminfo.read_text(encoding="utf-8", errors="ignore")
            for line in raw.splitlines():
                if line.startswith("MemTotal:"):
                    total_ram_mb = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail_ram_mb = int(line.split()[1]) // 1024
        except Exception:
            pass

    # En procesadores móviles ARM big.LITTLE conviene usar entre 2 y 6 hilos reales
    optimal_threads = max(1, min(cpu_count - 1 if cpu_count > 2 else cpu_count, 6))

    home = get_data_home()
    try:
        disk = shutil.disk_usage(home if home.exists() else Path.home())
        free_disk_mb = disk.free // (1024 * 1024)
    except Exception:
        free_disk_mb = 0

    return {
        "termux": is_termux(),
        "arch": arch,
        "os": platform.system(),
        "python": platform.python_version(),
        "cpu_cores": cpu_count,
        "optimal_threads": optimal_threads,
        "total_ram_mb": total_ram_mb,
        "avail_ram_mb": avail_ram_mb,
        "free_disk_mb": free_disk_mb,
        "has_ffmpeg": shutil.which("ffmpeg") is not None,
        "has_termux_api": shutil.which("termux-battery-status") is not None
        or shutil.which("termux-notification") is not None,
    }


def get_data_home() -> Path:
    """Retorna la carpeta base de datos de la IA (~/.ia_propia o $IA_HOME)."""
    env_home = os.environ.get("IA_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".ia_propia").resolve()


def get_repo_root() -> Path:
    """Retorna la raíz del repositorio donde vive ia_core."""
    return Path(__file__).resolve().parent.parent


def get_android_media_dirs() -> Dict[str, Optional[Path]]:
    """Detecta carpetas multimedia compartidas de Android si tienen permisos."""
    candidates = [
        Path.home() / "storage" / "shared",
        Path("/sdcard"),
        Path("/storage/emulated/0"),
    ]
    for base in candidates:
        try:
            if base.exists() and os.access(base, os.W_OK):
                pics = base / "Pictures" / "IA_Propia"
                movies = base / "Movies" / "IA_Propia"
                return {"pictures": pics, "movies": movies}
        except Exception:
            continue
    return {"pictures": None, "movies": None}


@dataclass
class IAConfig:
    """Configuración persistente del sistema IA Propia."""

    ai_name: str = "NEXUS"
    user_name: str = "Usuario"
    language: str = "es"
    # Motor de texto: "auto" (usa GGUF si existe, si no motor propio), "gguf", "propio"
    text_backend: str = "auto"
    active_gguf_model: str = ""
    active_sd_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    context_size: int = 2048
    threads: int = field(default_factory=lambda: get_system_resources()["optimal_threads"])
    # Imagen y video
    image_width: int = 512
    image_height: int = 384
    preview_in_terminal: bool = True
    video_width: int = 384
    video_height: int = 256
    video_fps: int = 12
    video_duration: float = 4.0
    video_with_audio: bool = True
    mirror_to_android_gallery: bool = True
    # Agente y seguridad
    safe_mode: bool = True
    max_agent_steps: int = 8
    speak_responses: bool = False

    @property
    def home_dir(self) -> Path:
        return get_data_home()

    @property
    def models_llm_dir(self) -> Path:
        return self.home_dir / "models" / "llm"

    @property
    def models_sd_dir(self) -> Path:
        return self.home_dir / "models" / "sd"

    @property
    def brain_dir(self) -> Path:
        return self.home_dir / "brain"

    @property
    def bin_dir(self) -> Path:
        return self.home_dir / "bin"

    @property
    def images_dir(self) -> Path:
        return self.home_dir / "outputs" / "imagenes"

    @property
    def videos_dir(self) -> Path:
        return self.home_dir / "outputs" / "videos"

    @property
    def memory_db_path(self) -> Path:
        return self.home_dir / "memoria.db"

    @property
    def config_file_path(self) -> Path:
        return self.home_dir / "config.json"

    def ensure_dirs(self) -> None:
        """Crea toda la estructura de directorios necesaria."""
        for folder in (
            self.home_dir,
            self.models_llm_dir,
            self.models_sd_dir,
            self.brain_dir,
            self.bin_dir,
            self.images_dir,
            self.videos_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Guarda la configuración en JSON."""
        self.ensure_dirs()
        self.config_file_path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> "IAConfig":
        """Carga la configuración existente o crea una nueva optimizada."""
        cfg = cls()
        cfg.ensure_dirs()
        if cfg.config_file_path.exists():
            try:
                raw = json.loads(cfg.config_file_path.read_text(encoding="utf-8"))
                valid_keys = {f.name for f in cfg.__dataclass_fields__.values()}
                for k, v in raw.items():
                    if k in valid_keys:
                        setattr(cfg, k, v)
            except Exception:
                pass
        else:
            cfg.save()
        return cfg

    def mirror_media_if_android(self, file_path: Path, media_type: str = "image") -> Optional[Path]:
        """Copia imágenes o videos a la Galería de Android (/sdcard/Pictures o Movies) si es accesible."""
        if not self.mirror_to_android_gallery:
            return None
        dirs = get_android_media_dirs()
        target_dir = dirs["pictures"] if media_type == "image" else dirs["movies"]
        if target_dir is None:
            return None
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            dest = target_dir / file_path.name
            shutil.copy2(file_path, dest)
            return dest
        except Exception:
            return None

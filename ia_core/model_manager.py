"""
Gestor de Motores C++ Locales (llama.cpp / stable-diffusion.cpp) y Modelos GGUF para Termux.
Permite:
- Detectar cuánta RAM tiene el celular y recomendar el modelo ideal
- Descargar modelos GGUF con reanudación automática (HTTP Range)
- Compilar/instalar llama.cpp y stable-diffusion.cpp en ARM64/Linux
- Importar cualquier archivo .gguf que el usuario ya tenga en Descargas
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import IAConfig, get_system_resources, is_termux

GGUF_CATALOG: Dict[str, Dict[str, Any]] = {
    "smollm2-360m": {
        "name": "SmolLM2 360M Instruct (Q8_0)",
        "filename": "SmolLM2-360M-Instruct-Q8_0.gguf",
        "size_mb": 386,
        "min_ram_mb": 1500,
        "url": "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q8_0.gguf",
        "desc": "Ultraligero y rapidísimo para cualquier celular Android.",
    },
    "qwen2.5-0.5b": {
        "name": "Qwen2.5 0.5B Instruct (Q4_K_M)",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size_mb": 398,
        "min_ram_mb": 1800,
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "desc": "Muy buen español y velocidad instantánea en Termux.",
    },
    "qwen2.5-1.5b": {
        "name": "Qwen2.5 1.5B Instruct (Q4_K_M)",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_mb": 1120,
        "min_ram_mb": 3200,
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "desc": "Equilibrio ideal calidad/velocidad en teléfonos de 4GB+ RAM (Español + Código).",
    },
    "qwen2.5-3b": {
        "name": "Qwen2.5 3B Instruct (Q4_K_M)",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_mb": 2050,
        "min_ram_mb": 5500,
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "desc": "Alta capacidad de razonamiento y Agente ReAct para celulares de 6GB-12GB RAM.",
    },
}


class ModelManager:
    """Administra binarios de inferencia local y modelos cuantizados."""

    def __init__(self, config: IAConfig):
        self.config = config
        self.config.ensure_dirs()

    def find_llama_binary(self) -> Optional[Path]:
        """Busca llama-cli o main compilado localmente o instalado en el sistema."""
        candidates = [
            self.config.bin_dir / "llama-cli",
            self.config.bin_dir / "main",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        for cmd in ("llama-cli", "llama"):
            found = shutil.which(cmd)
            if found:
                return Path(found)
        return None

    def find_sd_binary(self) -> Optional[Path]:
        """Busca el binario local 'sd' de stable-diffusion.cpp."""
        local_sd = self.config.bin_dir / "sd"
        if local_sd.exists() and local_sd.is_file():
            return local_sd
        for cmd in ("sd", "stable-diffusion"):
            found = shutil.which(cmd)
            if found:
                return Path(found)
        return None

    def list_installed_gguf(self) -> List[Path]:
        """Lista todos los modelos .gguf instalados localmente."""
        if not self.config.models_llm_dir.exists():
            return []
        return sorted(self.config.models_llm_dir.glob("*.gguf"))

    def list_installed_sd_models(self) -> List[Path]:
        """Lista modelos de difusión locales (.safetensors, .ckpt, .gguf en models/sd)."""
        if not self.config.models_sd_dir.exists():
            return []
        out: List[Path] = []
        for ext in ("*.gguf", "*.safetensors", "*.ckpt"):
            out.extend(self.config.models_sd_dir.glob(ext))
        return sorted(out)

    def get_active_gguf_path(self) -> Optional[Path]:
        """Obtiene la ruta al modelo GGUF activo si existe."""
        if self.config.active_gguf_model:
            p = Path(self.config.active_gguf_model)
            if not p.is_absolute():
                p = self.config.models_llm_dir / p
            if p.exists():
                return p
        installed = self.list_installed_gguf()
        if installed:
            self.config.active_gguf_model = installed[0].name
            self.config.save()
            return installed[0]
        return None

    def recommend_model_key(self) -> str:
        """Recomienda el mejor modelo GGUF según la RAM total del teléfono."""
        res = get_system_resources()
        ram = res["total_ram_mb"]
        if ram >= 5800:
            return "qwen2.5-3b"
        if ram >= 3500:
            return "qwen2.5-1.5b"
        if ram >= 2000:
            return "qwen2.5-0.5b"
        return "smollm2-360m"

    def import_local_gguf(self, source_path: Path) -> Path:
        """Vincula o copia un modelo .gguf existente en el teléfono (ej. ~/storage/downloads)."""
        src = Path(source_path).expanduser().resolve()
        if not src.exists() or src.suffix.lower() != ".gguf":
            raise FileNotFoundError(f"No se encontró un archivo .gguf válido en: {src}")
        dest = self.config.models_llm_dir / src.name
        if src != dest:
            try:
                if dest.exists():
                    dest.unlink()
                dest.symlink_to(src)
            except Exception:
                shutil.copy2(src, dest)
        self.config.active_gguf_model = dest.name
        self.config.save()
        return dest

    def download_gguf(
        self,
        model_key: str,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Descarga un modelo GGUF del catálogo con soporte de reanudación (.part)."""
        if model_key not in GGUF_CATALOG:
            raise ValueError(
                f"Modelo '{model_key}' desconocido. Opciones: {', '.join(GGUF_CATALOG.keys())}"
            )
        meta = GGUF_CATALOG[model_key]
        dest = self.config.models_llm_dir / meta["filename"]
        if dest.exists() and dest.stat().st_size > 10_000_000:
            self.config.active_gguf_model = dest.name
            self.config.save()
            return dest

        part = dest.with_suffix(".gguf.part")
        downloaded = part.stat().st_size if part.exists() else 0

        headers = {"User-Agent": "NEXUS-IA-Termux/1.0"}
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"

        req = urllib.request.Request(meta["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=45) as resp:
            content_len = resp.headers.get("Content-Length")
            total = (int(content_len) + downloaded) if content_len else meta["size_mb"] * 1024 * 1024
            mode = "ab" if downloaded > 0 and resp.status == 206 else "wb"
            if mode == "wb":
                downloaded = 0

            with open(part, mode) as fh:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total)

        part.rename(dest)
        self.config.active_gguf_model = dest.name
        self.config.save()
        return dest

    def build_llama_cpp(self) -> Path:
        """Instala o compila llama.cpp optimizado para el CPU local (Termux ARM64 o Linux)."""
        existing = self.find_llama_binary()
        if existing:
            return existing

        # En Termux intentamos primero el paquete precompilado oficial si existe
        if is_termux() and shutil.which("pkg"):
            subprocess.run(
                ["pkg", "install", "-y", "llama-cpp"],
                check=False,
            )
            existing = self.find_llama_binary()
            if existing:
                return existing

        for tool in ("git", "cmake"):
            if not shutil.which(tool):
                raise RuntimeError(
                    f"Falta '{tool}'. En Termux ejecuta: pkg install git clang cmake make"
                )

        src_dir = self.config.home_dir / "src" / "llama.cpp"
        src_dir.parent.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(src_dir)],
                check=True,
            )

        build_dir = src_dir / "build_local"
        res = get_system_resources()
        jobs = str(res["optimal_threads"])

        subprocess.run(
            [
                "cmake",
                "-S",
                str(src_dir),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=ON",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "llama-cli", "-j", jobs],
            check=True,
        )

        built = build_dir / "bin" / "llama-cli"
        if not built.exists():
            candidates = list(build_dir.rglob("llama-cli"))
            if not candidates:
                raise RuntimeError("La compilación terminó pero no se encontró el binario llama-cli.")
            built = candidates[0]

        target = self.config.bin_dir / "llama-cli"
        shutil.copy2(built, target)
        target.chmod(0o755)
        return target

    def status_report(self) -> Dict[str, Any]:
        """Genera resumen completo del estado de motores y modelos locales."""
        res = get_system_resources()
        llama_bin = self.find_llama_binary()
        sd_bin = self.find_sd_binary()
        ggufs = self.list_installed_gguf()
        active_gguf = self.get_active_gguf_path()
        rec_key = self.recommend_model_key()
        return {
            "resources": res,
            "llama_binary": str(llama_bin) if llama_bin else None,
            "sd_binary": str(sd_bin) if sd_bin else None,
            "installed_gguf": [p.name for p in ggufs],
            "active_gguf": active_gguf.name if active_gguf else None,
            "recommended_key": rec_key,
            "recommended_model": GGUF_CATALOG[rec_key],
        }

"""
Motor de Video 100% Local para Termux y Linux.
Capacidades:
1. Storyboard multi-escena ("Escena 1 | Escena 2 | Escena 3") con transiciones suaves cross-fade
2. Evolución física temporal real fotograma a fotograma (olas, nubes, cámara, luces, red neuronal CPPN)
3. Síntesis propia de banda sonora ambiental WAV (armónicos acordes a la escena)
4. Codificación adaptable:
   - Si `ffmpeg` está instalado (estándar en Termux): genera `.mp4` (H.264/MPEG4 + audio)
   - Sin `ffmpeg`: genera `.gif` animado (codificador GIF89a LZW propio en Python puro) + `.avi` RIFF nativo
5. Reproducción animada en vivo en la consola de Termux en TrueColor 24-bit
"""

from __future__ import annotations

import math
import random
import shutil
import struct
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import IAConfig
from .image_engine import (
    ImageEngine,
    RGBBuffer,
    _lerp_color,
    draw_text_on_buffer,
    parse_visual_scene,
    write_png_rgb,
)
from .memory import strip_accents


# ---------------------------------------------------------------------------
# Sintetizador de Audio Ambiental Propio (.wav PCM 16-bit)
# ---------------------------------------------------------------------------

CHORD_FREQS: Dict[str, Tuple[float, ...]] = {
    "cyberpunk": (110.0, 164.81, 220.0, 329.63),   # La menor synth
    "space": (130.81, 196.00, 246.94, 392.00),      # Do maj7 cósmico
    "night": (146.83, 220.00, 261.63, 349.23),      # Re menor nocturno
    "fire": (98.00, 146.83, 196.00, 293.66),        # Sol grave intenso
    "day": (174.61, 220.00, 261.63, 349.23),        # Fa mayor luminoso
    "snow": (196.00, 246.94, 293.66, 392.00),       # Sol mayor cristalino
    "sunset": (164.81, 207.65, 246.94, 329.63),     # Mi mayor cálido
}


def synthesize_ambient_wav(path: Path, duration_sec: float, mood: str = "sunset", sample_rate: int = 22050) -> Path:
    """Genera una pista musical/ambiental armónica en WAV de 16 bits sin librerías externas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = max(1, int(duration_sec * sample_rate))
    freqs = CHORD_FREQS.get(mood, CHORD_FREQS["sunset"])
    rng = random.Random(42)

    frames = bytearray()
    prev_noise = 0.0
    two_pi = math.tau

    for i in range(total_samples):
        t = i / sample_rate
        # Envolvente fade-in y fade-out suave
        env = 1.0
        if t < 0.45:
            env = t / 0.45
        elif duration_sec - t < 0.55:
            env = max(0.0, (duration_sec - t) / 0.55)

        lfo = 0.72 + 0.28 * math.sin(two_pi * 0.35 * t)
        val = 0.0
        for idx, f in enumerate(freqs):
            detune = 1.0 + 0.002 * idx * math.sin(two_pi * 0.2 * t)
            val += math.sin(two_pi * f * detune * t) * (0.28 / (1.0 + 0.25 * idx))

        # Arpegio suave si es cyberpunk o espacio
        if mood in ("cyberpunk", "space"):
            step_f = freqs[int(t * 3.0) % len(freqs)] * 2.0
            arp_env = max(0.0, 1.0 - ((t * 3.0) % 1.0) * 1.4)
            val += 0.14 * arp_env * math.sin(two_pi * step_f * t)

        # Brisa / olas filtradas suaves
        white = rng.uniform(-1.0, 1.0)
        prev_noise = 0.93 * prev_noise + 0.07 * white
        sample = (val * lfo + prev_noise * 0.12) * env
        pcm = max(-32767, min(32767, int(sample * 24000)))
        frames.extend(struct.pack("<h", pcm))

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))

    return path


# ---------------------------------------------------------------------------
# Codificador GIF89a Animado LZW en Python Puro (Cero dependencias)
# ---------------------------------------------------------------------------

def _lzw_compress_gif_indices(indices: bytes, min_code_size: int = 8) -> bytes:
    """Compresor LZW estándar para GIF89a con reinicio automático de tabla a 4096 códigos."""
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1

    out_bytes = bytearray()
    bit_buf = 0
    bit_count = 0

    def emit(code: int, width: int) -> None:
        nonlocal bit_buf, bit_count
        bit_buf |= code << bit_count
        bit_count += width
        while bit_count >= 8:
            out_bytes.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8

    code_size = min_code_size + 1
    next_code = eoi_code + 1
    table: Dict[Tuple[int, int], int] = {}

    emit(clear_code, code_size)
    if not indices:
        emit(eoi_code, code_size)
    else:
        prefix = indices[0]
        for b in indices[1:]:
            key = (prefix, b)
            if key in table:
                prefix = table[key]
            else:
                emit(prefix, code_size)
                if next_code < 4096:
                    table[key] = next_code
                    next_code += 1
                    if next_code > (1 << code_size) and code_size < 12:
                        code_size += 1
                else:
                    emit(clear_code, code_size)
                    table.clear()
                    code_size = min_code_size + 1
                    next_code = eoi_code + 1
                prefix = b
        emit(prefix, code_size)
        emit(eoi_code, code_size)

    if bit_count > 0:
        out_bytes.append(bit_buf & 0xFF)

    # Empaquetar en sub-bloques GIF de hasta 255 bytes
    blocks = bytearray([min_code_size])
    pos = 0
    total = len(out_bytes)
    while pos < total:
        chunk = out_bytes[pos : pos + 255]
        blocks.append(len(chunk))
        blocks.extend(chunk)
        pos += 255
    blocks.append(0)
    return bytes(blocks)


def write_animated_gif(path: Path, frames: List[RGBBuffer], fps: int = 12) -> Path:
    """Escribe un archivo GIF89a animado en bucle infinito usando cuantización 3-3-2 con dithering ligero."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No hay fotogramas para codificar el GIF.")

    # Si Pillow está instalado, aprovechar su cuantizador adaptativo; si no, nuestro GIF89a LZW puro
    try:
        from PIL import Image  # type: ignore

        pil_frames = []
        for f in frames:
            h, w = len(f), len(f[0])
            im = Image.new("RGB", (w, h))
            im.putdata([px for row in f for px in row])
            pil_frames.append(im)
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(20, int(1000 / max(1, fps))),
            loop=0,
            optimize=True,
        )
        return path
    except Exception:
        pass

    h = len(frames[0])
    w = len(frames[0][0])
    delay_cs = max(2, int(round(100.0 / max(1, fps))))

    # Construir tabla global de colores 3-3-2 (256 colores)
    gct = bytearray()
    for i in range(256):
        r3 = (i >> 5) & 0x07
        g3 = (i >> 2) & 0x07
        b2 = i & 0x03
        gct.append(int(r3 * 255 / 7))
        gct.append(int(g3 * 255 / 7))
        gct.append(int(b2 * 255 / 3))

    gif = bytearray(b"GIF89a")
    gif.extend(struct.pack("<HHBBB", w, h, 0xF7, 0, 0))
    gif.extend(gct)
    # Extensión NETSCAPE2.0 para bucle continuo
    gif.extend(b"!\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00")

    bayer2 = ((0, 2), (3, 1))
    for frame in frames:
        idx_buf = bytearray(w * h)
        pos = 0
        for y in range(h):
            row = frame[y]
            d_row = bayer2[y & 1]
            for x in range(w):
                r, g, b = row[x]
                d = (d_row[x & 1] - 1) * 6
                rc = max(0, min(255, r + d)) >> 5
                gc = max(0, min(255, g + d)) >> 5
                bc = max(0, min(255, b + d)) >> 6
                idx_buf[pos] = (rc << 5) | (gc << 2) | bc
                pos += 1

        gif.extend(b"!\xF9\x04\x00" + struct.pack("<H", delay_cs) + b"\x00\x00")
        gif.extend(b"," + struct.pack("<HHHHB", 0, 0, w, h, 0))
        gif.extend(_lzw_compress_gif_indices(bytes(idx_buf), 8))

    gif.append(0x3B)
    path.write_bytes(bytes(gif))
    return path


# ---------------------------------------------------------------------------
# Codificador RIFF AVI RGB de 24 bits en Python Puro
# ---------------------------------------------------------------------------

def write_uncompressed_avi(path: Path, frames: List[RGBBuffer], fps: int = 12) -> Path:
    """Escribe un contenedor de video estándar RIFF AVI (DIB RGB24) reproducible en VLC/mpv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = len(frames[0])
    w = len(frames[0][0])
    n_frames = len(frames)

    row_raw = w * 3
    pad = (4 - (row_raw % 4)) % 4
    stride = row_raw + pad
    frame_size = stride * h
    usec_per_frame = int(1_000_000 / max(1, fps))

    # Construir chunks '00db' + índice 'idx1'
    movi_payload = bytearray(b"movi")
    idx1_payload = bytearray()
    offset = 4  # relativo al inicio de 'movi'

    for frame in frames:
        buf = bytearray()
        # En DIB/BMP dentro de AVI las filas van de abajo hacia arriba en orden B, G, R
        for y in range(h - 1, -1, -1):
            for r, g, b in frame[y]:
                buf.append(b & 255)
                buf.append(g & 255)
                buf.append(r & 255)
            if pad:
                buf.extend(b"\x00" * pad)

        chunk_hdr = b"00db" + struct.pack("<I", frame_size)
        movi_payload.extend(chunk_hdr)
        movi_payload.extend(buf)
        idx1_payload.extend(b"00db" + struct.pack("<III", 0x10, offset, frame_size))
        offset += 8 + frame_size

    avih = struct.pack(
        "<14I",
        usec_per_frame,
        frame_size * fps,
        0,
        0x10,  # AVIF_HASINDEX
        n_frames,
        0,
        1,
        frame_size,
        w,
        h,
        0,
        0,
        0,
        0,
    )
    strh = struct.pack(
        "<4s4sIHHIIIIIIII4H",
        b"vids",
        b"DIB ",
        0,
        0,
        0,
        0,
        1,
        fps,
        0,
        n_frames,
        frame_size,
        0xFFFFFFFF,
        0,
        0,
        0,
        w,
        h,
    )
    strf = struct.pack(
        "<IiiHHIIiiII",
        40,
        w,
        h,
        1,
        24,
        0,  # BI_RGB
        frame_size,
        0,
        0,
        0,
        0,
    )

    def chunk(fourcc: bytes, data: bytes) -> bytes:
        even_pad = b"\x00" if (len(data) & 1) else b""
        return fourcc + struct.pack("<I", len(data)) + data + even_pad

    def riff_list(list_type: bytes, inner: bytes) -> bytes:
        return chunk(b"LIST", list_type + inner)

    strl = riff_list(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf))
    hdrl = riff_list(b"hdrl", chunk(b"avih", avih) + strl)
    movi = chunk(b"LIST", bytes(movi_payload))
    idx1 = chunk(b"idx1", bytes(idx1_payload))

    avi_body = b"AVI " + hdrl + movi + idx1
    avi_bytes = b"RIFF" + struct.pack("<I", len(avi_body)) + avi_body
    path.write_bytes(avi_bytes)
    return path


# ---------------------------------------------------------------------------
# Motor Principal de Video
# ---------------------------------------------------------------------------

class VideoEngine:
    """Sintetiza animaciones multi-escena con física temporal y audio local."""

    def __init__(self, config: IAConfig, image_engine: ImageEngine):
        self.config = config
        self.image_engine = image_engine

    def generate(
        self,
        prompt: str,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        with_subtitles: bool = True,
        out_path: Optional[Path] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Genera un video completo a partir de un prompt o storyboard separado por '|'.
        Ejemplo: "Amanecer en montañas | Ciudad cyberpunk de noche | Planeta con anillos"
        """
        dur = max(1.0, min(20.0, float(duration or self.config.video_duration)))
        rate = max(4, min(30, int(fps or self.config.video_fps)))
        w = width or self.config.video_width
        h = height or self.config.video_height
        # Garantizar dimensiones pares para compatibilidad H.264 yuv420p
        w = w - (w % 2)
        h = h - (h % 2)

        raw_scenes = [s.strip() for s in prompt.split("|") if s.strip()] or [prompt.strip() or "Paisaje"]
        scenes = [parse_visual_scene(s, seed=None) for s in raw_scenes]

        total_frames = max(4, int(round(dur * rate)))
        frames_per_scene = total_frames / len(scenes)
        fade_frames = max(1, min(5, int(rate * 0.4)))

        t0 = time.perf_counter()
        frames: List[RGBBuffer] = []

        for idx in range(total_frames):
            t_sec = idx / rate
            sc_idx = min(len(scenes) - 1, int(idx / frames_per_scene))
            cur_scene = scenes[sc_idx]
            frame = self.image_engine.render_scene_frame(cur_scene, w, h, t=t_sec)

            # Transición suave cross-fade hacia la siguiente escena
            if sc_idx < len(scenes) - 1:
                next_start = int((sc_idx + 1) * frames_per_scene)
                dist_to_cut = next_start - idx
                if 0 < dist_to_cut <= fade_frames:
                    alpha = 1.0 - (dist_to_cut / (fade_frames + 1))
                    next_frame = self.image_engine.render_scene_frame(scenes[sc_idx + 1], w, h, t=t_sec)
                    blended: RGBBuffer = [
                        [_lerp_color(frame[y][x], next_frame[y][x], alpha) for x in range(w)]
                        for y in range(h)
                    ]
                    frame = blended

            if with_subtitles and len(raw_scenes[sc_idx]) >= 3:
                scale = max(1, w // 220)
                draw_text_on_buffer(
                    frame,
                    raw_scenes[sc_idx][:30],
                    y_pos=h - 10 * scale - 6,
                    scale=scale,
                )

            frames.append(frame)
            if progress_cb:
                progress_cb(idx + 1, total_frames)

        ts = time.strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() else "_" for c in strip_accents(raw_scenes[0])[:24]).strip("_") or "video"
        has_ffmpeg = shutil.which("ffmpeg") is not None

        mood = scenes[0]["palette_key"]
        audio_path: Optional[Path] = None

        if has_ffmpeg:
            final_path = Path(out_path) if out_path else (self.config.videos_dir / f"vid_{ts}_{slug}.mp4")
            if final_path.suffix.lower() not in (".mp4", ".mkv", ".webm"):
                final_path = final_path.with_suffix(".mp4")
            final_path.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                for i, fr in enumerate(frames):
                    write_png_rgb(tmp / f"frame_{i:04d}.png", fr)

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(rate),
                    "-i",
                    str(tmp / "frame_%04d.png"),
                ]
                if self.config.video_with_audio:
                    wav_tmp = tmp / "soundtrack.wav"
                    synthesize_ambient_wav(wav_tmp, dur, mood=mood)
                    cmd.extend(["-i", str(wav_tmp), "-shortest"])

                cmd.extend(["-pix_fmt", "yuv420p", str(final_path)])
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if proc.returncode == 0 and final_path.exists():
                    encoder_label = "FFmpeg MP4 + Audio Sintetizado Local"
                    # También guardamos miniatura GIF ligera para compartir fácil si se desea
                    gif_companion = final_path.with_suffix(".gif")
                    write_animated_gif(gif_companion, frames, fps=rate)
                else:
                    has_ffmpeg = False

        if not has_ffmpeg:
            # Codificación 100% nativa en Python puro: GIF animado + contenedor RIFF AVI + pista WAV
            base_target = Path(out_path) if out_path else (self.config.videos_dir / f"vid_{ts}_{slug}.gif")
            final_path = base_target
            final_path.parent.mkdir(parents=True, exist_ok=True)

            if final_path.suffix.lower() == ".avi":
                write_uncompressed_avi(final_path, frames, fps=rate)
                gif_companion = final_path.with_suffix(".gif")
                write_animated_gif(gif_companion, frames, fps=rate)
            else:
                final_path = final_path.with_suffix(".gif")
                write_animated_gif(final_path, frames, fps=rate)
                avi_companion = final_path.with_suffix(".avi")
                write_uncompressed_avi(avi_companion, frames, fps=rate)
                gif_companion = final_path

            if self.config.video_with_audio:
                audio_path = final_path.with_suffix(".wav")
                synthesize_ambient_wav(audio_path, dur, mood=mood)

            encoder_label = "Codificador Propio Nativo (GIF89a + AVI RGB24 + WAV)"

        elapsed = round(time.perf_counter() - t0, 3)
        mirrored = self.config.mirror_media_if_android(final_path, media_type="video")

        return {
            "path": str(final_path),
            "gif_path": str(gif_companion),
            "audio_path": str(audio_path) if audio_path else None,
            "android_gallery_path": str(mirrored) if mirrored else None,
            "frames_count": total_frames,
            "fps": rate,
            "duration": dur,
            "width": w,
            "height": h,
            "scenes_count": len(scenes),
            "encoder": encoder_label,
            "elapsed_sec": elapsed,
            "frames": frames,
        }

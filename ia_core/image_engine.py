"""
Motor de Imágenes 100% Local para Termux y Linux (Cero dependencias obligatorias).
Incluye:
1. Codificador/Decodificador PNG 24-bit nativo en Python puro (struct + zlib)
2. Integración directa con `stable-diffusion.cpp` local en CPU ARM64 si hay pesos instalados
3. Sintetizador Visual Propio Semántico + Matemático + Red Neuronal CPPN:
   - Paisajes atmosféricos multicapa (montañas FBM, océano con reflejos reales, bosques, ciudades cyberpunk)
   - Escenas espaciales 3D (planetas iluminados, anillos, nebulosas, agujeros negros)
   - Fractales de Julia/Mandelbrot suaves
   - Red Neuronal Generativa CPPN (Compositional Pattern Producing Network) condicionada por el prompt
   - Parámetro temporal continuo `t` para animación fluida desde `video_engine.py`
4. Fuente bitmap propia escalable para títulos/pósters y filtros de edición local
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
import subprocess
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import IAConfig
from .memory import strip_accents
from .model_manager import ModelManager

RGBBuffer = List[List[Tuple[int, int, int]]]


# ---------------------------------------------------------------------------
# Codificador / Decodificador PNG 24-bit en Python puro
# ---------------------------------------------------------------------------

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def write_png_rgb(path: Path, pixels: RGBBuffer) -> Path:
    """Guarda un buffer RGB [H][W][(r,g,b)] como archivo PNG estándar de 24 bits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height = len(pixels)
    width = len(pixels[0]) if height else 0

    raw_rows = bytearray()
    for row in pixels:
        raw_rows.append(0)  # Filtro PNG tipo 0 (None)
        for r, g, b in row:
            raw_rows.append(r & 255)
            raw_rows.append(g & 255)
            raw_rows.append(b & 255)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    compressed = zlib.compress(bytes(raw_rows), level=6)

    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png_bytes)
    return path


def read_png_rgb(path: Path) -> RGBBuffer:
    """Lee un PNG RGB de 8 bits (o usa Pillow si está instalado para otros formatos)."""
    path = Path(path)
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            flat = list(rgb.getdata())
            return [flat[y * w : (y + 1) * w] for y in range(h)]
    except Exception:
        pass

    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Formato no soportado sin Pillow (se esperaba PNG estándar).")

    pos = 8
    width = height = 0
    idat_parts: List[bytes] = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        cdata = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", cdata)
            if bit_depth != 8 or color_type not in (2, 6):
                raise ValueError("Solo PNG RGB/RGBA de 8 bits soportado en lector puro.")
            channels = 3 if color_type == 2 else 4
        elif ctype == b"IDAT":
            idat_parts.append(cdata)
        elif ctype == b"IEND":
            break

    raw = zlib.decompress(b"".join(idat_parts))
    stride = width * channels
    pixels: RGBBuffer = []
    idx = 0
    prev_row = bytearray(stride)

    for _ in range(height):
        ftype = raw[idx]
        idx += 1
        cur = bytearray(raw[idx : idx + stride])
        idx += stride

        if ftype == 1:  # Sub
            for i in range(channels, stride):
                cur[i] = (cur[i] + cur[i - channels]) & 255
        elif ftype == 2:  # Up
            for i in range(stride):
                cur[i] = (cur[i] + prev_row[i]) & 255
        elif ftype == 3:  # Average
            for i in range(stride):
                left = cur[i - channels] if i >= channels else 0
                cur[i] = (cur[i] + ((left + prev_row[i]) >> 1)) & 255
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = cur[i - channels] if i >= channels else 0
                b = prev_row[i]
                c = prev_row[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[i] = (cur[i] + pr) & 255

        prev_row = cur
        row_rgb: List[Tuple[int, int, int]] = []
        for x in range(0, stride, channels):
            row_rgb.append((cur[x], cur[x + 1], cur[x + 2]))
        pixels.append(row_rgb)

    return pixels


# ---------------------------------------------------------------------------
# Fuente Bitmap Propia 5x7 para títulos, pósters y subtítulos de video
# ---------------------------------------------------------------------------

FONT_5X7: Dict[str, Tuple[int, ...]] = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    "A": (0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "B": (0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110),
    "C": (0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110),
    "D": (0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100),
    "E": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111),
    "F": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000),
    "G": (0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110),
    "H": (0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "I": (0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "J": (0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100),
    "K": (0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001),
    "L": (0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111),
    "M": (0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001),
    "N": (0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001),
    "O": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "P": (0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000),
    "Q": (0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
    "S": (0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110),
    "T": (0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100),
    "U": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "V": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100),
    "W": (0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001),
    "X": (0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001),
    "Y": (0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100),
    "Z": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111),
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    "3": (0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110),
    ".": (0, 0, 0, 0, 0, 0b01100, 0b01100),
    ",": (0, 0, 0, 0, 0b01100, 0b00100, 0b01000),
    ":": (0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0),
    "-": (0, 0, 0, 0b11111, 0, 0, 0),
    "!": (0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0, 0b00100),
    "?": (0b01110, 0b10001, 0b00010, 0b00100, 0b00100, 0, 0b00100),
}


def draw_text_on_buffer(
    pixels: RGBBuffer,
    text: str,
    y_pos: int,
    scale: int = 2,
    fg: Tuple[int, int, int] = (248, 250, 252),
    shadow: Tuple[int, int, int] = (10, 14, 26),
) -> None:
    """Dibuja texto centrado con sombra suave sobre un buffer RGB."""
    clean = strip_accents(text).upper()
    clean = "".join(ch if ch in FONT_5X7 else " " for ch in clean).strip()
    if not clean or not pixels:
        return

    h = len(pixels)
    w = len(pixels[0])
    max_chars = max(1, (w - 16) // (6 * scale))
    if len(clean) > max_chars:
        clean = clean[: max_chars - 2] + ".."

    total_w = len(clean) * 6 * scale - scale
    start_x = max(4, (w - total_w) // 2)

    for pass_offset, color in ((scale, shadow), (0, fg)):
        cx = start_x + pass_offset
        cy = y_pos + pass_offset
        for ch in clean:
            glyph = FONT_5X7.get(ch, FONT_5X7[" "])
            for gy in range(7):
                bits = glyph[gy]
                if bits == 0:
                    continue
                for gx in range(5):
                    if (bits >> (4 - gx)) & 1:
                        for sy in range(scale):
                            py = cy + gy * scale + sy
                            if 0 <= py < h:
                                row = pixels[py]
                                for sx in range(scale):
                                    px = cx + gx * scale + sx
                                    if 0 <= px < w:
                                        row[px] = color
            cx += 6 * scale


# ---------------------------------------------------------------------------
# Ruido Perlin / Value 2D rápido + Interpolación cromática
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


class _FastNoise:
    """Ruido continuo determinista ultrarrápido con tabla hash precalculada."""

    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.perm = [rng.random() for _ in range(1024)]

    def noise2d(self, x: float, y: float) -> float:
        xi = int(math.floor(x))
        yi = int(math.floor(y))
        xf = x - xi
        yf = y - yi
        u = xf * xf * (3.0 - 2.0 * xf)
        v = yf * yf * (3.0 - 2.0 * yf)

        p = self.perm
        h00 = p[(xi * 57 + yi * 131) & 1023]
        h10 = p[((xi + 1) * 57 + yi * 131) & 1023]
        h01 = p[(xi * 57 + (yi + 1) * 131) & 1023]
        h11 = p[((xi + 1) * 57 + (yi + 1) * 131) & 1023]

        nx0 = h00 + (h10 - h00) * u
        nx1 = h01 + (h11 - h01) * u
        return nx0 + (nx1 - nx0) * v

    def fbm1d(self, x: float, octaves: int = 4) -> float:
        val = 0.0
        amp = 0.5
        freq = 1.0
        for i in range(octaves):
            val += amp * self.noise2d(x * freq, i * 17.3)
            freq *= 2.03
            amp *= 0.5
        return val


# ---------------------------------------------------------------------------
# Sintetizador Visual Semántico + Red Neuronal CPPN Propia
# ---------------------------------------------------------------------------

PALETTES: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "sunset": {
        "sky_top": (24, 20, 68),
        "sky_mid": (194, 59, 90),
        "sky_bot": (253, 178, 76),
        "sun": (255, 232, 148),
        "mtn_far": (78, 42, 84),
        "mtn_near": (32, 20, 48),
        "water": (45, 32, 72),
        "accent": (255, 140, 82),
    },
    "cyberpunk": {
        "sky_top": (8, 8, 28),
        "sky_mid": (38, 12, 74),
        "sky_bot": (168, 24, 130),
        "sun": (0, 245, 255),
        "mtn_far": (26, 18, 58),
        "mtn_near": (12, 12, 28),
        "water": (14, 18, 42),
        "accent": (0, 255, 213),
    },
    "night": {
        "sky_top": (5, 10, 26),
        "sky_mid": (15, 28, 64),
        "sky_bot": (34, 62, 110),
        "sun": (238, 244, 255),
        "mtn_far": (22, 36, 66),
        "mtn_near": (10, 18, 34),
        "water": (12, 24, 48),
        "accent": (125, 211, 252),
    },
    "day": {
        "sky_top": (28, 108, 208),
        "sky_mid": (86, 172, 242),
        "sky_bot": (198, 234, 255),
        "sun": (255, 250, 205),
        "mtn_far": (92, 128, 168),
        "mtn_near": (38, 76, 62),
        "water": (28, 118, 176),
        "accent": (74, 222, 128),
    },
    "fire": {
        "sky_top": (22, 6, 8),
        "sky_mid": (136, 24, 16),
        "sky_bot": (252, 138, 28),
        "sun": (255, 238, 120),
        "mtn_far": (64, 18, 16),
        "mtn_near": (24, 8, 8),
        "water": (48, 14, 12),
        "accent": (255, 94, 20),
    },
    "snow": {
        "sky_top": (48, 76, 118),
        "sky_mid": (128, 164, 204),
        "sky_bot": (226, 238, 250),
        "sun": (255, 255, 255),
        "mtn_far": (162, 186, 214),
        "mtn_near": (212, 228, 244),
        "water": (78, 118, 158),
        "accent": (186, 230, 253),
    },
}


def parse_visual_scene(prompt: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Analiza semánticamente el prompt en español/inglés para configurar la escena."""
    low = strip_accents(prompt.lower())
    digest = hashlib.sha256(low.encode("utf-8")).digest()
    base_seed = seed if seed is not None else int.from_bytes(digest[:4], "big")

    # Modo principal
    if any(k in low for k in ("fractal", "mandelbrot", "julia", "caleidoscopio")):
        mode = "fractal"
    elif any(k in low for k in ("cppn", "red neuronal", "abstracto", "psicodelico", "energia", "ondas")):
        mode = "cppn"
    elif any(k in low for k in ("espacio", "galaxia", "planeta", "cosmos", "nebulosa", "agujero negro", "universo")):
        mode = "space"
    else:
        mode = "landscape"

    # Paleta cromática
    if any(k in low for k in ("cyberpunk", "neon", "futurista", "synthwave", "laser")):
        palette_key = "cyberpunk"
    elif any(k in low for k in ("fuego", "volcan", "lava", "rojo", "marte", "infierno")):
        palette_key = "fire"
    elif any(k in low for k in ("nieve", "invierno", "hielo", "artico", "glaciar")):
        palette_key = "snow"
    elif any(k in low for k in ("noche", "nocturn", "luna", "oscuro", "medianoche", "estrellas")):
        palette_key = "night"
    elif any(k in low for k in ("dia", "soleado", "primavera", "pradera", "azul")):
        palette_key = "day"
    else:
        palette_key = "sunset"

    has_city = any(k in low for k in ("ciudad", "cyberpunk", "edificios", "rascacielos", "urbano", "metropolis", "calle"))
    has_forest = any(k in low for k in ("bosque", "arbol", "selva", "pino", "naturaleza", "jungla"))
    has_water = not any(k in low for k in ("desierto", "arido")) and (
        any(k in low for k in ("mar", "oceano", "agua", "lago", "rio", "playa", "reflejo", "isla"))
        or palette_key in ("sunset", "cyberpunk")
    )
    has_snow_particles = palette_key == "snow" or "lluvia" in low or "tormenta" in low
    is_rain = "lluvia" in low or "tormenta" in low
    has_rings = any(k in low for k in ("saturno", "anillo", "agujero negro", "planeta"))

    # Texto en póster si pide cartel/portada/texto entre comillas
    quoted = None
    import re
    m_q = re.search(r'["«“]([^"»”]{2,32})["»”]', prompt)
    if m_q:
        quoted = m_q.group(1)

    return {
        "prompt": prompt,
        "seed": base_seed,
        "mode": mode,
        "palette_key": palette_key,
        "palette": PALETTES[palette_key],
        "has_city": has_city,
        "has_forest": has_forest,
        "has_water": has_water,
        "has_particles": has_snow_particles or palette_key == "fire",
        "is_rain": is_rain,
        "has_stars": palette_key in ("night", "cyberpunk", "sunset") or mode == "space",
        "has_rings": has_rings,
        "caption": quoted,
    }


class ImageEngine:
    """Generador y editor de imágenes 100% local."""

    def __init__(self, config: IAConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager

    def render_scene_frame(
        self,
        scene: Dict[str, Any],
        width: int,
        height: int,
        t: float = 0.0,
    ) -> RGBBuffer:
        """
        Renderiza un fotograma RGB completo en `t` segundos (t=0 para imagen estática,
        t>0 para animación fluida de cámara, agua, nubes, luces y red neuronal).
        """
        mode = scene["mode"]
        if mode == "cppn":
            buf = self._render_cppn(scene, width, height, t)
        elif mode == "fractal":
            buf = self._render_fractal(scene, width, height, t)
        elif mode == "space":
            buf = self._render_space(scene, width, height, t)
        else:
            buf = self._render_landscape(scene, width, height, t)

        if scene.get("caption"):
            scale = max(1, min(3, width // 160))
            draw_text_on_buffer(buf, scene["caption"], y_pos=height - 12 * scale - 8, scale=scale)
        return buf

    def _render_cppn(self, scene: Dict[str, Any], width: int, height: int, t: float) -> RGBBuffer:
        """
        Red Neuronal Generativa Propia CPPN (Compositional Pattern Producing Network).
        Mapea coordenadas espaciales (x, y, radio, ángulo) + latente temporal z(t) -> RGB.
        """
        rng = random.Random(scene["seed"])
        hidden = 8
        # Pesos para [x, y, r, sin(theta*k), cos(t), sin(t)] -> capa 1 -> capa 2 -> RGB
        w1 = [[rng.uniform(-3.6, 3.6) for _ in range(6)] for _ in range(hidden)]
        b1 = [rng.uniform(-1.5, 1.5) for _ in range(hidden)]
        w2 = [[rng.uniform(-2.4, 2.4) for _ in range(hidden)] for _ in range(hidden)]
        w_out = [[rng.uniform(-2.6, 2.6) for _ in range(hidden)] for _ in range(3)]
        freq_k = rng.choice([2.0, 3.0, 4.0, 5.0, 6.0])

        zt_c = math.cos(t * 0.9) * 0.65
        zt_s = math.sin(t * 0.9) * 0.65
        aspect = width / max(1, height)

        pixels: RGBBuffer = []
        sin = math.sin
        cos = math.cos
        tanh = math.tanh
        sqrt = math.sqrt
        atan2 = math.atan2

        for py in range(height):
            ny = (py / max(1, height - 1) - 0.5) * 2.0
            row: List[Tuple[int, int, int]] = []
            for px in range(width):
                nx = (px / max(1, width - 1) - 0.5) * 2.0 * aspect
                r = sqrt(nx * nx + ny * ny)
                ang = sin(atan2(ny, nx) * freq_k + t * 0.7)
                inp = (nx, ny, r, ang, zt_c, zt_s)

                h1 = [
                    sin(
                        inp[0] * w1[j][0]
                        + inp[1] * w1[j][1]
                        + inp[2] * w1[j][2]
                        + inp[3] * w1[j][3]
                        + inp[4] * w1[j][4]
                        + inp[5] * w1[j][5]
                        + b1[j]
                    )
                    for j in range(hidden)
                ]
                h2 = [
                    tanh(sum(h1[i] * w2[j][i] for i in range(hidden)))
                    for j in range(hidden)
                ]
                rc = int((0.5 + 0.5 * tanh(sum(h2[i] * w_out[0][i] for i in range(hidden)))) * 255)
                gc = int((0.5 + 0.5 * tanh(sum(h2[i] * w_out[1][i] for i in range(hidden)))) * 255)
                bc = int((0.5 + 0.5 * tanh(sum(h2[i] * w_out[2][i] for i in range(hidden)))) * 255)
                row.append((rc, gc, bc))
            pixels.append(row)
        return pixels

    def _render_fractal(self, scene: Dict[str, Any], width: int, height: int, t: float) -> RGBBuffer:
        """Renderizador de conjunto de Julia animado de alta suavidad cromática."""
        rng = random.Random(scene["seed"])
        base_angle = rng.uniform(0.0, math.tau) + t * 0.28
        cx = 0.7885 * math.cos(base_angle)
        cy = 0.7885 * math.sin(base_angle)
        zoom = 1.35 + 0.25 * math.sin(t * 0.5)
        aspect = width / max(1, height)
        max_iter = 38
        pal = scene["palette"]

        pixels: RGBBuffer = []
        for py in range(height):
            zy0 = ((py / max(1, height - 1)) - 0.5) * 2.4 / zoom
            row: List[Tuple[int, int, int]] = []
            for px in range(width):
                zx = ((px / max(1, width - 1)) - 0.5) * 2.4 * aspect / zoom
                zy = zy0
                it = 0
                while zx * zx + zy * zy < 4.0 and it < max_iter:
                    zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy
                    it += 1
                if it == max_iter:
                    row.append(pal["sky_top"])
                else:
                    ratio = it / max_iter
                    if ratio < 0.5:
                        col = _lerp_color(pal["sky_top"], pal["sky_mid"], ratio * 2.0)
                    else:
                        col = _lerp_color(pal["sky_mid"], pal["sun"], (ratio - 0.5) * 2.0)
                    row.append(col)
            pixels.append(row)
        return pixels

    def _render_space(self, scene: Dict[str, Any], width: int, height: int, t: float) -> RGBBuffer:
        """Escena cósmica 3D: nebulosa FBM + estrellas + planeta iluminado con anillos orbitales."""
        noise = _FastNoise(scene["seed"])
        rng = random.Random(scene["seed"])
        pal = scene["palette"]

        # Pre-generar estrellas
        stars: Dict[Tuple[int, int], int] = {}
        for _ in range((width * height) // 110):
            sx = rng.randrange(0, width)
            sy = rng.randrange(0, height)
            phase = rng.uniform(0, math.tau)
            bri = int(155 + 100 * math.sin(t * 3.2 + phase))
            stars[(sx, sy)] = max(80, min(255, bri))

        # Geometría del planeta central
        pcx = width * (0.52 + 0.04 * math.sin(t * 0.4))
        pcy = height * 0.48
        prad = min(width, height) * 0.24
        prad2 = prad * prad

        # Dirección de la luz solar en 3D
        lx = math.cos(0.6 + t * 0.45) * 0.85
        ly = -0.35
        lz = math.sin(0.6 + t * 0.45) * 0.85
        lnorm = math.sqrt(lx * lx + ly * ly + lz * lz)
        lx, ly, lz = lx / lnorm, ly / lnorm, lz / lnorm

        pixels: RGBBuffer = []
        for py in range(height):
            ny = py / max(1, height - 1)
            row: List[Tuple[int, int, int]] = []
            dy = py - pcy
            for px in range(width):
                nx = px / max(1, width - 1)
                # Fondo de nebulosa
                neb = noise.noise2d(nx * 3.5 + t * 0.08, ny * 3.5 - t * 0.05)
                neb2 = noise.noise2d(nx * 7.0 - t * 0.12, ny * 7.0)
                neb_val = max(0.0, min(1.0, neb * 0.65 + neb2 * 0.35))
                bg = _lerp_color((4, 6, 18), pal["sky_mid"], neb_val * 0.85)
                if neb_val > 0.68:
                    bg = _lerp_color(bg, pal["accent"], (neb_val - 0.68) * 2.2)

                star_b = stars.get((px, py))
                if star_b is not None:
                    bg = (min(255, bg[0] + star_b), min(255, bg[1] + star_b), min(255, bg[2] + star_b))

                dx = px - pcx
                d2 = dx * dx + dy * dy

                # Anillo orbital inclinado (detrás / delante)
                ring_rx = dx * 0.96 + dy * 0.28
                ring_ry = (-dx * 0.28 + dy * 0.96) * 3.2
                ring_d = math.sqrt(ring_rx * ring_rx + ring_ry * ring_ry)
                in_ring = scene["has_rings"] and (prad * 1.32 <= ring_d <= prad * 2.15)
                ring_front = dy > 0

                if in_ring and not ring_front and d2 >= prad2:
                    band = 0.5 + 0.5 * math.sin((ring_d - prad) * 0.45 - t * 2.0)
                    bg = _lerp_color(bg, pal["sun"], 0.35 + 0.4 * band)

                # Esfera planetaria 3D con textura superficial y atmósfera Fresnel
                if d2 < prad2:
                    dz = math.sqrt(prad2 - d2)
                    snx, sny, snz = dx / prad, dy / prad, dz / prad
                    tex = noise.noise2d(snx * 4.2 + t * 0.25, sny * 4.2)
                    surf_col = _lerp_color(pal["mtn_far"], pal["sky_bot"], tex)
                    diff = max(0.06, snx * lx + sny * ly + snz * lz)
                    rim = (1.0 - snz) ** 2.6
                    lit = (
                        min(255, int(surf_col[0] * diff + pal["accent"][0] * rim * 0.55)),
                        min(255, int(surf_col[1] * diff + pal["accent"][1] * rim * 0.55)),
                        min(255, int(surf_col[2] * diff + pal["accent"][2] * rim * 0.55)),
                    )
                    bg = lit
                elif d2 < (prad * 1.12) ** 2:
                    # Halo atmosférico exterior
                    glow = 1.0 - (math.sqrt(d2) - prad) / (prad * 0.12)
                    bg = _lerp_color(bg, pal["accent"], glow * 0.45)

                if in_ring and ring_front:
                    band = 0.5 + 0.5 * math.sin((ring_d - prad) * 0.45 - t * 2.0)
                    bg = _lerp_color(bg, pal["sun"], 0.48 + 0.38 * band)

                row.append(bg)
            pixels.append(row)
        return pixels

    def _render_landscape(self, scene: Dict[str, Any], width: int, height: int, t: float) -> RGBBuffer:
        """
        Sintetizador de paisaje/ciudad multicapa con perspectiva atmosférica,
        nubes FBM en movimiento, sol/luna con halo, montañas o rascacielos y reflejos en agua.
        """
        noise = _FastNoise(scene["seed"])
        rng = random.Random(scene["seed"])
        pal = scene["palette"]

        horizon_y = int(height * (0.64 if scene["has_water"] else 0.78))

        # Posición solar/lunar
        sun_x = width * (0.62 + 0.08 * math.sin(t * 0.35))
        sun_y = horizon_y * (0.44 - 0.06 * math.cos(t * 0.35))
        sun_r = min(width, height) * 0.11

        # Alturas precalculadas por columna para velocidad máxima en móvil
        far_mtn = [
            int(horizon_y - height * (0.10 + 0.22 * noise.fbm1d(x / max(1.0, width) * 3.8 + t * 0.04, 4)))
            for x in range(width)
        ]
        near_mtn = [
            int(horizon_y - height * (0.04 + 0.15 * noise.fbm1d(x / max(1.0, width) * 6.2 + 9.4 + t * 0.09, 4)))
            for x in range(width)
        ]

        # Silueta urbana si pide ciudad / cyberpunk
        buildings: List[Tuple[int, int]] = [(horizon_y, 0)] * width
        if scene["has_city"]:
            bx = 0
            b_idx = 0
            while bx < width:
                b_rng = random.Random(scene["seed"] + b_idx * 97)
                bw = max(6, int(width * b_rng.uniform(0.035, 0.085)))
                bh = int(height * b_rng.uniform(0.10, 0.36))
                top_y = max(4, horizon_y - bh)
                for cx in range(bx, min(width, bx + bw - 1)):
                    buildings[cx] = (top_y, b_idx + 1)
                bx += bw
                b_idx += 1

        # Árboles si pide bosque
        trees = [0] * width
        if scene["has_forest"]:
            for tx in range(0, width, max(4, width // 38)):
                t_rng = random.Random(scene["seed"] + tx * 31)
                th = int(height * t_rng.uniform(0.06, 0.15))
                tw = max(3, th // 2)
                for dx in range(-tw, tw + 1):
                    cx = tx + dx
                    if 0 <= cx < width:
                        cone = int(th * (1.0 - abs(dx) / (tw + 1)))
                        trees[cx] = max(trees[cx], cone)

        # Estrellas fijas con parpadeo temporal
        stars: Dict[Tuple[int, int], int] = {}
        if scene["has_stars"]:
            for _ in range((width * horizon_y) // 160):
                sx = rng.randrange(0, width)
                sy = rng.randrange(0, max(1, int(horizon_y * 0.72)))
                ph = rng.uniform(0, math.tau)
                stars[(sx, sy)] = max(90, min(255, int(170 + 85 * math.sin(t * 4.0 + ph))))

        # Primero renderizamos el cielo y el terreno superior (0 .. horizon_y)
        pixels: RGBBuffer = [[(0, 0, 0)] * width for _ in range(height)]

        for py in range(horizon_y):
            sky_t = py / max(1, horizon_y - 1)
            if sky_t < 0.55:
                base_sky = _lerp_color(pal["sky_top"], pal["sky_mid"], sky_t / 0.55)
            else:
                base_sky = _lerp_color(pal["sky_mid"], pal["sky_bot"], (sky_t - 0.55) / 0.45)

            row = pixels[py]
            dy_sun = py - sun_y
            for px in range(width):
                col = base_sky

                # Estrellas en cielo alto
                sb = stars.get((px, py))
                if sb is not None and sky_t < 0.62:
                    col = _lerp_color(col, (255, 255, 255), (sb / 255.0) * (1.0 - sky_t))

                # Sol / Luna y halo atmosférico
                dx_sun = px - sun_x
                dist_sun = math.sqrt(dx_sun * dx_sun + dy_sun * dy_sun)
                if dist_sun <= sun_r:
                    # Efecto synthwave de franjas si es cyberpunk
                    if scene["palette_key"] == "cyberpunk" and dy_sun > 0 and (int(dy_sun) % 5 < 2):
                        col = _lerp_color(col, pal["sun"], 0.35)
                    else:
                        col = pal["sun"]
                elif dist_sun < sun_r * 2.6:
                    glow = (1.0 - (dist_sun - sun_r) / (sun_r * 1.6)) ** 2.2
                    col = _lerp_color(col, pal["sun"], glow * 0.55)

                # Nubes suaves animadas
                if 0.08 < sky_t < 0.78:
                    c_val = noise.noise2d(px / max(1.0, width) * 6.5 + t * 0.18, py / max(1.0, height) * 10.0)
                    if c_val > 0.58:
                        cloud_alpha = min(0.65, (c_val - 0.58) * 1.7)
                        col = _lerp_color(col, pal["sky_bot"], cloud_alpha)

                # Montañas lejanas
                if py >= far_mtn[px]:
                    fog = min(0.55, (py - far_mtn[px]) / max(1.0, height * 0.22))
                    col = _lerp_color(pal["mtn_far"], pal["sky_bot"], fog * 0.35)

                # Montañas cercanas
                if py >= near_mtn[px]:
                    col = pal["mtn_near"]

                # Edificios iluminados (ciudad)
                b_top, b_id = buildings[px]
                if b_id > 0 and py >= b_top:
                    col = (14, 16, 28)
                    # Ventanas encendidas y neones
                    if (px % 5 in (1, 2)) and ((py - b_top) % 6 in (2, 3)) and py < horizon_y - 3:
                        win_hash = ((px // 5) * 37 + ((py - b_top) // 6) * 19 + b_id * 13) & 15
                        if win_hash < 9:
                            pulse = 0.75 + 0.25 * math.sin(t * 2.5 + win_hash)
                            win_col = pal["accent"] if (win_hash % 3 == 0) else pal["sun"]
                            col = _lerp_color(col, win_col, pulse)

                # Bosque en primer plano
                if trees[px] > 0 and py >= horizon_y - trees[px]:
                    tree_col = (12, 28, 22) if scene["palette_key"] != "snow" else (185, 210, 230)
                    col = tree_col

                row[px] = col

        # Parte inferior (horizon_y .. height): Agua con reflejo ondulado o suelo con perspectiva
        bottom_span = max(1, height - horizon_y)
        for py in range(horizon_y, height):
            depth = (py - horizon_y) / bottom_span
            row = pixels[py]
            if scene["has_water"]:
                wave_amp = 1.8 + 7.5 * depth
                persp_phase = 28.0 / (0.14 + depth) - t * 4.5
                for px in range(width):
                    n_pert = noise.noise2d(px * 0.045, depth * 12.0 - t * 0.8) * 3.2
                    wave = math.sin(persp_phase + n_pert)
                    sx = int(max(0, min(width - 1, px + wave * wave_amp)))
                    ref_y = max(0, horizon_y - 1 - int((py - horizon_y) * 0.88))
                    ref_col = pixels[ref_y][sx]
                    water_mix = _lerp_color(ref_col, pal["water"], 0.36 + 0.40 * depth)
                    # Destello solar/lunar sobre el camino de reflejo en el agua
                    sun_spread = sun_r * (0.65 + depth * 1.35) * (0.7 + 0.3 * n_pert)
                    if abs(px - sun_x) < sun_spread and wave > 0.35:
                        shimmer = (1.0 - abs(px - sun_x) / max(1.0, sun_spread)) * (wave - 0.35) * 1.3
                        water_mix = _lerp_color(water_mix, pal["sun"], min(0.88, shimmer))
                    elif scene["palette_key"] == "cyberpunk" and wave > 0.88 and (px % 7 < 3):
                        water_mix = _lerp_color(water_mix, pal["accent"], 0.38)
                    row[px] = water_mix
            else:
                for px in range(width):
                    g_noise = noise.noise2d(px * 0.08, py * 0.12 - t * 0.4)
                    ground = _lerp_color(pal["mtn_near"], pal["water"], depth * 0.7 + g_noise * 0.2)
                    row[px] = ground

        # Partículas dinámicas (nieve, lluvia o chispas de fuego)
        if scene["has_particles"]:
            p_rng = random.Random(scene["seed"] + 777)
            count = (width * height) // 140
            for i in range(count):
                bx = p_rng.uniform(0, width)
                by = p_rng.uniform(0, height)
                speed = p_rng.uniform(28.0, 58.0)
                if scene["is_rain"]:
                    px = int((bx + t * speed * 0.35) % width)
                    py = int((by + t * speed * 1.8) % height)
                    for dy in range(4):
                        if py + dy < height:
                            pixels[py + dy][px] = _lerp_color(pixels[py + dy][px], (180, 215, 255), 0.65)
                elif scene["palette_key"] == "fire":
                    px = int((bx + math.sin(t * 2.0 + i) * 6.0) % width)
                    py = int((by - t * speed) % height)
                    pixels[py][px] = (255, 210, 90)
                else:
                    px = int((bx + math.sin(t * 1.5 + i) * 8.0) % width)
                    py = int((by + t * speed * 0.7) % height)
                    pixels[py][px] = (248, 252, 255)

        return pixels

    def _run_stable_diffusion_cpp(
        self, prompt: str, out_path: Path, width: int, height: int, seed: Optional[int]
    ) -> bool:
        """Ejecuta stable-diffusion.cpp local si el binario y un modelo están instalados."""
        sd_bin = self.model_manager.find_sd_binary()
        sd_models = self.model_manager.list_installed_sd_models()
        if not sd_bin or not sd_models:
            return False

        model_path = sd_models[0]
        cmd = [
            str(sd_bin),
            "-m",
            str(model_path),
            "-p",
            prompt,
            "-o",
            str(out_path),
            "-W",
            str(width),
            "-H",
            str(height),
            "-t",
            str(self.config.threads),
            "--steps",
            "12",
        ]
        if seed is not None:
            cmd.extend(["-s", str(seed)])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return proc.returncode == 0 and out_path.exists()
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        out_path: Optional[Path] = None,
        force_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Genera una imagen PNG 100% local a partir del prompt."""
        w = width or self.config.image_width
        h = height or self.config.image_height
        ts = time.strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() else "_" for c in strip_accents(prompt)[:28]).strip("_") or "arte"
        target = Path(out_path) if out_path else (self.config.images_dir / f"img_{ts}_{slug}.png")
        target.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        if force_mode is None and self._run_stable_diffusion_cpp(prompt, target, w, h, seed):
            pixels = read_png_rgb(target)
            engine_used = "stable-diffusion.cpp (Local)"
            scene_info = {"mode": "diffusion", "palette_key": "sd"}
        else:
            scene_info = parse_visual_scene(prompt, seed=seed)
            if force_mode in ("cppn", "fractal", "space", "landscape"):
                scene_info["mode"] = force_mode
            pixels = self.render_scene_frame(scene_info, w, h, t=0.0)
            write_png_rgb(target, pixels)
            engine_used = f"Sintetizador Propio ({scene_info['mode'].upper()} · {scene_info['palette_key']})"

        elapsed = round(time.perf_counter() - t0, 3)
        mirrored = self.config.mirror_media_if_android(target, media_type="image")

        return {
            "path": str(target),
            "android_gallery_path": str(mirrored) if mirrored else None,
            "width": w,
            "height": h,
            "engine": engine_used,
            "elapsed_sec": elapsed,
            "pixels": pixels,
            "scene": scene_info,
        }

    def apply_filter(self, input_path: Path, filter_name: str, out_path: Optional[Path] = None) -> Dict[str, Any]:
        """Aplica filtros locales (bordes Sobel, cyberpunk, gris, invertir, sepia) a cualquier PNG."""
        src = Path(input_path).expanduser().resolve()
        pixels = read_png_rgb(src)
        h = len(pixels)
        w = len(pixels[0])
        fname = filter_name.strip().lower()

        out: RGBBuffer = [[(0, 0, 0)] * w for _ in range(h)]

        if fname == "bordes":
            lum = [[int(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in row] for row in pixels]
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    gx = (
                        -lum[y - 1][x - 1] + lum[y - 1][x + 1]
                        - 2 * lum[y][x - 1] + 2 * lum[y][x + 1]
                        - lum[y + 1][x - 1] + lum[y + 1][x + 1]
                    )
                    gy = (
                        -lum[y - 1][x - 1] - 2 * lum[y - 1][x] - lum[y - 1][x + 1]
                        + lum[y + 1][x - 1] + 2 * lum[y + 1][x] + lum[y + 1][x + 1]
                    )
                    mag = min(255, int(math.sqrt(gx * gx + gy * gy)))
                    out[y][x] = (int(mag * 0.4), int(mag * 0.9), mag)
        else:
            for y in range(h):
                for x in range(w):
                    r, g, b = pixels[y][x]
                    if fname == "gris":
                        v = int(0.299 * r + 0.587 * g + 0.114 * b)
                        out[y][x] = (v, v, v)
                    elif fname == "invertir":
                        out[y][x] = (255 - r, 255 - g, 255 - b)
                    elif fname == "sepia":
                        tr = min(255, int(0.393 * r + 0.769 * g + 0.189 * b))
                        tg = min(255, int(0.349 * r + 0.686 * g + 0.168 * b))
                        tb = min(255, int(0.272 * r + 0.534 * g + 0.131 * b))
                        out[y][x] = (tr, tg, tb)
                    else:  # cyberpunk
                        nr = min(255, int(r * 1.25 + 18))
                        ng = int(g * 0.72)
                        nb = min(255, int(b * 1.35 + 32))
                        out[y][x] = (nr, ng, nb)

        dest = Path(out_path) if out_path else src.with_name(f"{src.stem}_{fname}.png")
        write_png_rgb(dest, out)
        return {"path": str(dest), "width": w, "height": h, "pixels": out, "filter": fname}

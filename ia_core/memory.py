"""
Memoria persistente vectorial y episódica en SQLite (100% local, cero dependencias externas).
Implementa:
- Búsqueda híbrida BM25 + Similitud Coseno de trigramas (resistente a errores tipográficos en móvil)
- Perfil de usuario y hechos aprendidos (/aprender)
- Indexación de documentos y carpetas locales (/indexar)
- Historial de conversaciones y caché offline de búsquedas web
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import get_repo_root

STOPWORDS_ES = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "mas", "pero",
    "sus", "le", "ya", "o", "este", "si", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "tambien", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mi", "antes", "algunos",
    "que", "es", "son", "fue", "era", "eres", "soy", "cual", "como", "puedes",
    "hola", "dime", "explica", "hacer", "hace",
}


def strip_accents(text: str) -> str:
    """Elimina tildes conservando ñ/letras base para normalización de búsqueda."""
    norm = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def stem_es(word: str) -> str:
    """Stemmer ligero en español para unificar variaciones morfológicas."""
    if len(word) <= 4:
        return word
    for suffix in ("aciones", "iciones", "amiento", "imiento", "mente", "acion", "icion", "ando", "iendo"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    if word.endswith("es") and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and len(word) > 4:
        return word[:-1]
    return word


def tokenize(text: str, remove_stops: bool = True) -> List[str]:
    """Tokeniza y normaliza texto para indexación BM25."""
    clean = strip_accents(text)
    words = re.findall(r"[a-z0-9_]{2,}", clean)
    result: List[str] = []
    for w in words:
        if remove_stops and w in STOPWORDS_ES:
            continue
        result.append(stem_es(w))
    return result


def char_trigrams(text: str) -> Counter:
    """Genera vector disperso de trigramas de caracteres para similitud semántica/difusa."""
    clean = re.sub(r"\s+", " ", strip_accents(text)).strip()
    if len(clean) < 3:
        return Counter([clean]) if clean else Counter()
    padded = f" {clean} "
    return Counter(padded[i : i + 3] for i in range(len(padded) - 2))


def cosine_counter(a: Counter, b: Counter) -> float:
    """Similitud coseno exacta entre dos vectores dispersos Counter."""
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class LocalMemory:
    """Base de memoria vectorial, episódica y caché offline respaldada en SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._seed_base_knowledge_if_empty()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_hash TEXT UNIQUE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT DEFAULT 'usuario',
                    tokens_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    module TEXT DEFAULT 'texto',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS web_cache (
                    query_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def _seed_base_knowledge_if_empty(self) -> None:
        """Carga el corpus semilla en español la primera vez para funcionar 100% offline."""
        if self.count_knowledge() > 0:
            return
        corpus_path = get_repo_root() / "data" / "corpus_base_es.txt"
        if not corpus_path.exists():
            return
        raw = corpus_path.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"\n(?=\[TEMA:)", raw.strip())
        for block in blocks:
            lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
            if not lines:
                continue
            header = lines[0]
            if header.startswith("[TEMA:") and header.endswith("]"):
                title = header[6:-1].strip()
                body = "\n".join(lines[1:]).strip()
            else:
                title = "Conocimiento Base"
                body = "\n".join(lines)
            if body:
                self.add_knowledge(title=title, content=body, source="corpus_base")

    def count_knowledge(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM knowledge").fetchone()
            return int(row["c"]) if row else 0

    def add_knowledge(self, title: str, content: str, source: str = "usuario") -> bool:
        """Guarda un fragmento de conocimiento evitando duplicados exactos."""
        content = content.strip()
        if not content:
            return False
        doc_hash = hashlib.sha256(f"{title}::{content}".encode("utf-8")).hexdigest()[:24]
        tokens = Counter(tokenize(f"{title} {content}"))
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO knowledge (doc_hash, title, content, source, tokens_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (doc_hash, title.strip(), content, source, json.dumps(tokens), time.time()),
                )
                return True
            except Exception:
                return False

    def index_path(self, target: Path, max_chunk_chars: int = 900) -> Tuple[int, int]:
        """Indexa un archivo o directorio completo de texto/código en la memoria local."""
        target = Path(target).expanduser().resolve()
        if not target.exists():
            return (0, 0)

        allowed_exts = {
            ".txt", ".md", ".py", ".sh", ".json", ".yaml", ".yml",
            ".js", ".ts", ".html", ".css", ".c", ".cpp", ".h", ".rs", ".go",
        }
        files: List[Path] = []
        if target.is_file():
            files = [target]
        else:
            for p in sorted(target.rglob("*")):
                if any(part.startswith(".") or part in ("__pycache__", "node_modules", "venv") for part in p.parts):
                    continue
                if p.is_file() and p.suffix.lower() in allowed_exts and p.stat().st_size <= 512_000:
                    files.append(p)

        files_count = 0
        chunks_count = 0
        for fpath in files:
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text:
                continue
            files_count += 1
            paragraphs = re.split(r"\n\s*\n", text)
            buf = ""
            part_idx = 1
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(buf) + len(para) + 2 <= max_chunk_chars:
                    buf = f"{buf}\n\n{para}".strip()
                else:
                    if buf:
                        self.add_knowledge(
                            title=f"{fpath.name} (parte {part_idx})",
                            content=buf,
                            source=str(fpath),
                        )
                        chunks_count += 1
                        part_idx += 1
                    buf = para[:max_chunk_chars]
            if buf:
                self.add_knowledge(
                    title=f"{fpath.name} (parte {part_idx})",
                    content=buf,
                    source=str(fpath),
                )
                chunks_count += 1

        return (files_count, chunks_count)

    def search_knowledge(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Recuperación híbrida BM25 + Similitud Coseno de trigramas.
        Funciona en milisegundos en Termux sin requerir GPU ni librerías C++.
        """
        q_tokens = tokenize(query)
        q_trigrams = char_trigrams(query)

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, content, source, tokens_json, created_at FROM knowledge"
            ).fetchall()

        if not rows:
            return []

        docs: List[Tuple[sqlite3.Row, Counter, int]] = []
        df: Counter = Counter()
        total_len = 0

        for r in rows:
            tok_map = Counter(json.loads(r["tokens_json"]))
            doc_len = max(1, sum(tok_map.values()))
            total_len += doc_len
            for tok in tok_map.keys():
                df[tok] += 1
            docs.append((r, tok_map, doc_len))

        n_docs = len(docs)
        avgdl = total_len / max(1, n_docs)
        k1 = 1.5
        b = 0.75

        scored: List[Tuple[float, sqlite3.Row]] = []
        for r, tok_map, doc_len in docs:
            bm25 = 0.0
            for qt in q_tokens:
                tf = tok_map.get(qt, 0)
                if tf == 0:
                    continue
                n_q = df.get(qt, 0)
                idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
                denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
                bm25 += idf * ((tf * (k1 + 1.0)) / denom)

            # Similitud difusa por trigramas sobre título + inicio del contenido
            tri_sim = cosine_counter(q_trigrams, char_trigrams(f"{r['title']} {r['content'][:350]}"))
            score = bm25 + (tri_sim * 2.2)
            if score > 0.18:
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: List[Dict[str, Any]] = []
        for score, r in scored[:top_k]:
            results.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "content": r["content"],
                    "source": r["source"],
                    "score": round(score, 3),
                }
            )
        return results

    def set_fact(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO facts (key, value, updated_at) VALUES (?, ?, ?)",
                (key.strip().lower(), value.strip(), time.time()),
            )

    def get_facts(self) -> Dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM facts ORDER BY updated_at DESC").fetchall()
            return {r["key"]: r["value"] for r in rows}

    def log_message(self, role: str, content: str, module: str = "texto") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (role, content, module, created_at) VALUES (?, ?, ?, ?)",
                (role, content[:4000], module, time.time()),
            )

    def recent_messages(self, limit: int = 8) -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, module FROM messages ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = [{"role": r["role"], "content": r["content"], "module": r["module"]} for r in rows]
        out.reverse()
        return out

    def clear_messages(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages")

    def cache_web(self, query_key: str, payload: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO web_cache (query_key, payload_json, created_at) VALUES (?, ?, ?)",
                (query_key.strip().lower(), json.dumps(payload, ensure_ascii=False), time.time()),
            )

    def get_cached_web(self, query_key: str, max_age_sec: float = 86400 * 7) -> Optional[Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, created_at FROM web_cache WHERE query_key = ?",
                (query_key.strip().lower(),),
            ).fetchone()
        if not row:
            return None
        if time.time() - float(row["created_at"]) > max_age_sec:
            return None
        try:
            return json.loads(row["payload_json"])
        except Exception:
            return None

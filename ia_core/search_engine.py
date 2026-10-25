"""
Motor de Búsqueda Híbrido (100% sin API keys):
1. Búsqueda Local de Archivos y Código (nombres + contenido con extractos de línea)
2. Búsqueda en Memoria Vectorial Propia (SQLite RAG)
3. Búsqueda Web Abierta (Wikipedia ES/EN + DuckDuckGo HTML) con Caché Offline automático
4. Extractor limpio de páginas web para lectura y aprendizaje local
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory import LocalMemory, strip_accents, tokenize

USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 14; Termux) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"
)


class _CleanHTMLTextExtractor(HTMLParser):
    """Extrae texto legible eliminando scripts, estilos y navegación."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "header", "footer", "nav", "aside"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        t = tag.lower()
        if t in self.SKIP_TAGS:
            self._skip_depth += 1
        elif t in ("p", "br", "div", "li", "h1", "h2", "h3", "tr"):
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            cleaned = data.strip()
            if cleaned:
                self.chunks.append(cleaned + " ")

    def get_text(self) -> str:
        raw = html.unescape("".join(self.chunks))
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _unwrap_ddg_url(url: str) -> str:
    if "uddg=" in url:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return urllib.parse.unquote(qs["uddg"][0])
    if url.startswith("//"):
        return "https:" + url
    return url


class SearchEngine:
    """Buscador unificado local y web con persistencia offline."""

    TEXT_EXTENSIONS = {
        ".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
        ".js", ".ts", ".html", ".css", ".c", ".cpp", ".h", ".rs", ".go", ".java", ".kt",
    }

    def __init__(self, memory: LocalMemory):
        self.memory = memory

    def search_local_files(
        self,
        query: str,
        root_dir: Optional[Path] = None,
        max_results: int = 8,
    ) -> List[Dict[str, Any]]:
        """Busca coincidencias por nombre y contenido en archivos locales."""
        base = Path(root_dir or Path.cwd()).expanduser().resolve()
        if not base.exists():
            return []

        q_norm = strip_accents(query.strip())
        q_tokens = tokenize(query, remove_stops=False)
        if not q_tokens and q_norm:
            q_tokens = [q_norm]

        hits: List[Dict[str, Any]] = []
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".cache"}

        for path in base.rglob("*"):
            if any(part in skip_dirs for part in path.parts):
                continue
            if not path.is_file():
                continue

            score = 0.0
            fname_norm = strip_accents(path.name)
            for tok in q_tokens:
                if tok in fname_norm:
                    score += 3.0

            snippet = ""
            line_no = 0
            try:
                if path.suffix.lower() in self.TEXT_EXTENSIONS and path.stat().st_size <= 600_000:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    lines = content.splitlines()
                    for idx, line in enumerate(lines, start=1):
                        ln_norm = strip_accents(line)
                        matched = sum(1 for tok in q_tokens if tok in ln_norm)
                        if matched > 0:
                            score += matched * 1.2
                            if not snippet:
                                line_no = idx
                                snippet = line.strip()[:160]
            except Exception:
                pass

            if score > 0:
                try:
                    rel = str(path.relative_to(base))
                except ValueError:
                    rel = str(path)
                hits.append(
                    {
                        "type": "archivo_local",
                        "title": rel,
                        "path": str(path),
                        "line": line_no,
                        "snippet": snippet or f"Archivo coincidente ({path.stat().st_size} bytes)",
                        "score": round(score, 2),
                    }
                )

        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:max_results]

    def search_wikipedia(self, query: str, lang: str = "es", limit: int = 3) -> List[Dict[str, Any]]:
        """Consulta la API abierta de Wikipedia sin claves."""
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(limit),
                "utf8": "1",
                "format": "json",
            }
        )
        api_url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
        req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
        results: List[Dict[str, Any]] = []
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))

        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            raw_snip = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            snip = html.unescape(raw_snip).strip()
            page_url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            results.append(
                {
                    "type": "wikipedia",
                    "title": f"{title} (Wikipedia {lang.upper()})",
                    "url": page_url,
                    "snippet": snip,
                }
            )
        return results

    def search_duckduckgo(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Consulta DuckDuckGo HTML sin API keys ni rastreadores."""
        body = urllib.parse.urlencode({"q": query, "kl": "es-es"}).encode("utf-8")
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=body,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")

        results: List[Dict[str, Any]] = []
        blocks = re.findall(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]*class="[^"]*result__a|$)',
            raw_html,
            flags=re.S | re.I,
        )
        for href, raw_title, tail in blocks[:limit]:
            clean_title = html.unescape(re.sub(r"<[^>]+>", "", raw_title)).strip()
            snip_match = re.search(
                r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
                tail,
                flags=re.S | re.I,
            )
            clean_snip = ""
            if snip_match:
                clean_snip = html.unescape(re.sub(r"<[^>]+>", "", snip_match.group(1))).strip()
            clean_url = _unwrap_ddg_url(html.unescape(href))
            if clean_title and clean_url.startswith("http"):
                results.append(
                    {
                        "type": "web",
                        "title": clean_title,
                        "url": clean_url,
                        "snippet": clean_snip,
                    }
                )
        return results

    def fetch_clean_page(self, url: str, max_chars: int = 4000, learn: bool = False) -> Dict[str, Any]:
        """Descarga cualquier página web y extrae su texto limpio."""
        cached = self.memory.get_cached_web(f"url::{url}")
        if cached:
            return cached

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")

        title_m = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.S | re.I)
        title = html.unescape(title_m.group(1)).strip() if title_m else url

        parser = _CleanHTMLTextExtractor()
        parser.feed(raw)
        clean_text = parser.get_text()[:max_chars]

        payload = {"url": url, "title": title, "content": clean_text}
        self.memory.cache_web(f"url::{url}", payload)
        if learn and clean_text:
            self.memory.add_knowledge(title=title[:100], content=clean_text[:1500], source=url)
        return payload

    def search(
        self,
        query: str,
        mode: str = "todo",
        root_dir: Optional[Path] = None,
        auto_learn_web: bool = True,
    ) -> Dict[str, Any]:
        """
        Ejecuta búsqueda combinada según `mode`:
        - 'local': archivos locales + memoria propia (100% offline)
        - 'web': Wikipedia + DuckDuckGo (con fallback a caché offline si no hay red)
        - 'todo': combina memoria, archivos locales y web
        """
        out: Dict[str, Any] = {
            "query": query,
            "memory_hits": [],
            "local_files": [],
            "web_hits": [],
            "web_status": "omitido",
        }

        if mode in ("todo", "local"):
            out["memory_hits"] = self.memory.search_knowledge(query, top_k=4)
            out["local_files"] = self.search_local_files(query, root_dir=root_dir, max_results=6)

        if mode in ("todo", "web"):
            web_hits: List[Dict[str, Any]] = []
            errors: List[str] = []

            try:
                web_hits.extend(self.search_wikipedia(query, lang="es", limit=2))
            except Exception as e:
                errors.append(f"Wikipedia: {type(e).__name__}")

            try:
                web_hits.extend(self.search_duckduckgo(query, limit=4))
            except Exception as e:
                errors.append(f"DuckDuckGo: {type(e).__name__}")

            if web_hits:
                out["web_hits"] = web_hits
                out["web_status"] = "online"
                self.memory.cache_web(f"search::{query}", web_hits)
                if auto_learn_web:
                    combined = "\n".join(
                        f"{h['title']}: {h['snippet']}" for h in web_hits[:3] if h.get("snippet")
                    )
                    if combined:
                        self.memory.add_knowledge(
                            title=f"Búsqueda web: {query}",
                            content=combined,
                            source="busqueda_web_cache",
                        )
            else:
                cached = self.memory.get_cached_web(f"search::{query}")
                if cached:
                    out["web_hits"] = cached
                    out["web_status"] = "cache_offline"
                else:
                    out["web_status"] = "sin_conexion" if errors else "sin_resultados"

        return out

"""
Motor de Texto Dual 100% Local:
1. GGUF Local Runner (llama.cpp en CPU ARM64/x86) enriquecido con Memoria RAG propia
2. Motor Cognitivo + Red Neuronal Propia (PropiaNeuralLM + Evaluador AST matemático +
   Resumidor extractivo TF-IDF + Síntesis RAG + Generador de código y aprendizaje continuo)
"""

from __future__ import annotations

import ast
import datetime
import math
import operator
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import IAConfig, get_system_resources
from .memory import LocalMemory, strip_accents, tokenize
from .model_manager import ModelManager
from .neural_lm import PropiaNeuralLM

SAFE_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "raiz": math.sqrt,
    "sin": math.sin,
    "sen": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "factorial": math.factorial,
    "pi": math.pi,
    "e": math.e,
}

SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def safe_eval_math(expr: str) -> Optional[float]:
    """Evalúa expresiones matemáticas de forma 100% segura mediante AST."""
    cleaned = expr.strip().lower()
    cleaned = cleaned.replace("^", "**").replace("×", "*").replace("÷", "/")
    cleaned = re.sub(r"(\d)\s*,\s*(\d)", r"\1.\2", cleaned)

    # Extraer subexpresión matemática si viene acompañada de palabras ("cuánto es 25 * 14 + sqrt(16)")
    m = re.search(
        r"(?:cuanto\s+es|calcula|resuelve|evalua|resultado\s+de)\s+([0-9a-z_+\-*/().\s*]+)",
        strip_accents(cleaned),
    )
    candidate = m.group(1).strip() if m else cleaned
    candidate = candidate.rstrip("?= ")

    if not re.search(r"\d", candidate):
        return None
    if not re.fullmatch(r"[0-9a-z_+\-*/().,\s]+", candidate):
        return None
    # Debe tener al menos un operador o llamada a función
    if not any(op in candidate for op in ("+", "-", "*", "/", "(", "%")):
        return None

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in SAFE_MATH_FUNCS and isinstance(SAFE_MATH_FUNCS[node.id], (int, float)):
                return float(SAFE_MATH_FUNCS[node.id])
            raise ValueError("Identificador no permitido")
        if isinstance(node, ast.UnaryOp):
            val = _eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError("Operador unario no soportado")
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BIN_OPS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponente demasiado grande")
            return float(SAFE_BIN_OPS[type(node.op)](left, right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = SAFE_MATH_FUNCS.get(node.func.id)
            if callable(fn):
                args = [_eval_node(a) for a in node.args]
                return float(fn(*args))
        raise ValueError("Nodo AST no soportado")

    try:
        tree = ast.parse(candidate, mode="eval")
        res = _eval_node(tree)
        if math.isnan(res) or math.isinf(res):
            return None
        return res
    except Exception:
        return None


def summarize_text_extractive(text: str, max_sentences: int = 4, query: str = "") -> str:
    """Resumidor extractivo propio basado en centralidad TF y coincidencia semántica con la consulta."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if len(s.strip()) > 18]
    if not sents:
        return text.strip()[:500]

    q_tokens = set(tokenize(query)) if query else set()
    if q_tokens:
        matched_sents = [s for s in sents if set(tokenize(s)) & q_tokens]
        if matched_sents and len(matched_sents) <= max_sentences:
            return " ".join(matched_sents)

    if len(sents) <= max_sentences:
        return " ".join(sents)

    word_freq = Counter(tokenize(text))
    if not word_freq:
        return " ".join(sents[:max_sentences])

    scored: List[Tuple[float, int, str]] = []
    for idx, sent in enumerate(sents):
        toks = tokenize(sent)
        if not toks:
            continue
        tf_score = sum(word_freq[t] for t in toks) / (len(toks) ** 0.75)
        q_overlap = len(set(toks) & q_tokens) * 4.5 if q_tokens else 0.0
        pos_bonus = 1.15 if idx == 0 else 1.0
        scored.append(((tf_score + q_overlap) * pos_bonus, idx, sent))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
    top_ordered = [s for _, _, s in sorted(top, key=lambda x: x[1])]
    return " ".join(top_ordered)


class TextEngine:
    """Cerebro principal de generación, razonamiento, memoria y entrenamiento de texto."""

    def __init__(self, config: IAConfig, memory: LocalMemory, model_manager: ModelManager):
        self.config = config
        self.memory = memory
        self.model_manager = model_manager
        self.neural_lm = PropiaNeuralLM(self.config.brain_dir / "neural_lm_weights.json")

    def active_engine_label(self) -> str:
        """Etiqueta legible del motor actualmente activo para mostrar en el banner de Termux."""
        if self.config.text_backend != "propio":
            llama_bin = self.model_manager.find_llama_binary()
            gguf = self.model_manager.get_active_gguf_path()
            if llama_bin and gguf:
                return f"GGUF Local ({gguf.stem[:20]})"
        return f"Motor Neuronal Propio ({len(self.neural_lm.vocab)} tokens)"

    def has_local_gguf_ready(self) -> bool:
        if self.config.text_backend == "propio":
            return False
        return (
            self.model_manager.find_llama_binary() is not None
            and self.model_manager.get_active_gguf_path() is not None
        )

    def learn_fact_or_text(self, text: str, title: str = "Aprendizaje directo") -> Dict[str, Any]:
        """Guarda conocimiento en la base RAG y entrena la red neuronal propia."""
        saved = self.memory.add_knowledge(title=title, content=text, source="entrenamiento_usuario")
        stats = self.neural_lm.train_text(text, epochs=4, lr=0.07)
        return {"saved_rag": saved, "neural_stats": stats}

    def train_from_file(self, file_path: Path, epochs: int = 6, progress_cb=None) -> Dict[str, Any]:
        """Indexa un archivo completo en RAG y entrena la red neuronal propia sobre su contenido."""
        p = Path(file_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"No existe el archivo: {p}")
        content = p.read_text(encoding="utf-8", errors="ignore")
        files_n, chunks_n = self.memory.index_path(p)
        stats = self.neural_lm.train_text(content, epochs=epochs, progress_cb=progress_cb)
        stats["indexed_files"] = files_n
        stats["indexed_chunks"] = chunks_n
        return stats

    def _run_llama_cpp(self, prompt: str, system_prompt: str, max_tokens: Optional[int] = None) -> Optional[str]:
        """Ejecuta inferencia 100% offline mediante llama.cpp con formato ChatML."""
        llama_bin = self.model_manager.find_llama_binary()
        gguf_path = self.model_manager.get_active_gguf_path()
        if not llama_bin or not gguf_path:
            return None

        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        n_toks = max_tokens or self.config.max_tokens
        cmd = [
            str(llama_bin),
            "-m",
            str(gguf_path),
            "-p",
            full_prompt,
            "-n",
            str(n_toks),
            "-t",
            str(self.config.threads),
            "-c",
            str(self.config.context_size),
            "--temp",
            str(self.config.temperature),
            "--no-display-prompt",
            "-r",
            "<|im_end|>",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if proc.returncode != 0:
                return None
            out = proc.stdout.strip()
            out = out.replace("<|im_end|>", "").replace("<|eot_id|>", "").strip()
            return out if out else None
        except Exception:
            return None

    def generate_raw_llm(self, prompt: str, system_prompt: str = "", max_tokens: int = 384) -> Optional[str]:
        """Usado por el Agente ReAct cuando hay un modelo GGUF local activo."""
        if not self.has_local_gguf_ready():
            return None
        return self._run_llama_cpp(prompt, system_prompt, max_tokens=max_tokens)

    def _check_personal_memory_intent(self, prompt: str) -> Optional[str]:
        """Detecta enseñanzas directas del usuario en lenguaje natural."""
        norm = prompt.strip()
        low = strip_accents(norm)

        # Nombre del usuario
        m_name = re.search(r"\b(?:me llamo|mi nombre es)\s+([a-zA-ZáéíóúñÁÉÍÓÚÑ0-9_ -]{2,30})", norm, re.I)
        if m_name:
            name = m_name.group(1).strip(" .,!?")
            self.memory.set_fact("nombre_usuario", name)
            self.config.user_name = name
            self.config.save()
            return f"¡Mucho gusto, **{name}**! Lo he guardado en mi memoria local permanente."

        if re.search(r"\b(?:como me llamo|cual es mi nombre|sabes quien soy)\b", low):
            facts = self.memory.get_facts()
            saved_name = facts.get("nombre_usuario") or (
                self.config.user_name if self.config.user_name != "Usuario" else None
            )
            if saved_name:
                return f"Te llamas **{saved_name}**. Lo recuerdo en mi base de memoria local."
            return "Aún no me has dicho tu nombre. Puedes decirme *«Me llamo ...»* y lo recordaré siempre."

        # Aprendizaje explícito ("recuerda que ...", "aprende que ...")
        m_learn = re.match(r"^(?:recuerda que|aprende que|guarda que)\s+(.+)$", norm, re.I)
        if m_learn:
            fact_text = m_learn.group(1).strip()
            info = self.learn_fact_or_text(fact_text, title="Nota del usuario")
            return (
                f"He aprendido y guardado en mi memoria local:\n"
                f"> *{fact_text}*\n\n"
                f"Red neuronal actualizada (`vocabulario: {info['neural_stats']['vocab_size']} tokens`)."
            )

        return None

    def _generate_code_template(self, prompt: str) -> Optional[str]:
        """Genera plantillas limpias de código local si el usuario pide un programa en modo propio."""
        low = strip_accents(prompt)
        if not any(
            kw in low
            for kw in (
                "escribe un codigo", "crea un script", "haz un programa",
                "codigo en python", "script en python", "funcion en python",
                "script bash", "ejemplo de codigo",
            )
        ):
            return None

        if "primo" in low:
            return (
                "Aquí tienes una implementación eficiente en **Python** para verificar y listar números primos:\n\n"
                "```python\n"
                "import math\n\n"
                "def es_primo(n: int) -> bool:\n"
                "    if n < 2:\n"
                "        return False\n"
                "    if n in (2, 3):\n"
                "        return True\n"
                "    if n % 2 == 0:\n"
                "        return False\n"
                "    limite = int(math.isqrt(n))\n"
                "    return all(n % d != 0 for d in range(3, limite + 1, 2))\n\n"
                "if __name__ == '__main__':\n"
                "    primos = [x for x in range(1, 101) if es_primo(x)]\n"
                "    print('Primos hasta 100:', primos)\n"
                "```\n\n"
                "Si quieres que lo cree en un archivo y lo ejecute automáticamente, usa: `/agente "
                + prompt
                + "`"
            )

        if "calculadora" in low:
            return (
                "Aquí tienes una **calculadora interactiva** lista para Termux en Python:\n\n"
                "```python\n"
                "import math\n\n"
                "def calcular(expr: str) -> float:\n"
                "    permitidos = {'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'pi': math.pi}\n"
                "    return eval(expr, {'__builtins__': {}}, permitidos)\n\n"
                "if __name__ == '__main__':\n"
                "    print('Calculadora lista (escribe salir para terminar)')\n"
                "    while True:\n"
                "        op = input('calc> ').strip()\n"
                "        if op.lower() in ('salir', 'exit'):\n"
                "            break\n"
                "        try:\n"
                "            print('=', calcular(op))\n"
                "        except Exception as e:\n"
                "            print('Error:', e)\n"
                "```"
            )

        if "servidor" in low or "http" in low or "web" in low:
            return (
                "Puedes levantar un **servidor HTTP local** en Termux con Python puro:\n\n"
                "```python\n"
                "from http.server import HTTPServer, BaseHTTPRequestHandler\n\n"
                "class MiServidor(BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        self.send_response(200)\n"
                "        self.send_header('Content-Type', 'text/html; charset=utf-8')\n"
                "        self.end_headers()\n"
                "        self.wfile.write(b'<h1>Servidor activo desde Termux</h1>')\n\n"
                "if __name__ == '__main__':\n"
                "    print('Escuchando en http://0.0.0.0:8080')\n"
                "    HTTPServer(('0.0.0.0', 8080), MiServidor).serve_forever()\n"
                "```"
            )

        return (
            "Aquí tienes una estructura limpia en **Python** lista para ejecutar en Termux:\n\n"
            "```python\n"
            "#!/usr/bin/env python3\n"
            "import sys\n\n"
            "def main(args: list[str]) -> None:\n"
            f"    # Tarea solicitada: {prompt.strip()[:60]}\n"
            "    print('Ejecutando script en Termux con argumentos:', args)\n\n"
            "if __name__ == '__main__':\n"
            "    main(sys.argv[1:])\n"
            "```\n\n"
            "💡 *Consejo:* Usa `/agente` para crear, editar y probar archivos directamente en tu carpeta."
        )

    def reply(self, prompt: str, extra_context: str = "") -> Dict[str, Any]:
        """
        Genera una respuesta completa al mensaje del usuario.
        Usa llama.cpp GGUF si está instalado; de lo contrario usa el Motor Cognitivo + Neuronal Propio.
        """
        clean_prompt = prompt.strip()
        if not clean_prompt:
            return {"text": "Escribe una pregunta o comando (`/ayuda` para ver todo lo que puedo hacer).", "engine": "propio"}

        self.memory.log_message("user", clean_prompt, module="texto")

        # 1. Revisar intenciones de memoria directa ("me llamo...", "recuerda que...")
        mem_ans = self._check_personal_memory_intent(clean_prompt)
        if mem_ans:
            self.memory.log_message("assistant", mem_ans, module="texto")
            return {"text": mem_ans, "engine": "Memoria Propia"}

        # 2. Recuperar contexto relevante RAG de nuestra memoria SQLite
        rag_hits = self.memory.search_knowledge(clean_prompt, top_k=3)
        facts = self.memory.get_facts()

        # 3. Si hay un modelo GGUF local con llama.cpp activo, usarlo con RAG inyectado
        if self.has_local_gguf_ready():
            rag_text = "\n\n".join(f"- [{h['title']}]: {h['content']}" for h in rag_hits)
            facts_text = ", ".join(f"{k}={v}" for k, v in facts.items())
            sys_prompt = (
                f"Eres {self.config.ai_name}, una inteligencia artificial propia que se ejecuta 100% local "
                f"en Termux. Responde en español claro, preciso y directo."
            )
            if facts_text:
                sys_prompt += f"\nDatos recordados del usuario: {facts_text}."
            if rag_text:
                sys_prompt += f"\nContexto relevante en tu memoria local:\n{rag_text}"
            if extra_context:
                sys_prompt += f"\nInformación adicional:\n{extra_context}"

            llm_out = self._run_llama_cpp(clean_prompt, sys_prompt)
            if llm_out:
                self.memory.log_message("assistant", llm_out, module="texto")
                return {"text": llm_out, "engine": self.active_engine_label(), "rag_hits": rag_hits}

        # 4. Motor Cognitivo + Red Neuronal Propia (100% instantáneo, sin descargas externas)
        answer = self._cognitive_propio_reply(clean_prompt, rag_hits, facts, extra_context)
        self.memory.log_message("assistant", answer, module="texto")
        return {"text": answer, "engine": "Motor Neuronal Propio", "rag_hits": rag_hits}

    def _cognitive_propio_reply(
        self,
        prompt: str,
        rag_hits: List[Dict[str, Any]],
        facts: Dict[str, str],
        extra_context: str = "",
    ) -> str:
        low = strip_accents(prompt.lower().strip())
        user_greeting = f", {facts['nombre_usuario']}" if "nombre_usuario" in facts else ""

        # A. Cálculo matemático exacto AST
        math_val = safe_eval_math(prompt)
        if math_val is not None:
            formatted = int(math_val) if math_val == int(math_val) else round(math_val, 8)
            return f"Resultado del cálculo exacto:\n\n**`{formatted}`**"

        # B. Resumen de texto explícito ("resume: <texto largo>")
        m_sum = re.match(r"^(?:resume|resumir|haz un resumen de)\s*[:\-]?\s*(.+)$", prompt, re.I | re.S)
        if m_sum and len(m_sum.group(1).strip()) > 120:
            summary = summarize_text_extractive(m_sum.group(1).strip(), max_sentences=3)
            return f"**Resumen clave:**\n\n{summary}"

        # C. Fecha, hora y estado del sistema
        if any(p in low for p in ("que hora es", "que dia es", "fecha actual", "fecha de hoy")):
            now = datetime.datetime.now()
            return f"Son las **{now.strftime('%H:%M:%S')}** del **{now.strftime('%Y-%m-%d')}**."

        if any(p in low for p in ("cuanta ram", "estado del sistema", "recursos del telefono")):
            res = get_system_resources()
            return (
                f"**Estado actual de tu dispositivo ({res['arch']}):**\n"
                f"- **RAM disponible:** {res['avail_ram_mb']} MB / {res['total_ram_mb']} MB\n"
                f"- **Núcleos CPU:** {res['cpu_cores']} (hilos óptimos IA: {res['optimal_threads']})\n"
                f"- **Espacio libre:** {res['free_disk_mb']} MB\n"
                f"- **Entorno Termux:** {'Sí' if res['termux'] else 'No (Linux estándar)'}"
            )

        # D. Saludos e identidad
        if re.fullmatch(r"(?:hola|buenas|buenos dias|buenas tardes|buenas noches|hey|que tal)[!.? ]*", low):
            return (
                f"¡Hola{user_greeting}! Soy **{self.config.ai_name}**, tu inteligencia artificial propia en Termux.\n\n"
                "Tengo activos mis **5 módulos locales**:\n"
                "- **Texto:** Conversación, matemáticas, resúmenes, memoria RAG y red neuronal entrenable (`/entrenar`)\n"
                "- **Buscar:** Archivos locales y web abierta sin API keys (`/buscar`)\n"
                "- **Imágenes:** Arte neuronal CPPN y paisajes procedimentales en PNG + vista en consola (`/imagen`)\n"
                "- **Video:** Animaciones MP4/GIF/AVI con física temporal y audio (`/video`)\n"
                "- **Agente:** Programación autónoma, terminal y Termux:API (`/agente`)"
            )

        if any(
            p in low
            for p in ("quien eres", "que eres", "que puedes hacer", "cuales son tus funciones")
        ):
            return (
                f"Soy **{self.config.ai_name}**, una IA **100% propia y local** diseñada para ejecutarse en **Termux**.\n\n"
                "Combino una **Red Neuronal Propia entrenable** (`PropiaNeuralLM`) + **Memoria Vectorial SQLite (RAG)**, "
                "y también puedo ejecutar modelos **GGUF con `llama.cpp`** directamente en el procesador ARM de tu teléfono (`/modelos`).\n\n"
                "Puedo generar **Texto**, **Buscar** offline/online, crear **Imágenes**, renderizar **Videos** y actuar como **Agente**."
            )

        # E. Solicitudes de código
        code_ans = self._generate_code_template(prompt)
        if code_ans:
            return code_ans

        # F. Si viene contexto externo (ej. desde una búsqueda web reciente)
        if extra_context:
            summary = summarize_text_extractive(extra_context, max_sentences=4, query=prompt)
            return f"**Síntesis basada en la información encontrada:**\n\n{summary}"

        # G. Síntesis usando nuestra Memoria RAG + Red Neuronal Propia
        if rag_hits and rag_hits[0]["score"] >= 0.85:
            best = rag_hits[0]
            content = summarize_text_extractive(best["content"], max_sentences=3, query=prompt)
            parts = [f"{content}"]
            if len(rag_hits) > 1 and rag_hits[1]["score"] >= 1.2:
                extra = summarize_text_extractive(rag_hits[1]["content"], max_sentences=2, query=prompt)
                if extra != content:
                    parts.append(f"\n**Información relacionada ({rag_hits[1]['title']}):**\n{extra}")
            neural_cont = self.neural_lm.generate(prompt, max_words=24, temperature=self.config.temperature)
            if neural_cont and len(neural_cont) > 20 and best["source"] != "corpus_base":
                parts.append(f"\n*Síntesis neuronal:* {neural_cont}")
            return "\n".join(parts)

        # H. Generación directa con la Red Neuronal Propia + sugerencia útil de aprendizaje/búsqueda
        generated = self.neural_lm.generate(prompt, max_words=36, temperature=self.config.temperature)
        return (
            f"{generated}\n\n"
            f"💡 *Nota del motor propio:* Aún no tengo un documento específico sobre este tema en mi memoria local. "
            f"Puedes enseñármelo con `Recuerda que ...`, buscarlo e indexarlo con `/buscar {prompt[:40]}`, "
            f"entrenarme con un archivo (`/entrenar archivo.txt`) o activar un modelo GGUF completo con `/modelos`."
        )

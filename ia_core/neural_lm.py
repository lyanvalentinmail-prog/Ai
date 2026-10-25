"""
Red Neuronal de Lenguaje Propia y Entrenable (100% Python puro, sin dependencias pesadas).
Combina:
1. Red Neuronal Multicapa de Lenguaje (Embeddings aprendidos + Capa oculta tanh + Softmax con retropropagación real SGD)
2. Modelo N-grama Interpolado con Backoff de orden 4 para fluidez léxica inmediata
Permite entrenar tu propia IA directamente en Termux con `ia entrenar archivo.txt`.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .config import get_repo_root

TOKEN_PATTERN = re.compile(r"[A-Za-záéíóúüñÁÉÍÓÚÜÑ0-9_]+|[.,;:!?¿¡()\-]")


def tokenize_lm(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


class PropiaNeuralLM:
    """
    Modelo de lenguaje propio híbrido (Red Neuronal Embebida + N-grama de orden 4).
    Guarda y carga sus propios pesos sinápticos en JSON comprimido dentro de ~/.ia_propia/brain/.
    """

    def __init__(self, weights_path: Path, embed_dim: int = 20, hidden_dim: int = 32, context_len: int = 3):
        self.weights_path = Path(weights_path)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len

        self.vocab: List[str] = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.word_to_id: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}

        # Tablas N-grama (orden 1..4)
        self.unigrams: Counter = Counter()
        self.ngrams: Dict[int, Dict[str, Counter]] = {
            2: defaultdict(Counter),
            3: defaultdict(Counter),
            4: defaultdict(Counter),
        }

        # Pesos de la Red Neuronal Multicapa (se inicializan al construir vocabulario)
        self.E: List[List[float]] = []          # [V][embed_dim]
        self.W1: List[List[float]] = []         # [context_len * embed_dim][hidden_dim]
        self.b1: List[float] = []               # [hidden_dim]
        self.W2: List[List[float]] = []         # [hidden_dim][V_active]
        self.b2: List[float] = []               # [V_active]
        self.neural_vocab_limit: int = 320      # Vocabulario activo neuronal rápido en CPU móvil
        self.trained_steps: int = 0
        self.last_loss: float = 0.0

        if self.weights_path.exists():
            if not self.load():
                self._bootstrap_default()
        else:
            self._bootstrap_default()

    def _bootstrap_default(self) -> None:
        corpus_file = get_repo_root() / "data" / "corpus_base_es.txt"
        if corpus_file.exists():
            text = corpus_file.read_text(encoding="utf-8", errors="ignore")
            self.train_text(text, epochs=2, lr=0.08, max_neural_samples=450)
            self.save()

    def _init_neural_weights(self, active_v: int, seed: int = 42) -> None:
        rng = random.Random(seed)
        in_dim = self.context_len * self.embed_dim
        scale1 = 1.0 / math.sqrt(max(1, in_dim))
        scale2 = 1.0 / math.sqrt(max(1, self.hidden_dim))

        while len(self.E) < active_v:
            self.E.append([rng.uniform(-0.2, 0.2) for _ in range(self.embed_dim)])

        if len(self.W1) != in_dim or (self.W1 and len(self.W1[0]) != self.hidden_dim):
            self.W1 = [[rng.uniform(-scale1, scale1) for _ in range(self.hidden_dim)] for _ in range(in_dim)]
            self.b1 = [0.0 for _ in range(self.hidden_dim)]

        old_v = len(self.b2)
        if old_v == 0:
            self.W2 = [[rng.uniform(-scale2, scale2) for _ in range(active_v)] for _ in range(self.hidden_dim)]
            self.b2 = [0.0 for _ in range(active_v)]
        elif active_v > old_v:
            extra = active_v - old_v
            for h in range(self.hidden_dim):
                self.W2[h].extend(rng.uniform(-scale2, scale2) for _ in range(extra))
            self.b2.extend(0.0 for _ in range(extra))

    def _forward(self, ctx_ids: List[int], active_v: int) -> Tuple[List[float], List[float], List[float]]:
        """Paso hacia adelante (Forward pass): Embeddings -> Capa Oculta tanh -> Softmax."""
        x: List[float] = []
        for cid in ctx_ids:
            idx = cid if cid < active_v else 1  # <UNK>
            x.extend(self.E[idx])

        h = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            s = self.b1[j]
            for i, xi in enumerate(x):
                if xi != 0.0:
                    s += xi * self.W1[i][j]
            h[j] = math.tanh(max(-15.0, min(15.0, s)))

        logits = [0.0] * active_v
        max_logit = -1e9
        for k in range(active_v):
            s = self.b2[k]
            for j, hj in enumerate(h):
                s += hj * self.W2[j][k]
            logits[k] = s
            if s > max_logit:
                max_logit = s

        exps = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exps) or 1.0
        probs = [e / sum_exp for e in exps]
        return x, h, probs

    def train_text(
        self,
        text: str,
        epochs: int = 4,
        lr: float = 0.07,
        max_neural_samples: int = 1200,
        progress_cb: Optional[Callable[[int, int, float], None]] = None,
    ) -> Dict[str, float]:
        """
        Entrena tanto el modelo N-grama de orden 4 como la Red Neuronal Multicapa con retropropagación real.
        Retorna estadísticas de entrenamiento (pérdida inicial, pérdida final, perplejidad, vocabulario).
        """
        paragraphs = [
            p.strip()
            for p in re.split(r"\n+", text)
            if p.strip() and not p.strip().startswith("[TEMA:")
        ]
        if not paragraphs:
            return {"initial_loss": 0.0, "final_loss": 0.0, "perplexity": 1.0, "vocab_size": len(self.vocab)}

        tokenized_sents: List[List[str]] = []
        freq = Counter(self.unigrams)

        for para in paragraphs:
            toks = tokenize_lm(para.lower())
            if len(toks) < 2:
                continue
            seq = ["<BOS>"] * self.context_len + toks + ["<EOS>"]
            tokenized_sents.append(seq)
            for t in toks + ["<EOS>"]:
                freq[t] += 1

            # Actualizar tablas N-grama
            for t in toks:
                self.unigrams[t] += 1
            for order in (2, 3, 4):
                for idx in range(len(seq) - order + 1):
                    ctx = " ".join(seq[idx : idx + order - 1])
                    nxt = seq[idx + order - 1]
                    self.ngrams[order][ctx][nxt] += 1

        # Reordenar/ampliar vocabulario priorizando palabras frecuentes en ventana neuronal activa
        for word, _ in freq.most_common():
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.vocab)
                self.vocab.append(word)

        active_v = min(len(self.vocab), self.neural_vocab_limit)
        self._init_neural_weights(active_v)

        # Construir pares (contexto -> objetivo) para entrenamiento neuronal
        samples: List[Tuple[List[int], int]] = []
        for seq in tokenized_sents:
            ids = [self.word_to_id.get(w, 1) for w in seq]
            for i in range(self.context_len, len(ids)):
                target = ids[i]
                if target >= active_v:
                    continue
                ctx = [c if c < active_v else 1 for c in ids[i - self.context_len : i]]
                samples.append((ctx, target))

        if not samples:
            self.save()
            return {"initial_loss": 0.0, "final_loss": 0.0, "perplexity": 1.0, "vocab_size": len(self.vocab)}

        rng = random.Random(1337 + self.trained_steps)
        if len(samples) > max_neural_samples:
            samples = rng.sample(samples, max_neural_samples)

        initial_loss = 0.0
        final_loss = 0.0

        for ep in range(1, max(1, epochs) + 1):
            rng.shuffle(samples)
            total_loss = 0.0
            cur_lr = lr / (1.0 + 0.12 * (ep - 1))

            for ctx, target in samples:
                x, h, probs = self._forward(ctx, active_v)
                p_t = max(1e-9, probs[target])
                total_loss += -math.log(p_t)

                # Gradiente de salida (Cross-Entropy + Softmax): dL/dz = probs - one_hot(target)
                dz = list(probs)
                dz[target] -= 1.0

                # Retropropagar a W2, b2 y acumular dh
                dh = [0.0] * self.hidden_dim
                for j in range(self.hidden_dim):
                    hj = h[j]
                    row_w2 = self.W2[j]
                    s_dh = 0.0
                    for k in range(active_v):
                        g = dz[k]
                        s_dh += g * row_w2[k]
                        row_w2[k] -= cur_lr * hj * g
                    dh[j] = s_dh * (1.0 - hj * hj)  # derivada de tanh

                for k in range(active_v):
                    self.b2[k] -= cur_lr * dz[k]

                # Retropropagar a W1, b1 y tabla de Embeddings E
                dx = [0.0] * len(x)
                for i, xi in enumerate(x):
                    row_w1 = self.W1[i]
                    s_dx = 0.0
                    for j in range(self.hidden_dim):
                        g_h = dh[j]
                        s_dx += g_h * row_w1[j]
                        row_w1[j] -= cur_lr * xi * g_h
                    dx[i] = s_dx

                for j in range(self.hidden_dim):
                    self.b1[j] -= cur_lr * dh[j]

                for pos, cid in enumerate(ctx):
                    emb_row = self.E[cid]
                    offset = pos * self.embed_dim
                    for d in range(self.embed_dim):
                        emb_row[d] -= cur_lr * dx[offset + d]

                self.trained_steps += 1

            avg_loss = total_loss / len(samples)
            if ep == 1:
                initial_loss = avg_loss
            final_loss = avg_loss
            self.last_loss = final_loss
            if progress_cb:
                progress_cb(ep, epochs, final_loss)

        self.save()
        return {
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "perplexity": round(math.exp(min(10.0, final_loss)), 2),
            "vocab_size": len(self.vocab),
            "samples": len(samples),
        }

    def generate(
        self,
        seed_text: str,
        max_words: int = 45,
        temperature: float = 0.65,
        seed: Optional[int] = None,
    ) -> str:
        """
        Genera texto combinando las probabilidades de la Red Neuronal Multicapa
        con el modelo N-grama interpolado de orden 4.
        """
        rng = random.Random(seed)
        prompt_toks = [t.lower() for t in tokenize_lm(seed_text)]
        ctx_words = (["<BOS>"] * self.context_len + prompt_toks)[-self.context_len :]
        active_v = min(len(self.vocab), len(self.b2))

        generated: List[str] = []
        recent_set: List[str] = []

        for _ in range(max_words):
            scores: Dict[str, float] = defaultdict(float)

            # 1. Distribución N-grama con backoff (orden 4 -> 3 -> 2)
            for order, weight in ((4, 3.4), (3, 1.8), (2, 0.7)):
                key = " ".join(ctx_words[-(order - 1) :])
                cand_counter = self.ngrams[order].get(key)
                if cand_counter:
                    tot = sum(cand_counter.values())
                    for w, cnt in cand_counter.most_common(16):
                        scores[w] += weight * (cnt / tot)

            # 2. Distribución de la Red Neuronal Propia
            if active_v > 4:
                ctx_ids = [self.word_to_id.get(w, 1) for w in ctx_words]
                ctx_ids = [c if c < active_v else 1 for c in ctx_ids]
                _, _, probs = self._forward(ctx_ids, active_v)
                top_neural = sorted(range(4, active_v), key=lambda idx: probs[idx], reverse=True)[:16]
                for idx in top_neural:
                    scores[self.vocab[idx]] += 1.25 * probs[idx]

            if not scores:
                for w, cnt in self.unigrams.most_common(12):
                    scores[w] = float(cnt)

            # Penalización de repetición para evitar bucles
            for w in list(scores.keys()):
                if w in ("<PAD>", "<UNK>", "<BOS>"):
                    scores.pop(w, None)
                    continue
                if w in recent_set[-5:]:
                    scores[w] *= 0.25

            if not scores:
                break

            # Muestreo con temperatura
            temp = max(0.15, min(1.5, temperature))
            items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
            max_s = max(v for _, v in items)
            weights = [math.exp((v - max_s) / temp) for _, v in items]
            total_w = sum(weights) or 1.0
            pick = rng.random() * total_w
            acc = 0.0
            chosen = items[0][0]
            for (word, _), w_val in zip(items, weights):
                acc += w_val
                if acc >= pick:
                    chosen = word
                    break

            if chosen == "<EOS>":
                if len(generated) >= 8:
                    break
                continue

            generated.append(chosen)
            recent_set.append(chosen)
            ctx_words = (ctx_words + [chosen])[-self.context_len :]

        # Reconstruir puntuación limpia y mayúsculas iniciales
        text_out = ""
        capitalize_next = True
        for tok in generated:
            if tok in ".,;:!?)]":
                text_out += tok
                if tok in ".!?":
                    capitalize_next = True
            elif tok in "¿¡(":
                text_out += (" " if text_out else "") + tok
            else:
                word_fmt = tok.capitalize() if capitalize_next else tok
                capitalize_next = False
                text_out += ("" if not text_out or text_out.endswith(("¿", "¡", "(")) else " ") + word_fmt

        if text_out and text_out[-1] not in ".!?":
            text_out += "."
        return text_out.strip()

    def save(self) -> None:
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "context_len": self.context_len,
            "vocab": self.vocab,
            "unigrams": dict(self.unigrams.most_common(2500)),
            "ngrams": {
                str(order): {k: dict(v.most_common(12)) for k, v in table.items()}
                for order, table in self.ngrams.items()
            },
            "E": [[round(v, 5) for v in row] for row in self.E],
            "W1": [[round(v, 5) for v in row] for row in self.W1],
            "b1": [round(v, 5) for v in self.b1],
            "W2": [[round(v, 5) for v in row] for row in self.W2],
            "b2": [round(v, 5) for v in self.b2],
            "trained_steps": self.trained_steps,
            "last_loss": self.last_loss,
        }
        self.weights_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        try:
            raw = json.loads(self.weights_path.read_text(encoding="utf-8"))
            self.embed_dim = int(raw["embed_dim"])
            self.hidden_dim = int(raw["hidden_dim"])
            self.context_len = int(raw["context_len"])
            self.vocab = list(raw["vocab"])
            self.word_to_id = {w: i for i, w in enumerate(self.vocab)}
            self.unigrams = Counter(raw.get("unigrams", {}))
            for order in (2, 3, 4):
                self.ngrams[order] = defaultdict(Counter)
                for k, mp in raw.get("ngrams", {}).get(str(order), {}).items():
                    self.ngrams[order][k] = Counter(mp)
            self.E = raw.get("E", [])
            self.W1 = raw.get("W1", [])
            self.b1 = raw.get("b1", [])
            self.W2 = raw.get("W2", [])
            self.b2 = raw.get("b2", [])
            self.trained_steps = int(raw.get("trained_steps", 0))
            self.last_loss = float(raw.get("last_loss", 0.0))
            return bool(self.vocab and self.b2)
        except Exception:
            return False

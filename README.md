# ⚡ NEXUS — Inteligencia Artificial Propia 100% Local para Termux

**NEXUS** es una inteligencia artificial **propia, modular y 100% local** diseñada específicamente para ejecutarse desde la terminal de **Termux (Android ARM64)** y sistemas Linux, **sin depender de APIs de pago ni servicios en la nube**.

Funciona **al instante tras clonar el repositorio** (cero dependencias externas obligatorias, escrita sobre Python estándar optimizado) y además incluye un **gestor nativo de motores C++ locales (`llama.cpp` y `stable-diffusion.cpp`)** para escalar a modelos GGUF cuantizados según la memoria RAM de tu teléfono.

---

## 🚀 Capacidades Integradas (Los 5 Módulos)

| Módulo | Qué hace de forma 100% propia y local |
| :--- | :--- |
| **1. Texto** (`ia texto`) | **Cerebro Dual:** Combina una **Red Neuronal Multicapa Propia entrenable** (`PropiaNeuralLM` con embeddings + retropropagación SGD + N-grama de orden 4), **Memoria Vectorial SQLite (RAG BM25 + trigramas)**, evaluador matemático seguro AST, resumidor extractivo y soporte nativo para modelos **GGUF offline con `llama.cpp`**. |
| **2. Buscar** (`ia buscar`) | **Buscador Híbrido sin API keys:** Busca en tus **archivos y código local**, en la **memoria vectorial propia**, y cuando hay red consulta **Wikipedia + DuckDuckGo** guardando todo en **caché SQLite offline** para consultarlo después sin internet. Incluye lector limpio de páginas web (`/leer`). |
| **3. Imágenes** (`ia imagen`) | **Sintetizador Visual Propio + Red Neuronal CPPN + `stable-diffusion.cpp`:** Renderiza paisajes atmosféricos multicapa, ciudades cyberpunk, escenas espaciales 3D con planetas y anillos, fractales de Julia y arte neuronal **CPPN**, codificados con un **motor PNG propio de 24 bits**, filtros (`bordes`, `cyberpunk`, `sepia`, `gris`, `invertir`) y **vista previa a todo color (`▀` ANSI TrueColor) dentro de Termux**. |
| **4. Video** (`ia video`) | **Motor de Animación Multi-Escena + Audio WAV:** Acepta *storyboards* separados por `\|` con transiciones suaves, física temporal real (olas, nubes, luces, órbita planetaria, evolución neuronal), **banda sonora ambiental WAV sintetizada** y codificación adaptable en **MP4 (`ffmpeg`)** o **GIF89a LZW + RIFF AVI nativos** + reproductor en vivo en terminal. |
| **5. Agente** (`ia agente`) | **Agente Autónomo ReAct para Termux y Código:** Planifica pasos, crea y verifica scripts Python, edita código, ejecuta comandos de consola, investiga temas, genera multimedia y controla **Termux:API** (batería, notificaciones, voz TTS/STT, vibración, abrir en galería) protegido por un **Guardia de Seguridad de 3 niveles**. |

---

## 📲 Instalación Rápida en Termux (Android)

Abre **Termux** en tu teléfono y ejecuta:

```bash
pkg update -y && pkg install -y git python ffmpeg termux-api
git clone https://github.com/lyanvalentinmail-prog/Ai.git
cd Ai
bash install.sh
```

> 💡 **Opcional (acceso a la Galería de Android):** ejecuta `termux-setup-storage` una sola vez en Termux para que cada imagen o video que genere tu IA aparezca automáticamente también en `/sdcard/Pictures/IA_Propia` y `/sdcard/Movies/IA_Propia`.

Si deseas compilar de una vez el motor C++ `llama.cpp` para modelos `.gguf`:

```bash
bash install.sh --with-llama
```

---

## 💻 Uso desde la Consola de Termux

Tras instalarlo tendrás el comando global `ia` (o puedes ejecutarlo localmente con `./ia`):

### 1️⃣ Consola Interactiva Completa (REPL)
```bash
./ia
```
Cuenta con autocompletado con tecla `Tab`, historial persistente y detección inteligente en lenguaje natural (puedes escribir normalmente o usar `/ayuda`, `/imagen`, `/video`, `/buscar`, `/agente`, `/entrenar`, `/modelos`, `/auto`, `/voz`, `/doctor`).

---

### 2️⃣ Módulo de Texto y Entrenamiento Propio
```bash
# Conversación, preguntas y cálculo matemático exacto
./ia "Hola, ¿qué puedes hacer?"
./ia "¿Cuánto es sqrt(144) * 5 + 10?"
./ia "¿Qué es un agujero negro?"

# Enseñarle datos permanentes a tu IA
./ia "Me llamo Valentín"
./ia "Recuerda que mi proyecto secreto usa propulsión iónica"

# Entrenar los pesos de tu propia Red Neuronal + memoria RAG con tus textos
./ia entrenar mis_apuntes.txt --epocas 8

# Indexar toda una carpeta de código o notas para consultas offline
./ia indexar ~/proyectos/
```

---

### 3️⃣ Módulo Buscar (Offline + Web Abierta sin API Keys)
```bash
# Búsqueda combinada (Memoria propia + Archivos locales + Web con síntesis)
./ia buscar "arquitectura ARM64 NEON"

# Búsqueda 100% offline en tus archivos locales y memoria
./ia buscar "termux" --local

# Leer, limpiar anuncios, resumir y aprender una URL completa
./ia buscar --url "https://es.wikipedia.org/wiki/Inteligencia_artificial"
```

---

### 4️⃣ Módulo de Imágenes (100% Local + Vista ANSI en Terminal)
```bash
# Paisajes semánticos y ciudades cyberpunk
./ia imagen "Ciudad cyberpunk de noche con reflejo en el mar"

# Escenas espaciales 3D con planetas, nebulosas y anillos
./ia imagen "Planeta con anillos en una nebulosa espacial" --modo space

# Arte generado por Red Neuronal CPPN propia
./ia imagen "Ondas sinápticas de energía cuántica" --modo cppn

# Fractales matemáticos de alta definición
./ia imagen "Caleidoscopio cósmico" --modo fractal --ancho 640 --alto 480

# Aplicar filtros locales a una imagen existente
./ia imagen --filtro bordes --entrada ~/.ia_propia/outputs/imagenes/mi_imagen.png
```

---

### 5️⃣ Módulo de Video (Storyboard Animado + Audio Sintetizado)
```bash
# Video multi-escena con transiciones cross-fade y banda sonora WAV/MP4
./ia video "Amanecer en montañas | Ciudad cyberpunk de noche | Planeta con anillos" --duracion 5 --fps 12

# Generar y reproducir la animación en vivo a todo color dentro de Termux
./ia video "Viaje por una nebulosa espacial" --duracion 3 --fps 10 --reproducir
```

---

### 6️⃣ Módulo Agente Autónomo (Código + Terminal + Termux:API)
```bash
# Crear un programa, guardarlo y ejecutarlo automáticamente para verificar que funciona
./ia agente "Crea una calculadora científica en calc.py y pruébala"
./ia agente "Crea un script primos.py que calcule números primos hasta 100 y pruébalo"

# Analizar un directorio y guardar un informe en Markdown
./ia agente "Analiza los archivos de esta carpeta y guarda el resultado en reporte.md"

# Investigar un tema, guardar resumen y generar arte en una sola misión
./ia agente "Busca información sobre Termux, guarda notas.md y genera una imagen cyberpunk"

# Interactuar con Android mediante Termux:API
./ia agente "Consulta la batería y envía una notificación"
```

---

## 🧠 Activar Modelos GGUF Pesados Offline (`llama.cpp`)

Tu IA detecta automáticamente cuánta memoria RAM tiene tu teléfono y te recomienda el modelo cuantizado ideal (`SmolLM2-360M`, `Qwen2.5-0.5B`, `Qwen2.5-1.5B` o `Qwen2.5-3B`):

```bash
./ia modelos                         # Ver catálogo y recomendación según tu RAM
./ia modelos compilar-llama          # Instala/compila llama.cpp nativo en ARM64
./ia modelos descargar qwen2.5-1.5b  # Descarga con soporte de reanudación si se corta el WiFi
./ia modelos importar ~/storage/downloads/modelo.gguf  # O vincula cualquier .gguf propio
```

---

## 🧪 Estructura del Proyecto y Pruebas

```text
Ai/
├── ia                      # Ejecutable CLI principal para Termux y Linux
├── install.sh              # Instalador automático para Termux ($PREFIX/bin/ia)
├── data/
│   └── corpus_base_es.txt  # Conocimiento base semilla en español
├── ia_core/
│   ├── config.py           # Configuración, rutas y detección de hardware/Termux
│   ├── ui.py               # Interfaz ANSI TrueColor 24-bit y visor de imagen/video en consola
│   ├── memory.py           # Memoria vectorial SQLite (BM25 + trigramas coseno) y caché offline
│   ├── neural_lm.py        # Red Neuronal Multicapa Propia entrenable (SGD + N-grama orden 4)
│   ├── model_manager.py    # Gestor de llama.cpp, stable-diffusion.cpp y catálogo GGUF
│   ├── text_engine.py      # Cerebro Dual de Texto (GGUF + Motor Cognitivo/Neuronal Propio)
│   ├── search_engine.py    # Buscador local + Wikipedia/DuckDuckGo sin API keys
│   ├── image_engine.py     # Sintetizador visual semántico, 3D, Fractal, CPPN y PNG nativo
│   ├── video_engine.py     # Animador multi-escena, sintetizador WAV y encoders MP4/GIF89a/AVI
│   ├── termux_bridge.py    # Puente con Android Termux:API (TTS, STT, batería, notificaciones)
│   ├── agent_engine.py     # Agente autónomo ReAct con 14 herramientas y SafetyGuard
│   └── cli.py              # Enrutador de subcomandos y consola interactiva REPL
└── tests/
    └── test_all.py         # Suite de pruebas automatizadas de los 5 módulos
```

Para ejecutar todas las pruebas automatizadas:

```bash
python3 tests/test_all.py
```

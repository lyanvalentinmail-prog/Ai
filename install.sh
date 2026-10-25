#!/usr/bin/env bash
# ============================================================================
# Instalador Automático de NEXUS (IA Propia 100% Local) para Termux y Linux
# Uso:
#   bash install.sh                 # Instalación rápida (100% funcional al instante)
#   bash install.sh --with-llama    # También instala/compila llama.cpp para modelos GGUF
# ============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_LLAMA=0

for arg in "$@"; do
    case "$arg" in
        --with-llama) WITH_LLAMA=1 ;;
    esac
done

echo -e "\033[1;36m⚡ Instalando NEXUS — Inteligencia Artificial Propia 100% Local...\033[0m"

# 1. Detectar Termux vs Linux estándar
if [ -n "${TERMUX_VERSION:-}" ] || [[ "${PREFIX:-}" == *"com.termux"* ]] || [ -d "/data/data/com.termux/files/usr" ]; then
    echo -e "\033[1;32m✔ Entorno Android / Termux detectado.\033[0m"
    pkg update -y
    pkg install -y python ffmpeg git termux-api python-pillow || pkg install -y python git
    BIN_TARGET="${PREFIX}/bin/ia"
else
    echo -e "\033[1;34mℹ Entorno Linux estándar detectado.\033[0m"
    mkdir -p "${HOME}/.local/bin"
    BIN_TARGET="${HOME}/.local/bin/ia"
fi

# 2. Permisos de ejecución
chmod +x "${REPO_DIR}/ia"
chmod +x "${REPO_DIR}/install.sh"

# 3. Crear comando global `ia`
ln -sf "${REPO_DIR}/ia" "${BIN_TARGET}"
echo -e "\033[1;32m✔ Ejecutable global vinculado en: ${BIN_TARGET}\033[0m"

# 4. Inicializar base de memoria SQLite y pesos de la Red Neuronal Propia
python3 "${REPO_DIR}/ia" doctor

# 5. Opcional: instalar/compilar llama.cpp local
if [ "${WITH_LLAMA}" -eq 1 ]; then
    echo -e "\033[1;35m⚙ Configurando motor local C++ llama.cpp...\033[0m"
    if [ -n "${PREFIX:-}" ] && command -v pkg >/dev/null 2>&1; then
        pkg install -y clang cmake make
    fi
    python3 "${REPO_DIR}/ia" modelos compilar-llama
fi

echo ""
echo -e "\033[1;32m============================================================\033[0m"
echo -e "\033[1;32m¡Instalación completada! Ya puedes usar tu IA desde cualquier carpeta:\033[0m"
echo -e "  \033[1;36mia\033[0m                                  # Consola interactiva en Termux"
echo -e "  \033[1;36mia \"Hola, ¿cuánto es sqrt(144)*5?\"\033[0m  # Texto y cálculo directo"
echo -e "  \033[1;36mia buscar \"Termux Android\"\033[0m          # Buscador local + web sin API keys"
echo -e "  \033[1;36mia imagen \"Ciudad cyberpunk\"\033[0m        # Generar PNG + vista en terminal"
echo -e "  \033[1;36mia video \"Montañas | Galaxia\"\033[0m       # Generar video animado con audio"
echo -e "  \033[1;36mia agente \"Crea primos.py y pruébalo\"\033[0m # Agente autónomo"
echo -e "  \033[1;36mia modelos\033[0m                          # Descargar modelos GGUF offline"
if [ -n "${TERMUX_VERSION:-}" ] || [[ "${PREFIX:-}" == *"com.termux"* ]]; then
    echo -e "\n💡 Consejo Android: ejecuta \033[1;33mtermux-setup-storage\033[0m una vez para que tus imágenes"
    echo -e "   y videos aparezcan automáticamente en la Galería de tu teléfono."
fi
echo -e "\033[1;32m============================================================\033[0m"

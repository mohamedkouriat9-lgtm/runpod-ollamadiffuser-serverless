#!/bin/bash
set -e
set -x

echo "Configuration du stockage vers le Network Volume..."
mkdir -p "${MODELS_VOLUME_PATH}"
mkdir -p "${CACHE_VOLUME_PATH}"

ollamadiffuser config set models_dir "${MODELS_VOLUME_PATH}"
ollamadiffuser config set cache_dir "${CACHE_VOLUME_PATH}"

echo "=== Configuration effective (diagnostic) ==="
ollamadiffuser config
echo "============================================"

if [ -z "$DEFAULT_MODEL" ]; then
  echo "ERREUR : la variable d'environnement DEFAULT_MODEL doit être définie (ex: stable-diffusion-xl-base, z-image-turbo, flux.1-schnell, omnigen)."
  exit 1
fi

echo "Modèles disponibles sur le volume :"
ollamadiffuser list || true

set +x
echo "Chargement et démarrage du modèle '${DEFAULT_MODEL}' sur 0.0.0.0:8000..."
exec ollamadiffuser run "${DEFAULT_MODEL}" --host 0.0.0.0 --port 8000

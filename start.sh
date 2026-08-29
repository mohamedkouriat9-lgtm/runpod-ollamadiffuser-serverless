#!/bin/bash
set -e

echo "Configuration du stockage vers le Network Volume..."
ollamadiffuser config set models_dir "${MODELS_VOLUME_PATH}"
ollamadiffuser config set cache_dir "${CACHE_VOLUME_PATH}"

if [ -z "$DEFAULT_MODEL" ]; then
  echo "ERREUR : la variable d'environnement DEFAULT_MODEL doit être définie (ex: stable-diffusion-xl-base, z-image-turbo, flux.1-schnell, omnigen)."
  exit 1
fi

echo "Les modèles disponibles sur le volume :"
ollamadiffuser list || true

echo "Chargement et démarrage du modèle '${DEFAULT_MODEL}' sur 0.0.0.0:8000..."
exec ollamadiffuser run "${DEFAULT_MODEL}" --host 0.0.0.0 --port 8000


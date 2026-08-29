#!/bin/bash
set -e
set -x

echo "Configuration du stockage vers le Network Volume..."
mkdir -p "${MODELS_VOLUME_PATH}"
mkdir -p "${CACHE_VOLUME_PATH}"

ollamadiffuser config set models_dir "${MODELS_VOLUME_PATH}"
ollamadiffuser config set cache_dir "${CACHE_VOLUME_PATH}"

if [ -z "$DEFAULT_MODEL" ]; then
  echo "ERREUR : la variable d'environnement DEFAULT_MODEL doit être définie (ex: stable-diffusion-xl-base, z-image-turbo, flux.1-schnell, omnigen)."
  exit 1
fi

# OllamaDiffuser ne "redécouvre" pas automatiquement les modèles déjà
# présents sur disque après un simple changement de models_dir : il
# faut repasser par 'pull' pour qu'il vérifie l'intégrité des fichiers
# existants et les enregistre dans sa config interne. Comme les
# fichiers sont déjà complets sur le volume, cette étape est rapide
# (vérification d'intégrité seulement, pas de re-téléchargement).
echo "Enregistrement du modèle '${DEFAULT_MODEL}' auprès d'OllamaDiffuser..."
ollamadiffuser pull "${DEFAULT_MODEL}"

echo "=== Configuration effective (diagnostic) ==="
ollamadiffuser config
ollamadiffuser list
echo "============================================"

set +x
echo "Chargement et démarrage du modèle '${DEFAULT_MODEL}' sur 0.0.0.0:8000..."
exec ollamadiffuser run "${DEFAULT_MODEL}" --host 0.0.0.0 --port 8000

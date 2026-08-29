#!/bin/bash
set -e
set -x

echo "Configuration du stockage vers le Network Volume"
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
echo "Chargement et démarrage du modèle '${DEFAULT_MODEL}' en interne sur 127.0.0.1:8001..."
ollamadiffuser run "${DEFAULT_MODEL}" --host 127.0.0.1 --port 8001 &

echo "Attente que le modèle soit chargé et le serveur prêt..."
until curl -sf http://127.0.0.1:8001/api/health > /dev/null 2>&1; do
  sleep 2
done
echo "OllamaDiffuser est prêt en interne."

echo "Démarrage de nginx en façade sur 0.0.0.0:8000..."
exec nginx -g 'daemon off;'


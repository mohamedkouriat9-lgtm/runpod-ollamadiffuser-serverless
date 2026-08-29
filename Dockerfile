FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git dos2unix nginx && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ollamadiffuser && \
    pip uninstall -y hf-xet

WORKDIR /app
COPY start.sh /app/start.sh
RUN dos2unix /app/start.sh && chmod +x /app/start.sh

COPY nginx.conf /etc/nginx/sites-enabled/default

# Chemins vers le Network Volume RunPod (monté sur /runpod-volume
# en mode Serverless, confirmé dans la console). On y avait déjà
# stocké les modèles sous le dossier "ollamadiffuser-models" lors
# du pull effectué depuis le Pod classique.
ENV MODELS_VOLUME_PATH=/runpod-volume/ollamadiffuser-models
ENV CACHE_VOLUME_PATH=/runpod-volume/ollamadiffuser-cache

# Modèle chargé automatiquement au démarrage de CE conteneur.
# Une valeur différente doit être définie (via override d'endpoint
# RunPod) pour chacun des 4 déploiements :
#   stable-diffusion-xl-base | z-image-turbo | flux.1-schnell | omnigen
ENV HF_HUB_DISABLE_XET=1
ENV DEFAULT_MODEL=stable-diffusion-xl-base

EXPOSE 8000

ENTRYPOINT ["/app/start.sh"]

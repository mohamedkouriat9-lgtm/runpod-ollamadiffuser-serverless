"""
Patch pour ollamadiffuser/core/inference/strategies/generic_strategy.py

Problème : GenericPipelineStrategy charge un seul pipeline (ex: ZImagePipeline,
texte->image uniquement) et essaie de lui passer un paramètre 'image' lors
d'un appel /api/generate/img2img, ce que ce pipeline ne supporte pas
(TypeError: unexpected keyword argument 'image').

Solution DÉFENSIVE : on tente d'abord l'appel normal du pipeline, EXACTEMENT
comme avant ce patch. Ce n'est QUE si cet appel échoue avec un TypeError
mentionnant précisément 'image' ou 'mask_image' comme argument inattendu,
qu'on tente de convertir le pipeline déjà chargé vers son équivalent
img2img/inpaint via diffusers.AutoPipelineForImage2Image /
AutoPipelineForInpainting (.from_pipe()), qui réutilise les mêmes poids déjà
en mémoire sans recharger depuis le disque.

Conséquence importante : pour tout modèle dont le pipeline supporte déjà
nativement 'image' (ex: potentiellement OmniGen, pipeline unifié), le
comportement est STRICTEMENT IDENTIQUE à avant le patch - aucun risque de
régression, puisque le chemin de conversion n'est jamais emprunté dans ce cas.
"""
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else (
    "/opt/conda/lib/python3.11/site-packages/"
    "ollamadiffuser/core/inference/strategies/generic_strategy.py"
)

with open(TARGET, "r") as f:
    content = f.read()

if "AutoPipelineForImage2Image" in content:
    print("Patch déjà appliqué, rien à faire.")
    sys.exit(0)

# 1) Ajouter les imports nécessaires
content = content.replace(
    "from ..base import InferenceStrategy",
    "from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting\n"
    "from ..base import InferenceStrategy",
)

# 2) Initialiser les caches de pipelines convertis dans load()
content = content.replace(
    "            self.device = device\n"
    "            self.model_config = model_config\n",
    "            self.device = device\n"
    "            self.model_config = model_config\n"
    "            self._img2img_pipeline = None\n"
    "            self._inpaint_pipeline = None\n",
)

# 3) Remplacer le bloc d'appel du pipeline dans generate()
old_call_block = """        try:
            logger.info(
                f"Generating with {type(self.pipeline).__name__}: "
                f"steps={steps}, guidance={guidance}, seed={used_seed}"
            )
            output = self.pipeline(**gen_kwargs)
            image = output.images[0]
            return self._sanitize_image(image)
        except TypeError as e:
            # Some pipelines don't accept all standard params (e.g., width/height)
            # Retry without optional params
            logger.warning(f"Pipeline call failed: {e}. Retrying with minimal params.")
            for key in ("width", "height", "negative_prompt"):
                gen_kwargs.pop(key, None)
            output = self.pipeline(**gen_kwargs)
            image = output.images[0]
            return self._sanitize_image(image)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._create_error_image(str(e), prompt)"""

new_call_block = """        try:
            logger.info(
                f"Generating with {type(self.pipeline).__name__}: "
                f"steps={steps}, guidance={guidance}, seed={used_seed}"
            )
            output = self.pipeline(**gen_kwargs)
            image = output.images[0]
            return self._sanitize_image(image)
        except TypeError as e:
            err_msg = str(e)
            needs_img2img_fallback = (
                ("image" in gen_kwargs or "mask_image" in gen_kwargs)
                and "unexpected keyword argument" in err_msg
                and ("'image'" in err_msg or "'mask_image'" in err_msg)
            )
            if needs_img2img_fallback:
                active_pipeline = None
                try:
                    if "mask_image" in gen_kwargs:
                        if self._inpaint_pipeline is None:
                            self._inpaint_pipeline = AutoPipelineForInpainting.from_pipe(self.pipeline)
                            logger.info(
                                f"Basculé vers {type(self._inpaint_pipeline).__name__} pour l'inpainting"
                            )
                        active_pipeline = self._inpaint_pipeline
                    else:
                        if self._img2img_pipeline is None:
                            self._img2img_pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline)
                            logger.info(
                                f"Basculé vers {type(self._img2img_pipeline).__name__} pour l'img2img"
                            )
                        active_pipeline = self._img2img_pipeline
                except Exception as conv_err:
                    logger.warning(f"Impossible de convertir le pipeline: {conv_err}")

                if active_pipeline is not None:
                    output = active_pipeline(**gen_kwargs)
                    image = output.images[0]
                    return self._sanitize_image(image)

            # Some pipelines don't accept all standard params (e.g., width/height)
            # Retry without optional params
            logger.warning(f"Pipeline call failed: {e}. Retrying with minimal params.")
            for key in ("width", "height", "negative_prompt"):
                gen_kwargs.pop(key, None)
            output = self.pipeline(**gen_kwargs)
            image = output.images[0]
            return self._sanitize_image(image)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._create_error_image(str(e), prompt)"""

if old_call_block not in content:
    print("ERREUR : bloc cible introuvable, le patch ne peut pas être appliqué.")
    print("Le code source d'ollamadiffuser a peut-être changé de version.")
    sys.exit(1)

content = content.replace(old_call_block, new_call_block)

with open(TARGET, "w") as f:
    f.write(content)

print(f"Patch appliqué avec succès sur {TARGET}")

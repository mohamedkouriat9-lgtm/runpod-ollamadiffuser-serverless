"""
Patch pour ollamadiffuser/core/inference/strategies/flux_strategy.py

Même problème que pour ZImagePipeline : FluxPipeline ne supporte pas
nativement 'image'/'mask_image' (TypeError: unexpected keyword argument).
FluxStrategy est une stratégie dédiée séparée de GenericPipelineStrategy,
avec sa propre méthode generate() : elle a besoin de son propre patch.

Solution DÉFENSIVE identique en esprit à celle de generic_strategy.py :
on tente d'abord l'appel normal. Ce n'est QUE si l'exception est un
TypeError mentionnant précisément 'image'/'mask_image' comme argument
inattendu qu'on bascule vers AutoPipelineForImage2Image/Inpainting
.from_pipe(), qui réutilise les poids déjà chargés en mémoire.
"""
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else (
    "/opt/conda/lib/python3.11/site-packages/"
    "ollamadiffuser/core/inference/strategies/flux_strategy.py"
)

with open(TARGET, "r") as f:
    content = f.read()

if "AutoPipelineForImage2Image" in content:
    print("Patch déjà appliqué, rien à faire.")
    sys.exit(0)

# 1) Ajouter les imports nécessaires
content = content.replace(
    "from ..base import InferenceStrategy, SAFETY_DISABLED_KWARGS",
    "from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting\n"
    "from ..base import InferenceStrategy, SAFETY_DISABLED_KWARGS",
)

# 2) Initialiser les caches de pipelines convertis dans load()
old_init = (
    "            self.device = device\n"
    "            self.model_config = model_config\n"
)
new_init = (
    "            self.device = device\n"
    "            self.model_config = model_config\n"
    "            self._img2img_pipeline = None\n"
    "            self._inpaint_pipeline = None\n"
)
if old_init not in content:
    print("ERREUR : point d'insertion pour l'init des caches introuvable.")
    sys.exit(1)
content = content.replace(old_init, new_init)

# 2bis) Forcer le VAE tiling/slicing directement sur pipeline.vae, en
# prévention d'un CUDA OOM lors de l'encodage d'une image source en img2img
# (même logique que pour generic_strategy.py / ZImagePipeline).
old_mem_opt_call = "            self._apply_memory_optimizations()\n"
new_mem_opt_call = (
    "            self._apply_memory_optimizations()\n"
    "            try:\n"
    "                vae = getattr(self.pipeline, \"vae\", None)\n"
    "                if vae is not None:\n"
    "                    if hasattr(vae, \"enable_tiling\"):\n"
    "                        vae.enable_tiling()\n"
    "                        logger.info(\"Enabled VAE tiling (direct on pipeline.vae)\")\n"
    "                    if hasattr(vae, \"enable_slicing\"):\n"
    "                        vae.enable_slicing()\n"
    "                        logger.info(\"Enabled VAE slicing (direct on pipeline.vae)\")\n"
    "            except Exception as vae_opt_err:\n"
    "                logger.warning(f\"Impossible d'activer le VAE tiling/slicing direct: {vae_opt_err}\")\n"
)
if old_mem_opt_call not in content:
    print("ERREUR : point d'insertion pour le VAE tiling introuvable.")
    sys.exit(1)
content = content.replace(old_mem_opt_call, new_mem_opt_call)

# 3) Remplacer le bloc try/except de generate()
old_block = """        try:
            logger.info(f"Generating FLUX image: steps={steps}, guidance={guidance}, seed={used_seed}")
            output = self.pipeline(**gen_kwargs)
            return output.images[0]
        except RuntimeError as e:
            if "CUDA" in str(e) and self.device == "cpu":
                logger.warning("Device mismatch, retrying without generator")
                gen_kwargs.pop("generator", None)
                output = self.pipeline(**gen_kwargs)
                return output.images[0]
            raise
        except Exception as e:
            logger.error(f"FLUX generation failed: {e}")
            return self._create_error_image(str(e), prompt)"""

new_block = """        try:
            logger.info(f"Generating FLUX image: steps={steps}, guidance={guidance}, seed={used_seed}")
            output = self.pipeline(**gen_kwargs)
            return output.images[0]
        except RuntimeError as e:
            if "CUDA" in str(e) and self.device == "cpu":
                logger.warning("Device mismatch, retrying without generator")
                gen_kwargs.pop("generator", None)
                output = self.pipeline(**gen_kwargs)
                return output.images[0]
            raise
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
                    try:
                        output = active_pipeline(**gen_kwargs)
                        return output.images[0]
                    except Exception as gen_err:
                        logger.error(f"FLUX img2img generation failed: {gen_err}")
                        return self._create_error_image(str(gen_err), prompt)

            logger.error(f"FLUX generation failed: {e}")
            return self._create_error_image(str(e), prompt)
        except Exception as e:
            logger.error(f"FLUX generation failed: {e}")
            return self._create_error_image(str(e), prompt)"""

if old_block not in content:
    print("ERREUR : bloc cible introuvable dans flux_strategy.py.")
    print("Le code source d'ollamadiffuser a peut-être changé de version.")
    sys.exit(1)

content = content.replace(old_block, new_block)

with open(TARGET, "w") as f:
    f.write(content)

print(f"Patch appliqué avec succès sur {TARGET}")

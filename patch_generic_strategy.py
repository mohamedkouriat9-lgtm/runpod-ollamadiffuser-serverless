"""
Patch pour ollamadiffuser/core/inference/strategies/generic_strategy.py

Problème 1 : GenericPipelineStrategy charge un seul pipeline (ex:
ZImagePipeline, texte->image uniquement) et essaie de lui passer un
paramètre 'image' lors d'un appel /api/generate/img2img, ce que ce
pipeline ne supporte pas (TypeError: unexpected keyword argument 'image').
Solution : conversion à la volée vers l'équivalent img2img/inpaint via
diffusers.AutoPipelineForImage2Image/Inpainting (.from_pipe()).

Problème 2 : certains pipelines (ex: OmniGenPipeline) n'ont pas
d'équivalent Img2Img mappé dans diffusers, et utilisent un nom de
paramètre différent pour l'image d'entrée (ex: 'input_images', une
liste, au lieu de 'image'). Solution : si la conversion de pipeline
échoue/n'existe pas, on inspecte la signature du pipeline pour un nom
de paramètre alternatif connu, et on remappe 'image' vers ce paramètre
avant de retenter l'appel sur le pipeline original (pas de conversion).

Le tout reste DÉFENSIF : le chemin normal (appel direct du pipeline)
est toujours tenté en premier, inchangé. Ces mécanismes de secours ne
s'activent que sur l'erreur précise "unexpected keyword argument
'image'/'mask_image'", donc aucun risque de régression pour un modèle
dont le pipeline supporte déjà nativement ces paramètres.
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
    "import inspect\n"
    "from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting\n"
    "from ..base import InferenceStrategy",
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

# 2bis) Forcer le VAE tiling/slicing directement sur pipeline.vae si le
# wrapper de haut niveau pipeline.enable_vae_tiling() est absent (c'est le
# cas de ZImagePipeline, qui ne l'expose pas malgré un VAE qui le supporte).
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
                    try:
                        output = active_pipeline(**gen_kwargs)
                        image = output.images[0]
                        return self._sanitize_image(image)
                    except Exception as conv_call_err:
                        logger.warning(f"Appel du pipeline converti échoué: {conv_call_err}")

                # Pas de pipeline converti disponible (ex: OmniGen, pas de
                # variante Img2Img dans diffusers) : on tente un remapping
                # de paramètre vers un nom alternatif connu (ex: 'image' ->
                # 'input_images') si le pipeline original l'accepte.
                ALT_IMAGE_PARAM_CANDIDATES = ["input_images", "init_image", "input_image"]
                try:
                    sig_params = inspect.signature(self.pipeline.__call__).parameters
                except (TypeError, ValueError):
                    sig_params = {}
                remapped = False
                if "image" in gen_kwargs:
                    for alt_name in ALT_IMAGE_PARAM_CANDIDATES:
                        if alt_name in sig_params:
                            img_value = gen_kwargs.pop("image")
                            # Les pipelines à paramètre pluriel attendent une liste
                            gen_kwargs[alt_name] = [img_value] if alt_name.endswith("s") else img_value
                            logger.info(f"Paramètre 'image' remappé vers '{alt_name}' pour {type(self.pipeline).__name__}")
                            remapped = True
                            break
                if remapped:
                    try:
                        output = self.pipeline(**gen_kwargs)
                        image = output.images[0]
                        return self._sanitize_image(image)
                    except Exception as remap_err:
                        logger.error(f"Appel avec paramètre remappé échoué: {remap_err}")
                        return self._create_error_image(str(remap_err), prompt)

            # Some pipelines don't accept all standard params (e.g., width/height)
            # Retry without optional params
            logger.warning(f"Pipeline call failed: {e}. Retrying with minimal params.")
            for key in ("width", "height", "negative_prompt"):
                gen_kwargs.pop(key, None)
            try:
                output = self.pipeline(**gen_kwargs)
                image = output.images[0]
                return self._sanitize_image(image)
            except Exception as retry_err:
                logger.error(f"Generation failed on retry: {retry_err}")
                return self._create_error_image(str(retry_err), prompt)
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

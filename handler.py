import base64
import torch
from PIL import Image
from io import BytesIO
import runpod
import uuid
timport tempfile
from pathlib import Path
import pytorch_lightning as pl
from diffusers import StableDiffusionPipeline

import sys
import os

# Add the MaterialPalette directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'MaterialPalette'))

from concept import infer as concept_infer
from capture import get_data as capture_get_data
from capture import get_inference_module as capture_get_inference_module

# Global dict to hold models
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    """Load decomposition model & SD pipeline once."""
    if MODELS:
        return

    # Load PBR decomposer
    decomposer_ckpt = '/workspace/model.ckpt'
    if not os.path.exists(decomposer_ckpt):
        raise FileNotFoundError(f"Decomposer checkpoint not found at {decomposer_ckpt}")
    MODELS["decomposer"] = capture_get_inference_module(pt=decomposer_ckpt)

    # Load Stable Diffusion base pipeline
    base_model_id = "runwayml/stable-diffusion-v1-5"
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    MODELS["sd_pipeline"] = sd_pipe.to(DEVICE)

    print("✓ Models initialized.")


def handler(job):
    job_id  = job.get("id", uuid.uuid4())
    payload = job.get("input", {})

    # Lazy init
    if not MODELS:
        init()

    image_b64 = payload.get("image")
    if not image_b64:
        return {"error": "No 'image' field supplied."}

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        # Prepare input folders
        region_dir = tmp_dir / "region_0"
        mask_dir   = region_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        # Decode and save input image
        img_data = base64.b64decode(image_b64)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        img.save(region_dir / "image.png")
        # Full-white mask
        Image.new("L", img.size, 255).save(mask_dir / "000.png")

        # Create an empty LoRA directory (no .safetensors) to force base weights
        empty_lora = tmp_dir / "empty_lora"
        empty_lora.mkdir()

        try:
            # STEP 1: Use base weights only (no LoRA)
            print(f"[{job_id}] Using base weights (no LoRA)")

            # STEP 2: Generate texture views
            concept_infer(
                empty_lora,
                renorm=True,
                pipeline=MODELS["sd_pipeline"]
            )
            print(f"[{job_id}] Texture views generated.")

            # STEP 3: Decompose into PBR maps
            data_mod = capture_get_data(predict_dir=empty_lora, predict_ds='renorm')
            trainer  = pl.Trainer(
                accelerator='gpu',
                devices=1,
                precision=16,
                logger=False
            )
            trainer.predict(MODELS["decomposer"], dataloaders=data_mod.predict_dataloader())

            # Collect outputs
            out_dir = empty_lora / "renorm" / "predictions"
            maps = {
                "albedo":    list(out_dir.glob("*_albedo.png"))[0],
                "normal":    list(out_dir.glob("*_normal.png"))[0],
                "roughness": list(out_dir.glob("*_roughness.png"))[0],
                "metallic":  list(out_dir.glob("*_metallic.png"))[0],
            }

            # Encode results to base64
            def to_b64(path: Path):
                return base64.b64encode(path.read_bytes()).decode('ascii')

            return {
                "albedo_b64":    to_b64(maps["albedo"]),
                "normals_b64":   to_b64(maps["normal"]),
                "roughness_b64": to_b64(maps["roughness"]),
                "metallic_b64":  to_b64(maps["metallic"]),
            }

        except Exception as e:
            import traceback
            return {
                "error":     str(e),
                "traceback": traceback.format_exc().splitlines()[-10:]
            }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

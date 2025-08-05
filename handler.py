# handler.py (Definitive Final Version)
import base64
import torch
from PIL import Image
import numpy as np
from io import BytesIO
import runpod
import uuid
import tempfile
from pathlib import Path
import pytorch_lightning as pl
from diffusers import StableDiffusionPipeline

import sys
import os

# Add the MaterialPalette directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'MaterialPalette'))

import concept.config as concept_config
from concept import invert as concept_invert
from concept import infer as concept_infer
from capture import get_data as capture_get_data
from capture import get_inference_module as capture_get_inference_module

# Global dictionary to hold all our models
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    """Initialize all models and load them into memory."""
    if not MODELS:
        print("Initializing models...")
        matpal_checkpoint = '/workspace/model.ckpt'
        if not os.path.exists(matpal_checkpoint):
            raise FileNotFoundError(f"Decomposition model checkpoint not found at {matpal_checkpoint}")
        MODELS["decomposer"] = capture_get_inference_module(pt=matpal_checkpoint)

        base_model_id = "runwayml/stable-diffusion-v1-5"
        sd_pipeline = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, safety_checker=None)
        MODELS["sd_pipeline"] = sd_pipeline.to(DEVICE)

        print("All models loaded.")
    print("Model initialization successful.")

def handler(job):
    """The main handler function for the RunPod endpoint."""
    job_id = job.get("id", uuid.uuid4())
    job_input = job.get("input", {})

    if not MODELS:
        init()

    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided in the input."}

    with tempfile.TemporaryDirectory() as temp_dir_str:
        base_dir = Path(temp_dir_str)
        region_dir = base_dir / "region_0"
        mask_dir = region_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image.save(region_dir / "image.png")

        mask = Image.new("L", image.size, 255)
        mask.save(mask_dir / "000.png")

        try:
            # STEP 1: INVERT CONCEPT (Train the LoRA)
            print(f"[{job_id}] Step 1: Inverting concept to create LoRA...")
            lora_path = concept_invert(region_dir, pipeline=MODELS["sd_pipeline"])
            print(f"[{job_id}] LoRA created at: {lora_path}")
            torch.cuda.empty_cache()

            # STEP 2: INFER (Generate Texture Views)
            print(f"[{job_id}] Step 2: Generating texture views using LoRA...")
            concept_infer(lora_path, renorm=True)
            print(f"[{job_id}] Texture views generated.")

            # STEP 3: DECOMPOSE (Create PBR Maps)
            print(f"[{job_id}] Step 3: Decomposing views into PBR maps...")
            # --- FIX 1: Get the DataModule wrapper ---
            data_module = capture_get_data(predict_dir=lora_path, predict_ds='renorm')

            trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16, logger=False)

            # --- FIX 2: Give the Trainer the specific dataloader it needs ---
            trainer.predict(MODELS["decomposer"], dataloaders=data_module.predict_dataloader())

            output_path = lora_path / "renorm" / "predictions"
            pbr_maps = {
                "albedo": list(output_path.glob("*_albedo.png"))[0],
                "normal": list(output_path.glob("*_normal.png"))[0],
                "roughness": list(output_path.glob("*_roughness.png"))[0],
                "metallic": list(output_path.glob("*_metallic.png"))[0]
            }
            print(f"[{job_id}] PBR maps generated at: {output_path}")

            def to_b64(img_path):
                with open(img_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

            return {
                "albedo_b64": to_b64(pbr_maps["albedo"]),
                "normals_b64": to_b64(pbr_maps["normal"]),
                "roughness_b64": to_b64(pbr_maps["roughness"]),
                "metallic_b64": to_b64(pbr_maps["metallic"]),
            }

        except Exception as e:
            import traceback
            return {"error": f"An error occurred: {e}", "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
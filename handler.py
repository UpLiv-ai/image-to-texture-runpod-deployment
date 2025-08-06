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
import traceback

# Add the MaterialPalette directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'MaterialPalette'))

# We only need `invert` from concept.py and the capture functions
from concept import invert as concept_invert
from capture import get_data as capture_get_data
from capture import get_inference_module as capture_get_inference_module

# Global dict to hold models
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base model identifier
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"


def init():
    """Load all models once when the worker starts."""
    global MODELS
    if MODELS:
        return

    # Load PBR decomposer
    decomposer_ckpt = '/workspace/model.ckpt'
    if not os.path.exists(decomposer_ckpt):
        raise FileNotFoundError(f"Decomposer checkpoint not found at {decomposer_ckpt}")
    
    decomposer = capture_get_inference_module(pt=decomposer_ckpt)

    # Load base Stable Diffusion pipeline
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True
    )
    
    MODELS["decomposer"] = decomposer
    MODELS["sd_pipeline"] = sd_pipe.to(DEVICE)

    print("✓ Models initialized.")


def handler(job):
    """
    Handles a single job request to generate PBR maps from an input image.
    """
    job_id = job.get("id", str(uuid.uuid4()))
    payload = job.get("input", {})

    if not MODELS:
        init()

    sd_pipe = MODELS["sd_pipeline"]
    lora_path_for_cleanup = None

    try:
        # --- Get Inputs ---
        image_b64 = payload.get("image")
        prompt_template = payload.get("prompt", "a high resolution texture of <token>, 4k, pbr, photorealistic")
        train_steps = payload.get("train_steps", 20)
        infer_steps = payload.get("infer_steps", 50)

        if not image_b64:
            return {"error": "No 'image' field supplied in the input."}

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            region_dir = tmp_dir / "region_0"
            mask_dir = region_dir / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)

            # Decode and save input image and a full mask
            img_data = base64.b64decode(image_b64)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img.save(region_dir / "image.png")
            Image.new("L", img.size, 255).save(mask_dir / "000.png")

            # --- STEP 1: Train LoRA on the input image ---
            print(f"[{job_id}] Training LoRA for {train_steps} steps...")
            lora_path = concept_invert(region_dir, max_train_steps=train_steps)
            lora_path_for_cleanup = lora_path # Save for the `finally` block
            print(f"[{job_id}] Trained LoRA at: {lora_path}")
            torch.cuda.empty_cache()

            # --- STEP 2: Generate texture views using the new LoRA ---
            print(f"[{job_id}] Loading LoRA weights into pipeline...")
            sd_pipe.load_lora_weights(lora_path)

            placeholder_token = lora_path.parent.name.split('_')[-2]
            final_prompt = prompt_template.replace("<token>", placeholder_token)
            
            print(f"[{job_id}] Generating texture views with prompt: '{final_prompt}'...")
            generated_images = sd_pipe(
                [final_prompt] * 4,
                num_inference_steps=infer_steps,
                guidance_scale=7.5
            ).images

            generated_img_dir = lora_path / "renorm"
            generated_img_dir.mkdir(exist_ok=True)

            print(f"[{job_id}] Saving generated images...")
            for i, gen_img in enumerate(generated_images):
                gen_img.save(generated_img_dir / f"{i:03d}_renorm.png")

            print(f"[{job_id}] Texture views generated.")

            # --- STEP 3: Decompose views into PBR maps ---
            print(f"[{job_id}] Decomposing views into PBR maps...")
            data_mod = capture_get_data(predict_dir=lora_path, predict_ds='renorm')
            trainer = pl.Trainer(
                accelerator='gpu', devices=1, precision=16, logger=False
            )
            trainer.predict(MODELS["decomposer"], dataloaders=data_mod.predict_dataloader())

            predictions_dir = generated_img_dir / "predictions"
            map_names = ["albedo", "normal", "roughness", "metallic"]
            maps = {}
            for name in map_names:
                found = list(predictions_dir.glob(f"*_{name}.png"))
                if not found:
                    raise FileNotFoundError(f"Could not find predicted map for '{name}'")
                maps[name] = found[0]

            def to_b64(path: Path):
                return base64.b64encode(path.read_bytes()).decode('ascii')

            return {f"{k}_b64": to_b64(v) for k, v in maps.items()}

    except Exception as e:
        full_traceback = traceback.format_exc()
        print(f"An error occurred: {e}\n{full_traceback}")
        return {"error": str(e), "traceback": full_traceback.splitlines()}
    
    finally:
        # --- CLEANUP ---
        # Ensure LoRA is detached from the pipeline to be ready for the next job
        if lora_path_for_cleanup and hasattr(sd_pipe, 'unload_lora_weights'):
            print(f"[{job_id}] Unloading LoRA weights...")
            try:
                sd_pipe.unload_lora_weights()
            except Exception as e:
                print(f"Warning: could not unload LoRA weights: {e}")
        torch.cuda.empty_cache()
        print(f"[{job_id}] Handler finished.")


if __name__ == "__main__":
    # This allows you to test the handler without the `test_runner.py` script.
    # Usage: python handler.py path/to/your/image.png
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')
        
        job_input = {'input': {'image': b64_img}}
        result = handler(job_input)
        
        if 'error' in result:
            print("\n--- ERROR ---")
            print(result['error'])
        else:
            print("\n--- SUCCESS ---")
            output_dir = Path(f"output_{Path(image_path).stem}")
            output_dir.mkdir(exist_ok=True)
            for key, b64_data in result.items():
                map_name = key.replace('_b64', '')
                img_data = base64.b64decode(b64_data)
                save_path = output_dir / f"{map_name}.png"
                with open(save_path, "wb") as f:
                    f.write(img_data)
                print(f"Saved {save_path}")
    else:
        # Start the RunPod serverless worker
        runpod.serverless.start({"handler": handler})
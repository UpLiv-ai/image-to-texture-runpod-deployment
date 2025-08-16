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

# Import the necessary functions from the MaterialPalette library
import concept
import capture

# --- Configuration for Local Models ---
# Determine the base path for models depending on the RunPod environment
if os.path.exists('/runpod-volume'):
    base_volume_path = Path('/runpod-volume')
else:
    base_volume_path = Path('/workspace')

# Define the full path to the local Stable Diffusion model
BASE_MODEL_PATH = base_volume_path / 'models' / 'sd-v1-5'

# Global dict to hold models and device information
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init():
    """
    Initializes and loads the PBR decomposer model into memory.
    This function is called by RunPod once when the worker starts.
    """
    global MODELS
    # Prevent re-initialization
    if MODELS:
        return

    # Load the PBR decomposer model
    decomposer_ckpt = base_volume_path / 'model.ckpt'
    if not os.path.exists(decomposer_ckpt):
        raise FileNotFoundError(f"Decomposer checkpoint not found at {decomposer_ckpt}")
    
    decomposer = capture.get_inference_module(pt=str(decomposer_ckpt))
    
    # Store the decomposer model in the global dictionary
    MODELS["decomposer"] = decomposer

    print("✓ Decomposer model initialized successfully.")


def handler(job):
    """
    Handles a single job request to generate PBR maps from an input image.
    This function implements the full inference pipeline and ensures proper cleanup.
    """
    job_id = job.get("id", str(uuid.uuid4()))
    payload = job.get("input", {})

    # Initialize models if they haven't been loaded yet
    if not MODELS:
        init()

    try:
        # --- Get Inputs & Set Configuration ---
        image_b64 = payload.get("image")
        if not image_b64:
            return {"error": "No 'image' field was supplied in the input."}

        # Set processing parameters with defaults, allowing them to be overridden by the job input
        train_steps = payload.get("train_steps", 800)
        infer_steps = payload.get("infer_steps", 150)
        resolution = payload.get("resolution", 1024)
        prompt_key = payload.get("prompt_key", 'p3') # 'p3' is a predefined high-quality prompt

        # --- Use a Temporary Directory for All Artifacts ---
        # This ensures all data from a single run is isolated and automatically deleted.
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            
            # Save the decoded input image
            input_image_path = tmp_dir / "input_image.png"
            img_data = base64.b64decode(image_b64)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img.save(input_image_path)
            
            # --- Stage 1: Crop Input Image ---
            print(f"[{job_id}] Cropping input image...")
            regions_dir = concept.crop(input_image_path)
            
            # --- Stage 2: Train LoRA and Infer Top-View Image ---
            print(f"[{job_id}] Training LoRA for {train_steps} steps...")
            # We train on the first (and likely only) cropped region
            region_0_dir = next(regions_dir.iterdir())
            lora_path = concept.invert(region_0_dir, max_train_steps=train_steps)
            print(f"[{job_id}] Trained LoRA at: {lora_path}")

            print(f"[{job_id}] Inferring seamless top-view texture...")
            # Use the local BASE_MODEL_PATH for inference
            concept.infer(
                path=lora_path,
                renorm=True,
                base_model=str(BASE_MODEL_PATH),
                num_inference_steps=infer_steps,
                resolution=resolution,
                prompt=prompt_key,
                stitch_mode='wmean'
            )
            top_view_path = list((lora_path / 'out_renorm').glob('*_1K_*.png'))[0]
            print(f"[{job_id}] Generated top-view image: {top_view_path.name}")
            
            # --- Stage 3: Tile the Top-View into Overlapping Patches ---
            print(f"[{job_id}] Tiling generated texture into overlapping patches...")
            img_size, tile_size, overlap = resolution, 512, 64
            stride = tile_size - overlap
            full_image = np.array(Image.open(top_view_path).convert('RGB'), np.float32) / 255.0
            patches, coords = [], []

            for y0 in range(0, img_size, stride):
                for x0 in range(0, img_size, stride):
                    y1, x1 = min(y0 + tile_size, img_size), min(x0 + tile_size, img_size)
                    y0_, x0_ = y1 - tile_size, x1 - tile_size
                    coords.append((y0_, y1, x0_, x1))
                    patches.append(full_image[y0_:y1, x0_:x1, :])

            # Save patches to a dedicated directory for the decomposer
            patches_dir = tmp_dir / "patches_for_decomposition"
            patches_dir.mkdir()
            for i, patch_np in enumerate(patches):
                patch_img = Image.fromarray((patch_np * 255).astype(np.uint8))
                patch_img.save(patches_dir / f"patch_{i:03d}.png")
            
            # --- Stage 4: Decompose Patches into PBR Maps ---
            print(f"[{job_id}] Decomposing {len(patches)} patches into PBR maps...")
            data_mod = capture.get_data(predict_dir=patches_dir, predict_ds=None)
            trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16, logger=False)
            trainer.predict(MODELS["decomposer"], dataloaders=data_mod.predict_dataloader())
            
            # --- Stage 5 & 6: Gather and Blend PBR Patches ---
            print(f"[{job_id}] Blending PBR patches into full maps...")
            predictions_dir = patches_dir / "predictions"
            albedo_p, rough_p, norm_p = [], [], []
            
            for i in range(len(patches)):
                base_name = f"patch_{i:03d}"
                albedo_p.append(np.array(Image.open(predictions_dir / f"{base_name}_albedo.png"), np.float32) / 255.0)
                rough_p.append(np.array(Image.open(predictions_dir / f"{base_name}_roughness.png"), np.float32) / 255.0)
                # Note: The decomposer outputs 'normal.png' not 'normals.png'
                norm_p.append(np.array(Image.open(predictions_dir / f"{base_name}_normal.png"), np.float32) / 255.0)

            w = np.minimum(np.linspace(0, 1, tile_size), np.linspace(0, 1, tile_size)[::-1])
            window = np.outer(w, w)[..., np.newaxis] # Add channel dimension for broadcasting

            def blend(patch_list):
                canvas = np.zeros((img_size, img_size, 3), np.float32)
                weight = np.zeros((img_size, img_size, 1), np.float32)
                for patch, (y0, y1, x0, x1) in zip(patch_list, coords):
                    canvas[y0:y1, x0:x1] += patch * window
                    weight[y0:y1, x0:x1] += window
                return canvas / (weight + 1e-6)

            final_albedo = blend(albedo_p)
            final_roughness = blend(rough_p)
            final_normal = blend(norm_p)

            # --- Stage 7: Prepare and Return Output ---
            print(f"[{job_id}] Encoding final maps for output...")
            
            def numpy_to_b64(array):
                """Converts a NumPy array to a base64 encoded PNG string."""
                img = Image.fromarray((array.clip(0, 1) * 255).astype(np.uint8))
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode('ascii')

            return {
                "albedo_b64": numpy_to_b64(final_albedo),
                "roughness_b64": numpy_to_b64(final_roughness),
                "normal_b64": numpy_to_b64(final_normal),
                "top_view_b64": base64.b64encode(top_view_path.read_bytes()).decode('ascii')
            }

    except Exception as e:
        full_traceback = traceback.format_exc()
        print(f"An error occurred: {e}\n{full_traceback}")
        return {"error": str(e), "traceback": full_traceback.splitlines()}
    
    finally:
        # --- Final Cleanup ---
        # The TemporaryDirectory context manager handles file deletion.
        # This block ensures GPU memory is cleared for the next job.
        torch.cuda.empty_cache()
        print(f"[{job_id}] Handler finished.")


if __name__ == "__main__":
    # This block allows for local testing of the handler without a RunPod server.
    # Usage: python handler.py path/to/your/image.png
    if len(sys.argv) > 1:
        image_path_str = sys.argv[1]
        image_path = Path(image_path_str)
        
        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')
        
        job_input = {'input': {'image': b64_img}}
        result = handler(job_input)
        
        if 'error' in result:
            print("\n--- ERROR ---")
            print(result['error'])
            if 'traceback' in result:
                print("\n".join(result['traceback']))
        else:
            print("\n--- SUCCESS ---")
            output_dir = Path(f"output_{image_path.stem}")
            output_dir.mkdir(exist_ok=True)
            print(f"Saving results to {output_dir}/")
            
            for key, b64_data in result.items():
                map_name = key.replace('_b64', '')
                img_data = base64.b64decode(b64_data)
                save_path = output_dir / f"{map_name}.png"
                with open(save_path, "wb") as f:
                    f.write(img_data)
                print(f" ✓ Saved {save_path}")
    else:
        # This starts the RunPod serverless worker
        runpod.serverless.start({"handler": handler})

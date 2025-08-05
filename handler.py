#
# handler.py (Simplified for Material Palette only)
#

import os
import sys
import base64
import uuid
import shutil
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import runpod

# Add the MaterialPalette directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'MaterialPalette'))

# --- Import model-specific modules ---
from concept import crop as concept_crop, invert as concept_invert, infer as concept_infer
from capture import get_data as capture_get_data, get_inference_module as capture_get_inference_module
from pytorch_lightning import Trainer

# --- Global State for Models ---
MODELS = {
    "matpal_decomposer": None,
    "pl_trainer": None
}

def init():
    """
    Initializes the Material Palette model and stores it in the global MODELS dictionary.
    This function is called only once on worker startup.
    """
    global MODELS
    
    print("Initializing Material Palette model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Material Palette Decomposition Model
    matpal_checkpoint = "/workspace/model.ckpt"

    if not os.path.exists(matpal_checkpoint):
        raise FileNotFoundError(f"Material Palette checkpoint not found at {matpal_checkpoint}. Ensure it's on the network volume.")

    MODELS["matpal_decomposer"] = capture_get_inference_module(pt=matpal_checkpoint).to(device)
    print("Material Palette model loaded.")

    # 2. Initialize PyTorch Lightning Trainer for decomposition
    MODELS["pl_trainer"] = Trainer(accelerator='gpu', devices=1, precision=16)
    print("PyTorch Lightning Trainer initialized.")
    
    print("Model initialization successful.")


def image_to_base64(image):
    """Converts a PIL Image to a base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handler(job):
    """
    The main handler function for the serverless endpoint.
    Orchestrates the Material Palette pipeline.
    """
    # Ensure model is loaded
    if MODELS["matpal_decomposer"] is None:
        init()

    job_input = job['input']
    
    # --- Input Validation ---
    if 'image' not in job_input:
        return {"error": "Missing 'image' (base64 encoded) in input."}

    # --- Create a temporary directory for this job ---
    job_id = str(uuid.uuid4())
    temp_dir = os.path.join("/tmp", job_id)
    masks_dir = os.path.join(temp_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    # Use a generic name for the image and mask
    image_path = os.path.join(temp_dir, "input_image.png")
    mask_path = os.path.join(masks_dir, "full_mask.png")

    try:
        # --- Step 1: Decode image and create a full-image mask ---
        image_data = base64.b64decode(job_input['image'])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image.save(image_path)
        
        # Create a white mask of the same size as the input image.
        # This tells Material Palette to use the entire image.
        width, height = image.size
        mask_image = Image.new('L', (width, height), 255) # 'L' for grayscale, 255 for white
        mask_image.save(mask_path)
        print(f"[{job_id}] Generated full-image mask and saved to {mask_path}")

        # --- Step 2: Run Material Palette Pipeline Programmatically ---
        print(f"[{job_id}] Starting Material Palette pipeline...")

        # 2a. Crop the region based on the mask (concept.crop)
        regions = concept_crop(temp_dir)
        print(f"[{job_id}] Cropped regions extracted.")

        # 2b. Invert and Infer for each region (concept.invert, concept.infer)
        for region in regions.iterdir():
            print(f"[{job_id}] Inverting concept for region: {region.name}")
            lora = concept_invert(region)
            print(f"[{job_id}] Generating texture for region: {region.name}")
            concept_infer(lora, renorm=True)

        # 2c. Decompose the generated textures (capture module)
        print(f"[{job_id}] Preparing data for decomposition...")
        data_loader = capture_get_data(predict_dir=temp_dir, predict_ds='sd')
        
        print(f"[{job_id}] Running decomposition model...")
        MODELS["pl_trainer"].predict(MODELS["matpal_decomposer"], data_loader)
        print(f"[{job_id}] Decomposition complete.")

        # --- Step 3: Collect and encode output images ---
        output_dir = os.path.join(temp_dir, "lightning_logs/version_0/predict/sd/0/")
        
        albedo_path = os.path.join(output_dir, "albedo.png")
        normals_path = os.path.join(output_dir, "normals.png")
        roughness_path = os.path.join(output_dir, "roughness.png")

        if not all(os.path.exists(p) for p in [albedo_path, normals_path, roughness_path]):
            return {"error": "Decomposition did not produce the expected output files."}

        albedo_img = Image.open(albedo_path)
        normals_img = Image.open(normals_path)
        roughness_img = Image.open(roughness_path)

        return {
            "albedo_b64": image_to_base64(albedo_img),
            "normals_b64": image_to_base64(normals_img),
            "roughness_b64": image_to_base64(roughness_img)
        }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
    finally:
        # --- Cleanup: Always remove the temporary directory ---
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"[{job_id}] Cleaned up temporary directory: {temp_dir}")

# Start the RunPod serverless worker only when the script is executed directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
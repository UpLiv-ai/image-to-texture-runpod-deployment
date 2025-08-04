#
# handler_matpal_sam2.py
# Serverless handler for the integrated Material Palette and SAM 2 pipeline.
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

# --- Add repository paths to the system path ---
# This allows us to import modules directly from the cloned repositories.
sys.path.append('/app/MaterialPalette')
sys.path.append('/app/sam2')

# --- Import model-specific modules ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from concept import crop as concept_crop, invert as concept_invert, infer as concept_infer
from capture import get_data as capture_get_data, get_inference_module as capture_get_inference_module
from pytorch_lightning import Trainer

# --- Global State for Models ---
# This dictionary will hold the initialized models. By defining it in the global
# scope, the models are loaded only once when the worker starts (cold start),
# and are reused for subsequent requests (warm starts), dramatically improving performance.
MODELS = {
    "sam_predictor": None,
    "matpal_decomposer": None,
    "pl_trainer": None
}

def init():
    """
    Initializes all models and stores them in the global MODELS dictionary.
    This function is called only once on worker startup.
    """
    global MODELS
    
    print("Initializing models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize SAM 2 Predictor
    print("Loading SAM 2 model...")
    sam_checkpoint = "/runpod-volume/sam2/sam2_hiera_large.pt"
    sam_model_cfg = "/app/sam2/configs/sam2/sam2_hiera_l.yaml"
    
    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"SAM 2 checkpoint not found at {sam_checkpoint}. Ensure it's on the network volume.")
        
    sam_model = build_sam2(sam_model_cfg, sam_checkpoint).to(device)
    MODELS["sam_predictor"] = SAM2ImagePredictor(sam_model)
    print("SAM 2 model loaded.")

    # 2. Initialize Material Palette Decomposition Model
    print("Loading Material Palette decomposition model...")
    matpal_checkpoint = "/runpod-volume/material-palette/model.ckpt"

    if not os.path.exists(matpal_checkpoint):
        raise FileNotFoundError(f"Material Palette checkpoint not found at {matpal_checkpoint}. Ensure it's on the network volume.")

    MODELS["matpal_decomposer"] = capture_get_inference_module(pt=matpal_checkpoint).to(device)
    print("Material Palette model loaded.")

    # 3. Initialize PyTorch Lightning Trainer for decomposition
    # We configure the trainer once and reuse it.
    MODELS["pl_trainer"] = Trainer(accelerator='gpu', devices=1, precision=16)
    print("PyTorch Lightning Trainer initialized.")
    
    print("All models initialized successfully.")


def image_to_base64(image):
    """Converts a PIL Image to a base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handler(job):
    """
    The main handler function for the serverless endpoint.
    Orchestrates the SAM 2 -> Material Palette pipeline.
    """
    # Ensure models are loaded
    if MODELS["sam_predictor"] is None:
        init()

    job_input = job['input']
    
    # --- Input Validation ---
    if 'image' not in job_input:
        return {"error": "Missing 'image' (base64 encoded) in input."}
    if 'points' not in job_input:
        return {"error": "Missing 'points' for SAM prompt in input."}

    # --- Create a temporary directory for this job ---
    # This is crucial for isolating job data and ensuring the file structure
    # required by Material Palette is met.
    job_id = str(uuid.uuid4())
    temp_dir = os.path.join("/tmp", job_id)
    masks_dir = os.path.join(temp_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    image_path = os.path.join(temp_dir, "input_image.png")
    mask_path = os.path.join(masks_dir, "sam_mask.png")

    try:
        # --- Step 1: Decode image and prepare for SAM 2 ---
        image_data = base64.b64decode(job_input['image'])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image.save(image_path)
        
        # SAM expects image as a numpy array
        image_np = np.array(image)
        
        # SAM expects prompts as numpy arrays
        input_points = np.array(job_input['points'])
        input_labels = np.array([p for p in input_points])
        input_points = input_points[:, :2] # Keep only x, y

        # --- Step 2: Run SAM 2 to get the mask ---
        print(f"[{job_id}] Running SAM 2 prediction...")
        predictor = MODELS["sam_predictor"]
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image_np)
            masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False, # We want the single best mask
            )
        
        # The output mask is a boolean numpy array. Convert to an image.
        mask_np = masks
        mask_image = Image.fromarray(mask_np)
        mask_image.save(mask_path)
        print(f"[{job_id}] SAM 2 mask generated and saved to {mask_path}")

        # --- Step 3: Run Material Palette Pipeline Programmatically ---
        print(f"[{job_id}] Starting Material Palette pipeline...")

        # 3a. Crop the region based on the mask (concept.crop)
        # This function expects a directory path and finds the image and mask inside.
        regions = concept_crop(temp_dir)
        print(f"[{job_id}] Cropped regions extracted.")

        # 3b. Invert and Infer for each region (concept.invert, concept.infer)
        # This generates the texture views.
        for region in regions.iterdir():
            print(f"[{job_id}] Inverting concept for region: {region.name}")
            lora = concept_invert(region)
            print(f"[{job_id}] Generating texture for region: {region.name}")
            concept_infer(lora, renorm=True)

        # 3c. Decompose the generated textures (capture module)
        print(f"[{job_id}] Preparing data for decomposition...")
        # The 'sd' dataset points to the generated textures inside the temp_dir
        data_loader = capture_get_data(predict_dir=temp_dir, predict_ds='sd')
        
        print(f"[{job_id}] Running decomposition model...")
        MODELS["pl_trainer"].predict(MODELS["matpal_decomposer"], data_loader)
        print(f"[{job_id}] Decomposition complete.")

        # --- Step 4: Collect and encode output images ---
        output_dir = os.path.join(temp_dir, "lightning_logs/version_0/predict/sd/0/")
        
        # Find the output files (albedo, normals, roughness)
        albedo_path = os.path.join(output_dir, "albedo.png")
        normals_path = os.path.join(output_dir, "normals.png")
        roughness_path = os.path.join(output_dir, "roughness.png")

        if not all(os.path.exists(p) for p in [albedo_path, normals_path, roughness_path]):
            return {"error": "Decomposition did not produce the expected output files."}

        # Read and base64 encode the results
        albedo_img = Image.open(albedo_path)
        normals_img = Image.open(normals_path)
        roughness_img = Image.open(roughness_path)

        return {
            "albedo_b64": image_to_base64(albedo_img),
            "normals_b64": image_to_base64(normals_img),
            "roughness_b64": image_to_base64(roughness_img)
        }

    except Exception as e:
        # Return any errors that occur during the process
        return {"error": f"An error occurred: {str(e)}"}
    finally:
        # --- Cleanup: Always remove the temporary directory ---
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"[{job_id}] Cleaned up temporary directory: {temp_dir}")


# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
# handler.py (Corrected Version)
import base64
import torch
from PIL import Image
import numpy as np
from io import BytesIO
import runpod
import uuid
import tempfile
from pathlib import Path

import sys
import os

# Add the MaterialPalette directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'MaterialPalette'))

from concept import crop as concept_crop, invert as concept_invert, infer as concept_infer
from capture import pbr_capture

# Global dictionary to hold the models
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    """
    Initialize the models and load them into memory.
    """
    if not MODELS:
        print("Initializing Material Palette model...")
        matpal_checkpoint = '/workspace/model.ckpt'

        # Ensure the model file exists
        if not os.path.exists(matpal_checkpoint):
            raise FileNotFoundError(f"Material Palette checkpoint not found at {matpal_checkpoint}")

        # Use the capture module to load the decomposer
        MODELS["matpal_decomposer"] = pbr_capture.get_inference_module(pt=matpal_checkpoint).to(DEVICE)
        print("Material Palette model loaded.")

    print("Model initialization successful.")

def handler(job):
    """
    The main handler function for the RunPod endpoint.
    """
    job_id = job.get("id", uuid.uuid4())
    job_input = job.get("input", {})

    # Ensure model is initialized
    if not MODELS:
        init()

    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided in the input."}

    # Create a temporary directory for this job
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Decode the image and save to temp directory
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_path = temp_dir / "input_image.png"
        image.save(image_path)

        # Create a full-image mask (all white)
        mask = Image.new("L", image.size, 255)
        mask_path = temp_dir / "full_mask.png"
        mask.save(mask_path)
        print(f"[{job_id}] Generated full-image mask and saved to {mask_path}")

        try:
            print(f"[{job_id}] Starting Material Palette pipeline...")
            # --- THIS IS THE CORRECTED PART ---
            # We now pass the Path objects directly without str()
            albedo, normal, roughness, metallic, mat_name = concept_infer(
                temp_dir,
                MODELS["matpal_decomposer"],
                image_path,
                mask_path
            )

            # Convert output images to base64
            def to_b64(img_array):
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8")

            return {
                "albedo_b64": to_b64(albedo),
                "normals_b64": to_b64(normal),
                "roughness_b64": to_b64(roughness),
                "metallic_b64": to_b64(metallic),
                "material_name": mat_name
            }
        except Exception as e:
            return {"error": f"An error occurred: {e}"}

# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
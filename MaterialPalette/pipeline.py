import torch
from pathlib import Path
from PIL import Image
from pytorch_lightning import Trainer
# --- 1. IMPORT THE NEW SDXL PIPELINE ---
from diffusers import StableDiffusionXLInpaintPipeline

# Local Imports from MaterialPalette
import capture

def setup_models():
    """Loads all necessary models into memory once."""
    print("--- Initializing Models ---")
    # 1. Load the PBR decomposer model (no change here)
    pbr_module = capture.get_inference_module(pt='model.ckpt')

    # --- 2. LOAD THE UPGRADED SDXL INPAINTING MODEL ---
    inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", # New, more powerful model
        torch_dtype=torch.float16,
        variant="fp16" # Use the optimized fp16 variant
    ).to("cuda")

    print("✅ Models initialized.")
    return pbr_module, inpaint_pipe

def make_seamless(pipe, image):
    """
    Makes an input image seamlessly tileable using the SDXL inpainting model.
    """
    width, height = image.size

    # Create a mask that covers the edges of the image
    mask = Image.new("L", (width, height), 0)
    mask_data = mask.load()
    edge_width = width // 10

    for x in range(width):
        for y in range(height):
            if x < edge_width or x > width - edge_width or y < edge_width or y > height - edge_width:
                mask_data[x, y] = 255

    # Shift the image to bring the opposite edges together
    # shifted_image = Image.new("RGB", (width, height))
    # shifted_image.paste(image, (-width // 2, -height // 2))
    # shifted_image.paste(image, (width // 2, -height // 2))
    # shifted_image.paste(image, (-width // 2, height // 2))
    # shifted_image.paste(image, (width // 2, height // 2))

    # --- ENHANCED PROMPT FOR SDXL ---
    # SDXL understands more detailed and nuanced prompts
    prompt = prompt = "photorealistic top-down texture, seamless, tileable, perfectly diffuse lighting, no shadows, consistent color tones, high detail, 8k uhd photo"
    negative_prompt = "blurry, low quality, cartoon, watermark, text, signature"

    # Run the inpainting model
    inpainted_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=8.0,
        num_inference_steps=200 # SDXL often needs fewer steps than older models
    ).images[0]
    return inpainted_image

if __name__ == '__main__':
    # ==================================================================
    # === Configuration ===
    # ==================================================================
    INPUT_IMAGE_PATH = "./test_image_1.png"
    OUTPUT_DIR = Path(f"./fast_output_{Path(INPUT_IMAGE_PATH).stem}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    # ==================================================================

    # --- 1. SETUP ---
    pbr_module, inpaint_pipe = setup_models()

    # --- 2. MAKE SEAMLESS ---
    print(f"\n--- Stage 1: Making texture seamless for {INPUT_IMAGE_PATH} ---")
    # Load the image and ensure it's a good size for SDXL (e.g., 1024x1024)
    original_image = Image.open(INPUT_IMAGE_PATH).convert("RGB").resize((512, 512))
    
    seamless_image = make_seamless(inpaint_pipe, original_image)

    # Save the seamless texture so we can inspect it
    seamless_image_path = OUTPUT_DIR / "000_seamless_texture.png"
    seamless_image.save(seamless_image_path)
    print(f"✅ Seamless texture saved to: {seamless_image_path}")

    # --- 3. DECOMPOSE TO PBR MAPS ---
    print("\n--- Stage 2: Decomposing views into PBR maps ---")
    data_module = capture.get_data(predict_dir=OUTPUT_DIR, predict_ds='png')

    trainer = Trainer(default_root_dir=OUTPUT_DIR, accelerator='gpu', devices=1, precision=16)
    trainer.predict(pbr_module, dataloaders=data_module.predict_dataloader())

    print(f"\n✅ Pipeline finished! Check the '{OUTPUT_DIR.name}/predictions' folder for your PBR maps.")
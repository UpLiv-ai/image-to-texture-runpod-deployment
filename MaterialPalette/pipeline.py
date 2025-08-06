import torch
from pathlib import Path
from PIL import Image
from pytorch_lightning import Trainer
from diffusers import StableDiffusionInpaintPipeline

# Local Imports from MaterialPalette
import capture

def setup_models():
    """Loads all necessary models into memory once."""
    print("--- Initializing Models ---")
    # 1. Load the PBR decomposer model
    pbr_module = capture.get_inference_module(pt='model.ckpt')
    
    # 2. Load a pre-trained inpainting model for making textures seamless
    #    This model is optimized for this kind of task.
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    print("✅ Models initialized.")
    return pbr_module, inpaint_pipe

def make_seamless(pipe, image):
    """
    Makes an input image seamlessly tileable using an inpainting model.
    """
    width, height = image.size
    
    # Create a mask that covers the edges of the image
    # The model will "fill in" these masked areas to blend the seams.
    mask = Image.new("L", (width, height), 0)
    mask_data = mask.load()
    
    # Mask the right and bottom edges
    for x in range(width):
        for y in range(height):
            if x > width * 0.75 or y > height * 0.75:
                mask_data[x, y] = 255
    
    # Shift the image to bring the opposite edges together
    # This sets up the inpainting task.
    shifted_image = Image.new("RGB", (width, height))
    shifted_image.paste(image, (-width // 2, -height // 2))
    shifted_image.paste(image, (width // 2, -height // 2))
    shifted_image.paste(image, (-width // 2, height // 2))
    shifted_image.paste(image, (width // 2, height // 2))

    prompt = "a seamless, tileable texture"
    
    # Run the inpainting model
    inpainted_image = pipe(
        prompt=prompt,
        image=shifted_image,
        mask_image=mask,
        guidance_scale=7.5
    ).images[0]
    
    return inpainted_image

if __name__ == '__main__':
    # ==================================================================
    # === Configuration ===
    # ==================================================================
    # Point this to the original, unprepared image file
    INPUT_IMAGE_PATH = "./test_image_1.JPG" 
    
    # Where to save the final PBR maps
    OUTPUT_DIR = Path(f"./fast_output_{Path(INPUT_IMAGE_PATH).stem}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    # ==================================================================

    # --- 1. SETUP ---
    pbr_module, inpaint_pipe = setup_models()
    
    # --- 2. MAKE SEAMLESS ---
    print(f"\n--- Stage 1: Making texture seamless for {INPUT_IMAGE_PATH} ---")
    original_image = Image.open(INPUT_IMAGE_PATH).convert("RGB").resize((512, 512))
    
    seamless_image = make_seamless(inpaint_pipe, original_image)
    
    # Save the seamless texture so we can inspect it
    seamless_image_path = OUTPUT_DIR / "000_seamless_texture.png"
    seamless_image.save(seamless_image_path)
    print(f"✅ Seamless texture saved to: {seamless_image_path}")

    # --- 3. DECOMPOSE TO PBR MAPS ---
    print("\n--- Stage 2: Decomposing views into PBR maps ---")
    
    # Use the library's function to load the single seamless image we just created
    data_module = capture.get_data(predict_dir=OUTPUT_DIR, predict_ds='png')
    
    # Run the final prediction
    trainer = Trainer(default_root_dir=OUTPUT_DIR, accelerator='gpu', devices=1, precision=16)
    trainer.predict(pbr_module, dataloaders=data_module.predict_dataloader())

    print(f"\n✅ Pipeline finished! Check the '{OUTPUT_DIR.name}/predictions' folder for your PBR maps.")
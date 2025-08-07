from pathlib import Path
from pytorch_lightning import Trainer
import concept
import capture
import glob # Keep this import

# --- ADD THIS LINE TO FIX THE ERROR ---
# from capture.data.module import Dataset 
# -------------------------------------


if __name__ == '__main__':
    # Define the path to your test data directory
    test_data_path = Path('./prepared_pipeline_data/run_test_image_2')

    # --- Stage 1: Crop the input image ---
    print("--- Stage 1: Cropping input image ---")
    regions = concept.crop(test_data_path)

    # --- Stage 2: Train LoRA and Generate Texture Views ---
    print("\n--- Stage 2: Training LoRA and generating views ---")
    lora_path = None
    for region in regions.iterdir():
        if region.is_dir(): # Ensure we are processing the '000' directory
            lora_path = concept.invert(region, max_train_steps=50)
            concept.infer(lora_path, renorm=False) # Use renorm=False to simplify output

    if not lora_path:
        raise FileNotFoundError("LoRA training did not produce a valid path.")

    # --- Stage 3: Decompose Views into PBR Maps ---
    print("\n--- Stage 3: Decomposing views into PBR maps ---")
    
    # The generated images are in the 'outputs' subdirectory of the LoRA path.
    generated_images_path = lora_path / "outputs"
    print(f"✅ Loading final images from: {generated_images_path}")

    # The generated files have complex names. We will find them directly.
    # We will not use capture.get_data as it's too restrictive.
    # We find all .png files in the generated_images_path.
    image_files = list(generated_images_path.glob('*.png'))

    if not image_files:
        raise FileNotFoundError(f"No generated PNGs found in {generated_images_path}")

    print(f"Found {len(image_files)} generated images to process.")
    
    # Create a DataModule manually with the correct dataset
    # Now that 'Dataset' is imported, this line will work.
    dataset = Dataset(image_files)
    data_module = capture.data.DataModule(predict_set=dataset, batch_size=1)
    
    # Load the decomposer model
    pbr_module = capture.get_inference_module(pt='model.ckpt')

    # Proceed with the final inference
    trainer = Trainer(default_root_dir=test_data_path, accelerator='gpu', devices=1, precision=16)
    trainer.predict(pbr_module, dataloaders=data_module)

    print("\n✅ pipeline.py test finished successfully! Check for PBR maps.")
from pathlib import Path
from pytorch_lightning import Trainer
import concept
import capture

if __name__ == '__main__':
    # ==================================================================
    # === Configuration ===
    # ==================================================================
    LORA_TRAINING_STEPS = 50
    test_data_path = Path('./prepared_pipeline_data/run_test_image_1')
    # ==================================================================

    # --- Stage 1: Crop the input image ---
    print("--- Stage 1: Cropping input image ---")
    regions = concept.crop(test_data_path)

    # --- Stage 2: Train LoRA and Generate Texture Views ---
    print(f"\n--- Stage 2: Training LoRA for {LORA_TRAINING_STEPS} steps and generating views ---")
    lora_path = None
    for region in regions.iterdir():
        if region.is_dir():
            lora_path = concept.invert(region, max_train_steps=LORA_TRAINING_STEPS)
            # Let the library's own infer function handle the complex LoRA loading
            concept.infer(lora_path, renorm=False)

    if not lora_path:
        raise FileNotFoundError("LoRA training did not produce a valid path.")

    # --- Stage 3: The Bridge - Correctly connecting Stage 2 to Stage 4 ---
    # We know from logs that the generated images are in the 'outputs' subdirectory.
    generated_images_path = lora_path / "outputs"
    print(f"\n--- Stage 3: Preparing to decompose images from: {generated_images_path} ---")

    if not generated_images_path.exists():
        raise FileNotFoundError(f"The expected output directory was not created: {generated_images_path}")

    # Use the library's intended function to load the data from the correct folder.
    data_module = capture.get_data(predict_dir=test_data_path, predict_ds='sd')

    # --- Stage 4: Decompose Views into PBR Maps ---
    print("\n--- Stage 4: Decomposing views into PBR maps ---")
    
    # Load the decomposer model
    pbr_module = capture.get_inference_module(pt='model.ckpt')

    # Proceed with the final inference
    trainer = Trainer(default_root_dir=test_data_path, accelerator='gpu', devices=1, precision=16)
    # The dataloader is now created correctly by the get_data function
    trainer.predict(pbr_module, data_module) 

    print("\n✅ Pipeline finished! Check the 'predictions' folder inside the 'outputs' directory.")
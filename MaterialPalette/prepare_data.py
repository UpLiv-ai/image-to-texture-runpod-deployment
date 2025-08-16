from PIL import Image
from pathlib import Path

# --- Configuration ---
# List your 5 source JPEG/JPG/PNG images here
source_images = [
    "test_image_1.png",
    "test_image_2.png",
    "test_image_3.png",
    "test_image_4.png",
    "test_image_5.jpeg",
    "test_image_6.png",
    "test_image_7.jpeg",
]

# The parent directory where the prepared data folders will be created
output_parent_dir = Path("prepared_pipeline_data")
output_parent_dir.mkdir(exist_ok=True)


# --- Main Script ---
print(f"Starting data preparation for {len(source_images)} images...")

for img_path_str in source_images:
    source_path = Path(img_path_str)
    
    if not source_path.exists():
        print(f"⚠️  Warning: Source image not found, skipping: {source_path}")
        continue

    # 1. Create the main directory for this image
    # e.g., 'prepared_pipeline_data/run_test_image_1'
    run_dir = output_parent_dir / f"run_{source_path.stem}"
    run_dir.mkdir(exist_ok=True)
    
    # 2. Open the image with Pillow
    with Image.open(source_path).convert("RGB") as img:
        # Save it as 'image.png' inside the run directory
        img.save(run_dir / "image.png")
        
        # 3. Create the 'masks' subdirectory
        masks_dir = run_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        # 4. Create a solid white mask with the same dimensions
        # 'L' mode is for 8-bit grayscale. 255 is pure white.
        white_mask = Image.new('L', img.size, 255)
        white_mask.save(masks_dir / "000.png")

    print(f"✅  Successfully prepared data for '{source_path.name}' in '{run_dir}'")

print("\nData preparation complete!")
from pathlib import Path
from pytorch_lightning import Trainer
import concept
import capture
import os
from PIL import Image
import numpy as np

if __name__ == '__main__': 
    # Configuration
    LORA_TRAINING_STEPS = 800
    test_data_path = Path('./prepared_pipeline_data/run_test_image_4')
    BASE_MODEL_PATH = Path('/workspace/models/sd-v1-5')
    os.environ.update({'TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_OFFLINE':'1'})

    # --- START OF MODIFICATIONS ---

    # Configuration for optional, in-place pre-downscaling.
    # Set to True for materials with large patterns like tiles.
    PRE_DOWNSCALE = False
    DOWNSCALE_SIZE = (768, 768)

    if PRE_DOWNSCALE:
        print(f"✅ In-place pre-downscaling enabled. Resizing images to {DOWNSCALE_SIZE}...")
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']
        image_files = [p for ext in image_extensions for p in test_data_path.glob(ext)]

        if not image_files:
            print(f"⚠️ Warning: No images found in {test_data_path}. Nothing to downscale.")
        else:
            for img_path in image_files:
                # Create a path for the backup file, e.g., 'image.png.original'
                backup_path = img_path.with_suffix(img_path.suffix + '.original')

                # Source path for reading the image. If a backup exists, use it.
                # Otherwise, use the original image path and create a backup.
                source_path = img_path
                if not backup_path.exists():
                    # print(f"Backing up {img_path.name} to {backup_path.name}")
                    # img_path.rename(backup_path)
                    # source_path = backup_path
                    pass
                else:
                    pass
                    # print(f"Backup for {img_path.name} already exists. Using it as the source.")

                # Open the source image, resize it, and save it to the original file path.
                with Image.open(source_path) as img:
                    img_resized = img.convert('RGB').resize(DOWNSCALE_SIZE, Image.Resampling.LANCZOS)
                    img_resized.save(img_path) # This saves the resized image as the new 'image.png'
                    print(f"Resized image saved, overwriting {img_path}")
    else:
        print("✅ Pre-downscaling disabled. Using original images.")

    # --- END OF MODIFICATIONS ---
    
    # Stage 1: crop input
    regions = concept.crop(test_data_path)

    # Stage 2: train & infer full top-view
    lora_path = None
    for region in regions.iterdir():
        if region.is_dir():
            lora_path = concept.invert(region, max_train_steps=LORA_TRAINING_STEPS)
            concept.infer(
                path=lora_path,
                renorm=True,
                base_model=BASE_MODEL_PATH,
                num_inference_steps=150,
                resolution=1024,
                prompt='p3',
                stitch_mode='wmean'
             )

    if not lora_path:
        raise FileNotFoundError('LoRA training did not produce a valid path.')

    outputs_dir = lora_path/'out_renorm'
    if not outputs_dir.exists():
        raise FileNotFoundError(f'Missing outputs dir: {outputs_dir}')

    # Stage 3: tile the generated top-view into overlapping patches
    img_size, tile_size, overlap = 1024, 512, 64
    stride = tile_size - overlap
    top = list(outputs_dir.glob('*_1K_*.png'))[0]
    full = np.array(Image.open(top).convert('RGB'), np.float32)/255.0
    patches, coords = [], []
    for y0 in range(0, img_size, stride):
        for x0 in range(0, img_size, stride):
            y1, x1 = min(y0+tile_size,img_size), min(x0+tile_size,img_size)
            y0_, x0_ = y1-tile_size, x1-tile_size
            coords.append((y0_,y1,x0_,x1))
            patches.append(full[y0_:y1,x0_:x1,:])

    # dump all patches into outputs_dir/sd/
    # sd_dir = outputs_dir / 'sd'
    # sd_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, patch in enumerate(patches):
        out_name = f'patch_{idx}.png'          # names like patch_0.png, patch_1.png, ...
        img = Image.fromarray((patch * 255).astype(np.uint8))
        img.save(outputs_dir / out_name)

        # run decomposer on all patches by pointing directly to the staging folder
    # omit predict_ds so the DataModule loads all .png in staging
    dm = capture.get_data(
      predict_dir=test_data_path,
      predict_ds='sd'    # must be 'sd'
    )
    dm.setup('predict')
    loader = dm.predict_dataloader()
    pm = capture.get_inference_module(pt='/workspace/model.ckpt')
    trainer = Trainer(default_root_dir=outputs_dir, accelerator='gpu', devices=1, precision=16)
    trainer.predict(pm, dataloaders=loader)

    # Stage 5: gather decomposed maps per patch
    albedo_patches, rough_patches, norm_patches = [], [], []
    for idx in range(len(patches)):
        base = f'patch_{idx}'
        albedo = Image.open(outputs_dir/f'{base}_albedo.png')
        rough = Image.open(outputs_dir/f'{base}_roughness.png')
        norm = Image.open(outputs_dir/f'{base}_normals.png')
        albedo_patches.append(np.array(albedo, np.float32)/255.0)
        rough_patches.append(np.array(rough, np.float32)/255.0)
        norm_patches.append(np.array(norm, np.float32)/255.0)

    # Stage 6: blend back into full maps
    w = np.minimum(np.linspace(0,1,tile_size), np.linspace(0,1,tile_size)[::-1])
    window = np.outer(w,w)
    def blend(plist):
        canvas = np.zeros((img_size,img_size,3),np.float32)
        weight = np.zeros((img_size,img_size,1),np.float32)
        for patch,(y0,y1,x0,x1) in zip(plist,coords):
            canvas[y0:y1,x0:x1] += patch*window[:,:,None]
            weight[y0:y1,x0:x1] += window[:,:,None]
        return canvas/(weight+1e-6)

    alb = blend(albedo_patches)
    rough = blend(rough_patches)
    norm = blend(norm_patches)

    # save final PBR maps
    pred_dir = outputs_dir/'predictions'
    pred_dir.mkdir(exist_ok=True)
    Image.fromarray((alb.clip(0,1)*255).astype(np.uint8)).save(pred_dir/'albedo_1K.png')
    Image.fromarray((rough.clip(0,1)*255).astype(np.uint8)).save(pred_dir/'roughness_1K.png')
    Image.fromarray((norm.clip(0,1)*255).astype(np.uint8)).save(pred_dir/'normal_1K.png')
    print('✅ Done: high-res PBR maps at', pred_dir)

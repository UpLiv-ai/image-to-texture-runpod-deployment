from huggingface_hub import snapshot_download

# This will download the Stability AI 4x upscaler model
# into your models/sd-x4-upscaler folder
snapshot_download(
    repo_id="stabilityai/stable-diffusion-x4-upscaler",
    repo_type="model",
    local_dir="models/sd-x4-upscaler",
    local_dir_use_symlinks=False
)
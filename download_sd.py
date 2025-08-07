from huggingface_hub import snapshot_download

# this will download everything under runwayml/stable-diffusion-v1-5
# into your models/sd-v1-5 folder
snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    repo_type="model",
    local_dir="models/sd-v1-5",
    local_dir_use_symlinks=False  # ensure it copies files, not just symlinks
)

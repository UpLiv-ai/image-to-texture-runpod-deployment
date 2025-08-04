#
# Dockerfile for the Integrated Material Palette + SAM 2 Serverless Endpoint
#

# Step 1: Start from a modern RunPod PyTorch image with development tools
# This image provides a compatible Python/PyTorch/CUDA stack and the necessary
# compilers (gcc, nvcc) for building custom kernels required by SAM 2.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Step 2: Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Step 3: Install essential system packages
# git and wget are needed for cloning repositories and downloading files.
# libgl1 is a dependency for OpenCV.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Step 4: Set the working directory
WORKDIR /app

# Step 5: Clone the required model repositories from GitHub
RUN git clone https://github.com/astra-vision/MaterialPalette.git
RUN git clone https://github.com/facebookresearch/sam2.git

# Step 6: Install all Python dependencies in a unified block
# This is the most critical step, where dependency conflicts are resolved.
RUN \
    # First, upgrade pip itself
    python3 -m pip install --upgrade pip && \
    # Install the RunPod SDK for serverless workers
    pip install runpod && \
    # Upgrade torch and torchvision to meet SAM 2's strict requirement (>=2.5.1)
    # This is the cornerstone of our dependency resolution strategy.
    pip install torch>=2.5.1 torchvision>=0.20.1 --index-url https://download.pytorch.org/whl/cu121 && \
    # Install SAM 2. The '-e' flag installs it in editable mode, which is standard
    # for this repository and will trigger the compilation of its custom CUDA kernels.
    pip install -e./sam2 && \
    # Now, install the dependencies for Material Palette.
    # We install them from its `deps.yml` but must be careful.
    # The original deps.yml specifies pytorch, which we've already handled.
    # We will install the other dependencies directly via pip.
    pip install \
    "lightning==1.8.3" \
    "diffusers==0.19.3" \
    "peft==0.5.0" \
    "opencv-python" \
    "jsonargparse" \
    "easydict"

# Step 7: Copy the handler script into the container's working directory
COPY handler_matpal_sam2.py /app/handler.py

# Step 8: Define the default command to start the RunPod serverless worker
# The '-u' flag ensures that Python output is unbuffered, which is good practice for logging.
CMD ["python", "-u", "handler.py"]
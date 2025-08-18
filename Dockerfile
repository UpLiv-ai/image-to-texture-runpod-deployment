FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Basic dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        git wget bzip2 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set WORKDIR before any file operations
WORKDIR /app

# Download & install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /app/miniconda.sh && \
    bash /app/miniconda.sh -b -p /opt/conda && \
    rm /app/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# --- FIX: Provide the specific channels when accepting the Terms of Service ---
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create your designated environment from the YAML file
COPY deps-conda.yml .
RUN conda env create -f deps-conda.yml

# Copy application code
COPY . .

# Install pip requirements directly into the 'matpal' environment.
RUN conda run -n matpal pip install -r requirements.txt && \
    conda run -n matpal pip install scipy runpod

# Set the ENTRYPOINT to ensure all subsequent commands run within the matpal environment.
ENTRYPOINT ["conda", "run", "-n", "matpal", "--no-capture-output"]

# The CMD is now appended to the ENTRYPOINT
CMD ["python", "-u", "handler.py"]
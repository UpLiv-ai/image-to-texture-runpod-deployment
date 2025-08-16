# Use the base image you specified
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install basic dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        bzip2 \
        libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
WORKDIR /root
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH
ENV PATH=/root/miniconda3/bin:$PATH

# Clone your repo
WORKDIR /app
RUN git clone https://github.com/UpLiv-ai/image-to-texture-runpod-deployment.git .
    
# Create conda environment from deps-conda.yml
RUN conda env create -f deps-conda.yml

# Activate environment by default
SHELL ["conda", "run", "-n", "matpal", "/bin/bash", "-c"]

# Install pip requirements inside conda env
RUN pip install -r requirements.txt && \
    pip install scipy runpod

# Copy handler file (if you want to override existing one, otherwise skip)
COPY handler.py /app/handler.py

# Run handler.py with conda env
CMD ["conda", "run", "-n", "matpal", "python", "-u", "handler.py"]

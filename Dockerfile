FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Basic dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        git wget bzip2 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Download & silently install Miniconda (batch mode, auto-license acceptance)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Update PATH for conda
ENV PATH=/root/miniconda3/bin:$PATH

# Initialize Conda for bash (makes conda activate available)
RUN conda init bash

WORKDIR /app

# Create your designated environment
RUN conda env create -f deps-conda.yml

# Make new shell always use the matpal env
RUN echo "conda activate matpal" >> /root/.bashrc

# Switch default shell to login bash to trigger conda init + auto-activate
SHELL ["/bin/bash", "--login", "-c"]

# Install pip requirements inside the matpal environment
RUN pip install -r requirements.txt && \
    pip install scipy runpod

# Ensure your handler runs in the matpal env
CMD ["python", "-u", "handler.py"]

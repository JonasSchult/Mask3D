# or you can pull from xhm126/mask3d:v1.0
FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libjpeg-dev \
    vim \
    gcc \
    make \
    htop \
    python \
    wget \
    apt-utils \
    zip \
    unzip \
    python-numpy \
    python3 \
    python3-cups \
    python3-numpy \
    gzip \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-setuptools \
    python3-wheel \
    sudo \
    ninja-build \
    libopenblas-dev \
    xterm \
    xauth \
    openssh-server \
    tmux \
    mate-desktop-environment-core \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# python 3.10 conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

# packages
RUN conda update -n base -c defaults conda && \
    conda install openblas-devel -c anaconda && \
    pip3 install --upgrade pip && \
    pip3 install setuptools && \
    pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu116.html && \
    pip3 install pytorch-lightning==1.7.2 fire==0.4.0 imageio==2.21.1 tqdm wandb python-dotenv==0.20.0 pyviz3d==0.2.28 scipy==1.9.0 plyfile==0.7.4 scikit-learn==1.1.2 trimesh==3.14.0 loguru==0.6.0 albumentations==1.2.1 volumentations==0.1.8 && \
    pip3 install open3d && \
    pip3 install ninja==1.10.2.3 && \
    pip3 install antlr4-python3-runtime==4.8 && \
    pip3 install black==21.4b2 && \
    pip3 install omegaconf==2.0.6 hydra-core==1.0.5 

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
    
# git
RUN mkdir -p /workspace/project/script && mkdir -p /data && \
    pip3 install natsort fvcore==0.1.5.post20220512 cloudpickle==2.1.0 pycocotools==2.0.4 && \    
    pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps && \
    git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" /workspace/MinkowskiEngine && \
    cd /workspace/MinkowskiEngine/ && \
    python setup.py install --force_cuda --blas=openblas && \
    git clone https://github.com/ScanNet/ScanNet.git /workspace/project/ScanNet && \
    cd /workspace/project/ScanNet/Segmentator/ && \
    make && \
    git clone https://github.com/JonasSchult/Mask3D.git /workspace/project/Mask3D && \
    mkdir -p /workspace/project/Mask3D/eval_output && \
    cd /workspace/project/Mask3D/third_party/pointnet2 && python setup.py install 


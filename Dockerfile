ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN dpkg --add-architecture i386
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
                                         wget mc git unzip unrar\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install --upgrade pip
RUN pip install -r /mmdetection/requirements/build.txt
RUN pip install --no-cache-dir -e .
WORKDIR /

# Install telebot
RUN python3 -m pip install pyTelegramBotAPI

# Install jupyter hub
RUN python3 -m pip install jupyterhub
RUN python3 -m pip install notebook
RUN python3 -m pip install labelme2coco
RUN python3 -m pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension

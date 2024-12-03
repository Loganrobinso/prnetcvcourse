FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# APT INSTALL (for CV2)
RUN apt-get update && apt-get install libgl1 ffmpeg libsm6 libxext6 -y

# PIP
RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt /tmp
RUN python3 -m pip install -r /tmp/requirements.txt --no-warn-script-location

# SETUP HUGGINGFACE MODEL CACHE
RUN mkdir /.cache
RUN chmod 777 /.cache

# APP DIRECTORY PATH
WORKDIR /app
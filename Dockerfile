FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN apt-get update && pip install --upgrade pip && apt-get install libsm6 libxext6 libxrender-dev
CMD bash

FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN apt-get update && pip install --upgrade pip
CMD bash

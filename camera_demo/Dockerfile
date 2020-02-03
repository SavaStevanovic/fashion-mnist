FROM tensorflow/tensorflow:1.15.0-py3
RUN apt-get update && pip install --upgrade pip && apt-get install -y libsm6 libxext6 libxrender-dev nano
ENV QT_X11_NO_MITSHM=1
CMD bash

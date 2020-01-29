FROM tensorflow/tensorflow:1.15.0-gpu-py3
COPY ./project /app
RUN apt-get update && pip install --upgrade pip && pip install -e ./app/. && pip install -r ./app/requirements.txt
CMD bash

FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV SRC_DIR /src

COPY src $SRC_DIR
WORKDIR $SRC_DIR

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
RUN chmod +x ./train.sh ./inference.sh
RUN pip install -r requirements.txt

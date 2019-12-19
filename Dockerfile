FROM test:4

ENV SRC_DIR /src

COPY src $SRC_DIR
WORKDIR $SRC_DIR

RUN chmod +x ./train.sh ./inference.sh
RUN pip install -r requirements.txt

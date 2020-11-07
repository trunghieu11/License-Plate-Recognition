FROM ubuntu:18.04

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get install -y libsm6 libxext6 libxrender-dev \
  && apt-get install mlocate \
  && DEBIAN_FRONTEND=noninteractive apt-get -y install libopencv-dev \
  && apt install ffmpeg

WORKDIR /src
# COPY . /src

COPY requirements.txt /src/
RUN pip3 install -r requirements.txt

# CMD python3 predict_video.py
CMD gunicorn -b 0.0.0.0:8888 web:server
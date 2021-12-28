FROM tensorflow/tensorflow:2.5.1-gpu
ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev redis-server cifs-utils libgl1-mesa-dev gettext

RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/
RUN pip install -r requirements.txt
ADD . /code/
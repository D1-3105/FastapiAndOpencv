FROM python:3.10-slim

RUN mkdir /app/
COPY . /app/
WORKDIR /app/
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libgtk2.0-dev -y
RUN apt-get install pkg-config -y
RUN pip install -r requirements.txt


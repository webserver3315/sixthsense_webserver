FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install -y python3-pip python3-dev build-essential
RUN pip3 install scikit-build
RUN pip3 install cmake
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

RUN apt-get install -y sudo
RUN sudo apt-get install -y libgl1-mesa-dev
COPY . .

CMD [ "gunicorn", "-b 0.0.0.0:8000", "app:app"]
# CMD [ "python", "./your-daemon-or-script.py" ]
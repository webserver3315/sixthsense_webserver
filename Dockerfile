FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

WORKDIR /usr/src/app

RUN apt-get install -y python-pip python-dev build-essential
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install -y sudo
RUN sudo apt-get install -y libgl1-mesa-dev
COPY . .

CMD [ "gunicorn", "-b 0.0.0.0:8000", "app:app"]
# CMD [ "python", "./your-daemon-or-script.py" ]
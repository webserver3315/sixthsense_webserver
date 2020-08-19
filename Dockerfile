FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN sudo apt-get update
RUN sudo apt-get install -y libgl1-mesa-dev
COPY . .

CMD [ "gunicorn", "-b 0.0.0.0:8000", "app:app"]
# CMD [ "python", "./your-daemon-or-script.py" ]
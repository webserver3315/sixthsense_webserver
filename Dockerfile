FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "gunicorn", "-b 0.0.0.0:8000", "app:app"]
# CMD [ "python", "./your-daemon-or-script.py" ]
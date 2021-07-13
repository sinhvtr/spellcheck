# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
# RUN export https_proxy='http://10.60.28.99:81'
CMD ["python", "app.py"]

FROM python:3.9-buster

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .  

ENV PYTHONPATH=/app

ENTRYPOINT ["python"]
CMD ["./app_files/app.py", "--host=0.0.0.0"]


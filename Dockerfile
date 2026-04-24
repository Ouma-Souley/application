FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY data.csv .
COPY src ./src
COPY app ./app

RUN chmod +x ./app/run.sh

EXPOSE 8000

CMD ["bash", "-c", "./app/run.sh"]
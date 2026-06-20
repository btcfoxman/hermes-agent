FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY ai_marketing_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ai_marketing_api/main.py /app/main.py

EXPOSE 8095
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8095"]

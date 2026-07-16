FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY ai_marketing_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ai_marketing_api /app/ai_marketing_api

EXPOSE 8095
CMD ["uvicorn", "ai_marketing_api.main:app", "--host", "0.0.0.0", "--port", "8095"]

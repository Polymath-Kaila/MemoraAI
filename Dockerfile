
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
 Copy Google credentials securely if needed
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/memoraai-475411-6eeaab22f2f3.json"
EXPOSE 8000
EXPOSE 8501



CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]







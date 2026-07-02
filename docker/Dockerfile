FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --timeout 120 fastapi==0.111.0 uvicorn==0.30.0 pydantic==2.7.0 python-dotenv==1.0.0

RUN pip install --no-cache-dir --timeout 300 torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 120 transformers==4.44.0 sentence-transformers==3.0.0

RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Walid692/finsentinel-camembert')"

COPY src/api/main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
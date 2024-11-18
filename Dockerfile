FROM python:3.12-slim
COPY requirements.txt /app/requirements.txt

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]

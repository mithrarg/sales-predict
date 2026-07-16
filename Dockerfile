# 1. Base Image
FROM python:3.10-slim

# 2. Install Linux System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Establish Working Directory
WORKDIR /app

# 4. Copy and Install Python Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Web Application Source Code
COPY . .

# 6. Networking Configuration
EXPOSE 10000

# 7. Start Production Server Execution
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]

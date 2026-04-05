FROM python:3.12-slim

WORKDIR /app

# Install server dependencies only — no ML libs, no openai client
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source packages
COPY env/       ./env/
COPY tasks/     ./tasks/
COPY graders/   ./graders/
COPY server/    ./server/

# Copy fixtures — required at runtime for fixture loading
COPY fixtures/  ./fixtures/

# Copy openenv manifest
COPY openenv.yaml .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.11.1-slim
RUN apt-get update
RUN apt-get install -y git
WORKDIR /
RUN git clone --depth 1 https://github.com/gaurishg/translango-fastapi-backend.git translango-server
WORKDIR /translango-server
RUN pip install -r requirements.txt
ENTRYPOINT uvicorn app:app --host 0.0.0.0 --port $PORT --env-file SECRETS/.env

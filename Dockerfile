FROM python:3.11.1-slim
RUN apt update
RUN apt-get install -y git
WORKDIR /
RUN git clone --depth 1 https://github.com/UTokyo-PBL/yolov7-flask.git translango-server
WORKDIR /translango-server
RUN pip install -r requirements.txt
COPY --chown=root:root .aws/ /root/.aws/
COPY --chown=root:root SECRETS /translango-server/SECRETS
ENTRYPOINT uvicorn app:app --host 0.0.0.0 --port $PORT --env-file SECRETS/.env

FROM python:3.11.1
RUN apt update
WORKDIR /
RUN git clone --depth 1 https://github.com/UTokyo-PBL/yolov7-flask.git translango-server
WORKDIR /yolov7-flask
RUN pip install -r requirements.txt
COPY --chown=root:root .aws/ /root/
COPY --chown=root:root SECRETS /yolov7-flask/
ENTRYPOINT uvicorn app:app --host 0.0.0.0 --port $PORT --env-file SECRETS/.env

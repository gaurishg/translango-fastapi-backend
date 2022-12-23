FROM gaurishg/yolov7:nvidia-pytorch-22.11-py3
RUN apt update
RUN apt install -y git python3-pip python3-venv zip htop screen libgl1-mesa-glx wget
WORKDIR /
RUN git clone https://github.com/UTokyo-PBL/yolov7-flask.git
WORKDIR /yolov7-flask
RUN pip install -r requirements.txt --no-cache-dir
WORKDIR /yolov7-flask/webapp
CMD exec gunicorn --bind :80 --timeout 0 app:app
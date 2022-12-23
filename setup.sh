#!/usr/bin/env bash

sudo apt update
sudo apt install -y git python3-pip python3-venv zip htop screen libgl1-mesa-glx wget
python3 -m venv project_env
source project_env/bin/activate
pip install -r requirements.txt --no-cache-dir
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
cd ..
sudo project_env/bin/python webapp/app.py
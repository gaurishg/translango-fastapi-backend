# yolov7-flask

This repository combines YOLOv7 and Flask to provide an API which can be used for obtaining object bounding boxes.

## Project Initialization
If you are using ubuntu, follow these steps for setting up the project
1. `sudo apt update`
2. `sudo apt install -y git python3-pip python3-venv zip htop screen libgl1-mesa-glx wget`
3. `git clone https://github.com/UTokyo-PBL/yolov7-flask.git`
4. `cd yolov7-flask`
5. `python3 -m venv project_env`
6. `source project_env/bin/activate`
7. `pip install -r yolov7/requirements.txt`
8. `pip install flask`
9. `cd yolov7`
10. `wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt`
11. `cd ..`
12. `sudo project_env/bin/python webapp/app.py`

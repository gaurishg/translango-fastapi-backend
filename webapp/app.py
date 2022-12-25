#app.py

import sys
import os
current_path = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(current_path, '..')))
sys.path.append(os.path.abspath(os.path.join(current_path, '..', 'yolov7')))
# print(sys.path)
# import yolov7.translango
from typing import List, Dict, Tuple # type: ignore

from fastapi import FastAPI

app = FastAPI()

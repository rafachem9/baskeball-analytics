import os
from ultralytics import YOLO
from utils import INPUT_TEST_DIR, MODEL_DIR

MODEL_NAME = 'best.pt'
VIDEO_NAME = 'nba_test_image.jpg'
MODEL_TRAIN_DIR = 'train16'

# Configurations
CONFIDENCE_LEVEL = 0.6

# Paths
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_TRAIN_DIR, 'weights', MODEL_NAME)
IMAGE_PATH = os.path.join(INPUT_TEST_DIR, VIDEO_NAME)

# Cargar el modelo entrenadoq
model = YOLO(MODEL_PATH)

# Hacer inferencia en una imagen
results = model(IMAGE_PATH,
                save=True,
                imgsz=640,
                conf=CONFIDENCE_LEVEL)

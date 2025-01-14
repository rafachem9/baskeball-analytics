import os
import cv2
from ultralytics import YOLO
from utils import INPUT_TEST_DIR, MODEL_DIR

MODEL_NAME = 'best.pt'
VIDEO_NAME = 'partido-mas-telde-san-isidro.mp4'
DIAGRAM_IMAGE_NAME = 'basket_court.png'
MODEL_TRAIN_DIR = 'train16'

# Configurations
CONFIDENCE_LEVEL = 0.6

# Paths
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_TRAIN_DIR, 'weights', MODEL_NAME)
VIDEO_PATH = os.path.join(INPUT_TEST_DIR, VIDEO_NAME)

# Cargar el modelo entrenado
model = YOLO(MODEL_PATH)

# Leer un video
cap = cv2.VideoCapture(VIDEO_PATH)  # Cambia '0' para usar una webcam


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar inferencia en cada cuadro
    results = model.predict(frame, conf=CONFIDENCE_LEVEL)

    # Dibujar las detecciones en el cuadro
    annotated_frame = results[0].plot()

    # Mostrar el cuadro procesado
    cv2.imshow('YOLO Inference', annotated_frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

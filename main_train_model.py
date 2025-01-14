import os
from ultralytics import YOLO
from functions.executions_utils import execution
from utils import PROJECT_DIR


def main():
    """
    Entrenar el modelo de detección de balones
    :return:
    """
    data_yaml = os.path.join(PROJECT_DIR, 'datasets', 'data.yaml')

    # Crear y entrenar el modelo
    # Parece que con el modelo L y XL no se puede entrenar, solo con el modelo S y M. He probado m con pocos epocs y no detecta nada.
    model = YOLO('yolov8m.yaml')

    model.train(data=data_yaml,
                epochs=100,
                imgsz=640
                )


if __name__ == '__main__':
    execution(main, 'Train model')

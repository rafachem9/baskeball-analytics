from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time
from utils import PROJECT_DIR
from utils_images.functions_util import get_rectangule_coords

# Configuración
model_path = f'{PROJECT_DIR}/datasets/input-test/best.pt'
video_path = f'{PROJECT_DIR}/datasets/input-test/partido-mas-telde-san-isidro.mp4'
# video_path = f'{PROJECT_DIR}/datasets/input-test/grancanariavslanzarote.mp4'

# https://www.kaggle.com/code/nityampareek/using-deepsort-object-tracker-with-yolov5
# https://medium.com/@gayathri.s.de/object-detection-and-tracking-with-yolov8-and-deepsort-5d5981752151

objects_names = ['Ball', 'Hoop', 'Period', 'Player', 'Ref', 'Shot Clock', 'Team Name', 'Team Points', 'Time Remaining']
nivel_confianza = 0.6

# Cargar el modelo y DeepSORT
model = YOLO(model_path)
deepsort = DeepSort()
deepsort_ref = DeepSort()
deepsort_ball = DeepSort()

# Leer el video
cap = cv2.VideoCapture(video_path)


def get_objects(boxes, nivel_confianza, objects_names, class_name):
    detections = []
    for i, box in enumerate(boxes):
        x_center, y_center, w, h = box

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # Check if the box is inside the frame
        # cv2.rectangle(frame, (x1, y1, int(w), int(h)), (0, 0, 255), 2)
        # cv2.putText(frame, f'ID: {i}', (int(box[0]), int(box[1]) - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        box_update = [x1, y1, int(w), int(h)]
        object_name = objects_names[int(class_ids[i])]

        if confidences[i] > nivel_confianza:
            if object_name == class_name:
                print(box_update)
                detections.append((box_update, confidences[i], object_name))
    return detections


def plot_result(frame, tracks, class_name, color):
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()
        track_id = track.track_id
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.putText(frame, f'{class_name}: {track_id}', (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


while cap.isOpened():
    ret, frame = cap.read()
    start = time.perf_counter()
    if not ret:
        break

    # Inferencia con YOLO
    results = model.predict(frame, conf=nivel_confianza)
    boxes = results[0].boxes.xywh.tolist()
    confidences = results[0].boxes.conf.tolist()
    class_ids = results[0].boxes.cls.tolist()

    # PLot detections on the frame
    img = results[0].plot()
    cv2.imshow('detect', img)

    # get player detections
    detections_player = get_objects(boxes, nivel_confianza, objects_names, 'Player')
    detections_referee = get_objects(boxes, nivel_confianza, objects_names, 'Ref')
    detections_ball = get_objects(boxes, nivel_confianza, objects_names, 'ball')

    # Follow the players
    tracks_player = deepsort.update_tracks(detections_player, frame=frame)
    frame = plot_result(frame, tracks_player, 'Player', color=(52, 161, 235))

    # Follow the referee
    tracks_referee = deepsort_ref.update_tracks(detections_referee, frame=frame)
    frame = plot_result(frame, tracks_referee, 'Ref', color=(128, 128, 128))

    # Follow the ball
    tracks_ball = deepsort_ball.update_tracks(detections_ball, frame=frame)
    frame = plot_result(frame, tracks_ball, 'Ball', color=(0, 255, 0))

    # Mostrar FPS
    end = time.perf_counter()
    fps = 1 / (end - start)
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

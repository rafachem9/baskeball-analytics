import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import INPUT_TEST_DIR, MODEL_DIR

MODEL_NAME = 'best.pt'
VIDEO_NAME = 'partido-mas-telde-san-isidro.mp4'
DIAGRAM_IMAGE_NAME = 'basket_court.png'
MODEL_TRAIN_DIR = 'train16'

# Configurations
OBJECT_NAMES = ['Ball', 'Hoop', 'Period', 'Player', 'Ref', 'Shot Clock', 'Team Name', 'Team Points', 'Time Remaining']
CONFIDENCE_LEVEL = 0.6

# Paths
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_TRAIN_DIR, 'weights', MODEL_NAME)
VIDEO_PATH = os.path.join(INPUT_TEST_DIR, VIDEO_NAME)
DIAGRAM_IMAGE_PATH = os.path.join(INPUT_TEST_DIR, DIAGRAM_IMAGE_NAME)

# Court coordinates for homography
COURT_COORDS = np.array(
    [[-100, 750], [1500, 650], [1100, 450], [100, 480]], np.int32
)
DIAGRAM_POINTS = np.array([[22, 22], [642, 22], [642, 388], [22, 388]])
MATRIX_TRANSFORMATION, _ = cv2.findHomography(COURT_COORDS, DIAGRAM_POINTS)


def extract_detections(yolo_results, confidence_level, object_names, target_class):
    """
    Extract detections for a specific object class from YOLO results.
    """
    boxes = yolo_results[0].boxes.xywh.tolist()
    confidences = yolo_results[0].boxes.conf.tolist()
    class_ids = yolo_results[0].boxes.cls.tolist()

    detections = []
    for i, box in enumerate(boxes):
        if confidences[i] > confidence_level and object_names[int(class_ids[i])] == target_class:
            x_center, y_center, w, h = box
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            detections.append(([x1, y1, int(w), int(h)], confidences[i], target_class))
    return detections


def plot_tracks(image, tracks, class_name, color, diagram=None, homography_matrix=None):
    """
    Plot tracked objects on the frame and optionally map them to the diagram.
    """
    for track in tracks:
        if not track.is_confirmed():
            continue

        # Bounding box and ID
        bbox = track.to_ltrb()
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box and label on the video frame
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{class_name} {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if diagram is not None and homography_matrix is not None:
            # Compute the center of the bounding box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(image, (center_x, y2), 5, color, -1)

            center_point = np.array([[[center_x, y2]]], dtype=np.float32)

            # Transform the center point using the homography matrix
            transformed_center = cv2.perspectiveTransform(center_point, homography_matrix)[0][0]
            transformed_center = tuple(map(int, transformed_center))

            # Draw the point on the diagram
            cv2.circle(diagram, transformed_center, 5, color, -1)
            cv2.putText(diagram, f'{class_name} {track_id}', (transformed_center[0] - 20, transformed_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    """
    Main function to run the basketball tracking system.
    :return:
    """
    # Load YOLO model and DeepSORT trackers
    model = YOLO(MODEL_PATH)
    deepsort = DeepSort()
    deepsort_ref = DeepSort()
    deepsort_ball = DeepSort()

    # Video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()

        # Load the diagram image
        diagram_image = cv2.imread(DIAGRAM_IMAGE_PATH)

        # Draw the court area on the frame
        cv2.polylines(frame, [COURT_COORDS], isClosed=True, color=(0, 255, 0), thickness=2)

        # Perform YOLO detection
        results = model.predict(frame, conf=CONFIDENCE_LEVEL)

        # Show YOLO detections for visualization
        cv2.imshow('YOLO Detections', results[0].plot())

        # Extract detections for players, referees, and the ball
        detections_player = extract_detections(results, CONFIDENCE_LEVEL, OBJECT_NAMES, 'Player')
        detections_referee = extract_detections(results, CONFIDENCE_LEVEL, OBJECT_NAMES, 'Ref')
        detections_ball = extract_detections(results, CONFIDENCE_LEVEL, OBJECT_NAMES, 'Ball')

        # Update and plot tracks for players, referees, and the ball
        plot_tracks(frame, deepsort.update_tracks(detections_player, frame=frame), 'Player', (255, 0, 0),
                    diagram=diagram_image, homography_matrix=MATRIX_TRANSFORMATION)
        plot_tracks(frame, deepsort_ref.update_tracks(detections_referee, frame=frame), 'Ref', (128, 128, 128),
                    diagram=diagram_image, homography_matrix=MATRIX_TRANSFORMATION)
        plot_tracks(frame, deepsort_ball.update_tracks(detections_ball, frame=frame), 'Ball', (0, 255, 0),
                    diagram=diagram_image, homography_matrix=MATRIX_TRANSFORMATION)

        # Compute and display FPS
        fps = 1 / (time.perf_counter() - start_time)
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Show the tracking results and the diagram
        cv2.imshow('Video Tracking', frame)
        cv2.imshow('Court Diagram', diagram_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

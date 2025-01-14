import os
import cv2


def visualize_annotations(image_folder, label_folder, output_folder, class_names):
    os.makedirs(output_folder, exist_ok=True)
    for img_file in os.listdir(image_folder):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):
            img_path = os.path.join(image_folder, img_file)
            label_path = os.path.join(label_folder, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

            img = cv2.imread(img_path)
            h, w, _ = img.shape

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, class_names[int(class_id)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)

            cv2.imwrite(os.path.join(output_folder, img_file), img)


# Cambia estas rutas según tu dataset
visualize_annotations('datasets/train/images',
                      'datasets/train/labels',
                      'datasets/output/v2',
                      class_names=['Ball', 'Hoop', 'Period', 'Player', 'Ref', 'Shot Clock', 'Team Name', 'Team Points', 'Time Remaining'])
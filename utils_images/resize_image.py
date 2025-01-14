import cv2
from utils import PROJECT_DIR_FIX

INPUT_TEST_DIR = f'{PROJECT_DIR_FIX}/datasets/input-test'

# Cargar la imagen
image = cv2.imread(f'{INPUT_TEST_DIR}/img.png')
# Obtener dimensiones
height, width, channels = image.shape
print(f"Ancho: {width}, Alto: {height}, Canales: {channels}")

# resize image to imgsz=640
imgsz = 640
image = cv2.resize(image, (imgsz, imgsz))

#save image
cv2.imwrite(f'{INPUT_TEST_DIR}/image-test-1.png', image)

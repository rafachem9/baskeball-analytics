import cv2
import numpy as np
from utils import PROJECT_DIR

# Crear una imagen negra (720x1280 píxeles)
image = np.zeros((720, 1280, 3), dtype="uint8")

# Abrir la imagen del diagrama de la cancha
diagram_image_path = f'{PROJECT_DIR}/datasets/input-test/basket_court.png'
diagram_image = cv2.imread(diagram_image_path)
print("Diagram image shape:", diagram_image.shape)

height, width, _ = diagram_image.shape

# Abrir la imagen del video (o imagen de prueba)
video_path = f'{PROJECT_DIR}/datasets/input-test/holograpic_test.png'
image = cv2.imread(video_path)
print("Video image shape:", image.shape)

# Coordenadas del trapecio (por ejemplo, pueden representar un área del campo de juego)
trap_coords = np.array(
    [[-100, 700],  # Esquina inferior izquierda
     [1500, 650],  # Esquina inferior derecha
     [1100, 450],  # Esquina superior derecha
     [100, 480]],  # Esquina superior izquierda
    np.int32)

# Dibujar el contorno del trapecio en la imagen del video
color = (0, 255, 0)  # Verde
thickness = 2
cv2.polylines(image, [trap_coords], isClosed=True, color=color, thickness=thickness)

# Puntos del diagrama para calcular la homografía
diagram_points = np.array([[0, 0], [width, 0], [width, height], [0, height]])
homography_matrix, status = cv2.findHomography(trap_coords, diagram_points)

# Coordenadas del jugador en el video
player_coords = [85, 557, 55, 153]  # x1, y1, w, h
x1, y1, w, h = player_coords
center_x = x1 + w // 2
center_y = y1 + h // 2

# Dibujar un círculo para el jugador en la imagen del video
cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

# Crear un punto para el centro del jugador
center_point = np.array([[[center_x, center_y]]], dtype=np.float32)  # Debe ser de forma (1, 1, 2)

# Aplicar la homografía
transformed_center = cv2.perspectiveTransform(center_point, homography_matrix)

# Asegurarse de que la transformación es exitosa
transformed_center = transformed_center[0][0]  # Extraer el punto

# Verificar las coordenadas transformadas
print("Transformed center:", transformed_center)

# Convertir las coordenadas transformadas a enteros
transformed_center = tuple(map(int, transformed_center))

print('Transformed center:', transformed_center)
cv2.circle(diagram_image, [50, 50], 5, color, -1)

# Dibujar el punto transformado en el diagrama
cv2.circle(diagram_image, transformed_center, 5, (0, 0, 255), -1)
cv2.circle(diagram_image, transformed_center, 5, (0, 0, 255), -1)

cv2.rectangle(diagram_image, (22,22), (642, 388), (0, 255, 0), 2)

# Puntos del rectángulo (marcar la posición del jugador)
x2, y2 = x1 + w, y1 + h
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde para el original

# Mostrar la imagen del video con el círculo y el rectángulo
cv2.imshow("Partido", image)

# Mostrar la imagen del diagrama con la posición transformada del jugador
cv2.imshow("Diagrama", diagram_image)

# Esperar hasta que se presione una tecla
cv2.waitKey(0)
cv2.destroyAllWindows()

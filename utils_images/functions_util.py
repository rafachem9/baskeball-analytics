import numpy as np


def get_center_point(player_coords):
    """
    Get the center point of the player
    :param player_coords:
    :return:
    """
    x1, y1, w, h = player_coords
    center_x = x1 + w // 2
    center_y = y1 + h // 2

    # Crear un punto para el centro del jugador
    return np.array([[[center_x, center_y]]], dtype=np.float32)  # Debe ser de forma (1, 1, 2)


def get_rectangule_coords(player_coords):
    """
    Get the rectangle coordinates
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    x1, y1, w, h = player_coords
    x2, y2 = x1 + w, y1 + h

    # Puntos del rectángulo
    return np.array([
        [[x1, y1]],  # Superior izquierda
        [[x2, y1]],  # Superior derecha
        [[x1, y2]],  # Inferior izquierda
        [[x2, y2]]  # Inferior derecha
    ], dtype=np.float32)

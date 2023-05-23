import cv2
import cv2.aruco as aruco
import numpy as np
import math

# Configuración del diccionario y los parámetros del detector
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
params = aruco.DetectorParameters_create()

# Definición de los puntos del cubo en 3D
obj_points = np.array([
    [-0.5, -0.5, 0],
    [-0.5, 0.5, 0],
    [0.5, 0.5, 0],
    [0.5, -0.5, 0],
    [-0.5, -0.5, 1],
    [-0.5, 0.5, 1],
    [0.5, 0.5, 1],
    [0.5, -0.5, 1]
])

# Definición de los puntos del cubo en 2D
img_points = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
])

# Creación de la matriz de la cámara
camera_matrix = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])

# Creación de la distorsión de la cámara
dist_coeffs = np.array([0, 0, 0, 0, 0])

# Carga del video de la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Bucle principal
while True:
    # Captura del frame de la cámara
    ret, frame = cap.read()

    # Conversión del frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección del marcador ArUco
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=params)

    # Si se encontró algún marcador
    if ids is not None:
        # Estimación de la pose del marcador
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 1, camera_matrix, dist_coeffs)

        # Proyección del cubo sobre el marcador
        for i in range(len(ids)):
            aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 1)
            img_points, _ = cv2.projectPoints(obj_points, rvec[i], tvec[i], camera_matrix, dist_coeffs)
            img_points = np.int32(img_points).reshape(-1, 2)
            cv2.drawContours(frame, [img_points[:4]], -1, (0, 255, 0), 3)
            for i, j in zip(range(4), range(4, 8)):
                cv2.line(frame, tuple(img_points[i]), tuple(img_points[j]), (0, 255, 0), 3)
            cv2.drawContours(frame, [img_points[4:]], -1)
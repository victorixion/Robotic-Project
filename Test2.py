import numpy as np
import cv2
import sys
import time
import math

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:

        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 00), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

            print("[Interference] ArUco maker ID: {}".format(markerID))

    return image

def getTranslationMatrix(tvec):
    T = np.identity(n=4)
    T[0:3, 3] = tvec
    return T

def getTransformMatrix(rvec, tvec):
    mat = getTranslationMatrix(tvec)
    mat[:3, :3] = cv2.Rodrigues(rvec)[0]
    return mat

def relativeTransformMatrix(rotation, translation):
    xC, xS = math.cos(rotation[0]), math.sin(rotation[0])
    yC, yS = math.cos(rotation[1]), math.sin(rotation[1])
    zC, zS = math.cos(rotation[2]), math.sin(rotation[2])
    dX = translation[0]
    dY = translation[1]
    dZ = translation[2]
    Translate_matrix = np.array([[1, 0, 0, dX],
                                 [0, 1, 0, dY],
                                 [0, 0, 1, dZ],
                                 [0, 0, 0, 1]])
    Rotate_X_matrix = np.array([[1, 0, 0, 0],
                                [0, xC, -xS, 0],
                                [0, xS, xC, 0],
                                [0, 0, 0, 1]])
    Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                                [0, 1, 0, 0],
                                [-yS, 0, yC, 0],
                                [0, 0, 0, 1]])
    Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                                [zS, zC, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    return np.dot(Rotate_Z_matrix, np.dot(Rotate_Y_matrix, np.dot(Rotate_X_matrix, Translate_matrix)))

def pose_estimation(frame, aruco_dict_type, maztrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters,
                                                                cameraMatrix=maztrix_coefficients,
                                                                distCoeff=distortion_coefficients)
    
 

    if corners:
                for ids, corners in zip(ids, corners):

                    cv2.polylines(
                        frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
                    )
                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)
                    top_right = corners[0].ravel()
                    top_left = corners[1].ravel()
                    bottom_right = corners[2].ravel()
                    bottom_left = corners[3].ravel()
                    
                    cv2.putText(
                        frame,
                        f"id: {ids[0]}",
                        top_right,
                        cv2.FONT_HERSHEY_PLAIN,
                        1.3,
                        (200, 100, 0),
                        2,
                        cv2.LINE_AA,
                    )  

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters,
                                                                cameraMatrix=maztrix_coefficients,
                                                                distCoeff=distortion_coefficients)
    
    if len(corners) > 0:
        for i in range(0, len(ids)):
            if 14 in ids:
                index = list(ids).index(14)
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[index], 0.08, maztrix_coefficients,
                                                                            distortion_coefficients)

                rvec = rvec[0][0]
                tvec = tvec[0][0]
                transformMatrix = getTransformMatrix(rvec, tvec)
                nueva = relativeTransformMatrix([0, 0, 0], [-0.04, -0.04, 0])
                transformMatrix = np.dot(transformMatrix, nueva)
                rmat = transformMatrix[:3, :3]
                tmat = transformMatrix[:3, 3:]

                cv2.aruco.drawAxis(frame, maztrix_coefficients, distortion_coefficients, rmat, tmat, 0.2)   
                                    
    return frame



aruco_type = "DICT_4X4_250"

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type  ])

arucoParams = cv2.aruco.DetectorParameters_create()

intrinsic_camera = np.array(((508.39421333, 0, 316.43609709), (0, 508.66148346, 237.33058), (0, 0, 1)))
distortion = np.array((2.25992394e-01, -1.13146106e+00, -9.51019994e-04, -7.87978482e-04, 1.54786562e+00))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)

    cv2.imshow('Estimated pose', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
#cv2.drestroyAllWindows()
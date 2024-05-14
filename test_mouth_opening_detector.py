# -*- coding: utf-8 -*-

import cv2
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks


def Create(outer_points,d_outer,inner_points,d_inner):
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cap = cv2.VideoCapture(0)

    while(True):
        ret, img = cap.read()
        rects = find_faces(img, face_model)
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            draw_marks(img, shape)
            cv2.putText(img, 'Press r to record Mouth distances', (30, 30), font,
                        1, (0, 255, 255), 2)
            cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            for i in range(100):
                for i, (p1, p2) in enumerate(outer_points):
                    d_outer[i] += shape[p2][1] - shape[p1][1]
                for i, (p1, p2) in enumerate(inner_points):
                    d_inner[i] += shape[p2][1] - shape[p1][1]
            break
    
    d_outer[:] = [x / 100 for x in d_outer]
    d_inner[:] = [x / 100 for x in d_inner]
        
    cap.release()
    cv2.destroyAllWindows()

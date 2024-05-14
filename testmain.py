import cv2
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks
import threading
import time

from test_eye_tracker import nothing, print_eye_pos,process_thresh,contouring,find_eyeball_position,eye_on_mask
from test_head_pose_estimation import get_2d_points,draw_annotation_box,head_pose_points
from test_person_ans_phone import load_darknet_weights,draw_outputs,DarknetConv,DarknetResidual,DarknetBlock,Darknet,YoloConv,YoloOutput,yolo_boxes,yolo_nms,YoloV3,weights_download
from test_mouth_opening_detector import Create

from audio_part import audio




def head_pose(face_model,landmark_model):

    cap = cv2.VideoCapture(0)

    outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
    d_outer = [0]*5
    inner_points = [[61, 67], [62, 66], [63, 65]]
    d_inner = [0]*3

    Create(outer_points,d_outer,inner_points,d_inner)

    ret, img = cap.read()
    thresh = img.copy()

    cv2.namedWindow('image')
    kernel = np.ones((9, 9), np.uint8)

    cv2.createTrackbar('threshold', 'image', 75, 255, nothing)
    
    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]
    

    
        
    # face_model = get_face_detector()
    # landmark_model = get_landmark_model()
    # cap = cv2.VideoCapture(0)
    # ret, img = cap.read()


    size = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    

    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416

    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


    yolo = YoloV3(yolo_anchors,yolo_anchor_masks)
    load_darknet_weights(yolo, 'models/yolov3.weights') 

    # cap = cv2.VideoCapture(0)


    while(True):
        ret, img = cap.read()
        if ret == False:
            break
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 320))
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)
        image = image / 255
        class_names = [c.strip() for c in open("models/classes.TXT").readlines()]
        boxes, scores, classes, nums = yolo(image)
        count=0
        for i in range(nums[0]):
            if int(classes[0][i] == 0):
                count +=1
            if int(classes[0][i] == 67):
                print('Mobile Phone detected')
        if count == 0:
            print('No person detected')
        elif count > 1: 
            print('More than one person detected')
            
        # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

        # cv2.imshow('Prediction', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    
    

        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],     # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            # for p in image_points:
            #     cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            # cv2.line(img, p1, p2, (0, 255, 255), 2)
            # cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
            # for (x, y) in marks:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90
                
            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1/m)))
            except:
                ang2 = 90
                
                # print('div by zero error')
            if ang1 >= 48:
                print('Head down')
                cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
            elif ang1 <= -48:
                print('Head up')
                cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
            
            if ang2 >= 48:
                print('Head right')
                cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
            elif ang2 <= -48:
                print('Head left')
                cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
            
            # cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            # cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
        # cv2.imshow('img', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break



        rects = faces
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            cnt_outer = 0
            cnt_inner = 0
            # draw_marks(img, shape[48:])
            for i, (p1, p2) in enumerate(outer_points):
                if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                    cnt_outer += 1 
            for i, (p1, p2) in enumerate(inner_points):
                if d_inner[i] + 2 <  shape[p2][1] - shape[p1][1]:
                    cnt_inner += 1
            if cnt_outer > 3 and cnt_inner > 2:
                print('Mouth open')
                # cv2.putText(img, 'Mouth open', (30, 30), font,1, (0, 255, 255), 2)
            # show the output image with the face detections + facial landmarks






        # ret, img = cap.read()
        # rects = find_faces(img, face_model)
        
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask, end_points_left = eye_on_mask(mask, left, shape)
            mask, end_points_right = eye_on_mask(mask, right, shape)
            mask = cv2.dilate(mask, kernel, 5)
            
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            mid = int((shape[42][0] + shape[39][0]) // 2)
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            threshold = cv2.getTrackbarPos('threshold', 'image')
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = process_thresh(thresh)
            
            eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
            eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
            print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
            # for (x, y) in shape[36:48]:
            #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
            
        # cv2.imshow('eyes', img)
        # cv2.imshow("image", thresh)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


face_model = get_face_detector()
landmark_model = get_landmark_model()
# ret, img = cap.read()
head_pose(face_model,landmark_model)
# eye_trak(face_model,landmark_model,cap,ret,img)

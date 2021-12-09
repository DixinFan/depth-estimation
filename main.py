import cv2
# Function for stereo vision and depth estimation
import triangulation as tri
import calibration
# Mediapipe for face detection
import mediapipe as mp


mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
# Open both cameras
cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap_left =  cv2.VideoCapture(1, cv2.CAP_DSHOW)
# Stereo vision setup parameters
frame_rate = 28    #Camera frame rate (maximum at 120 fps)
B = 10.5               #Distance between the cameras [cm]
f = 2.37              #Camera lense's focal length [mm]
alpha = 88        #Camera field of view in the horisontal plane [degrees]


def calibrate_depth(depth):
    # (51：19) (83：22)
    slope = 11
    bias = -159
    depth = slope * depth + bias
    depth = round(depth,  0)
    # ones_place = depth % 10
    # print('test')
    # print(depth)
    # print(ones_place)
    # if ones_place in range(1 , 3):
    #     depth =  depth - ones_place
    # elif ones_place in range(7, 10):
    #     depth =  depth - ones_place + 10
    # depth = round(depth,  -1)
    # print(depth)
    return depth

def caculate_height(depth, scale_h_left):
    """
    h/d     51      83
    180     0.31    0.41
    175     0.36    0.46
    170     0.46    0.52
    165     0.5     0.54
    """
    slope_1 = 0.003
    slope_2 = -0.01
    bias = 1.957
    height = (scale_h_left - slope_1*depth - bias) / slope_2
    return height


# Main program loop with face detector and depth estimation using stereo vision
with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    scale_h_left = 0
    while(cap_right.isOpened() and cap_left.isOpened()):
        succes_right, frame_right = cap_right.read()
        succes_left, frame_left = cap_left.read()
        # CALIBRATION
        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
        # If cannot catch any frame, break
        if not succes_right or not succes_left:                    
            break
        else:
            # Convert the BGR image to RGB
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
            # Process the image and find faces
            results_right = face_detection.process(frame_right)
            results_left = face_detection.process(frame_left)
            # Convert the RGB image to BGR
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
            # CALCULATING DEPTH
            center_right = 0
            center_left = 0
            if results_right.detections:
                for id, detection in enumerate(results_right.detections):
                    mp_draw.draw_detection(frame_right, detection)
                    bBox = detection.location_data.relative_bounding_box
                    h, w, c = frame_right.shape
                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                    center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
                    cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            if results_left.detections:
                for id, detection in enumerate(results_left.detections):
                    mp_draw.draw_detection(frame_left, detection)
                    bBox = detection.location_data.relative_bounding_box
                    h, w, c = frame_left.shape
                    # print(bBox)
                    # print(h)
                    # print('test')
                    # print(bBox.ymin)
                    # print(bBox.height)
                    # print(bBox.height / 2)
                    # scale_h = bBox.ymin + bBox.height / 2
                    # print(round(scale_h, 2))
                    # print(bBox.ymin + bBox.height / 2)
                    scale_h_left = bBox.ymin + bBox.height / 2
                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                    center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
                    # print(boundBox)
                    cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            # If no ball can be caught in one camera show text "TRACKING LOST"
            if not results_right.detections or not results_left.detections:
                cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            else:
                # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
                # All formulas used to find depth is in video presentaion
                # print(center_point_right)
                # print("陈")
                # print(center_point_left)
                depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)
                depth = calibrate_depth(depth)
                height = caculate_height(depth, scale_h_left)
                cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                cv2.putText(frame_left, "Height: " + str(round(height,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                # cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            # Show the frames
            cv2.imshow("frame right", frame_right) 
            cv2.imshow("frame left", frame_left)
            # Hit "q" to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
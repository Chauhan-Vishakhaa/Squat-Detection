import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (joint)
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return round(angle, 2)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract key joint coordinates
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate knee angle
            angle = calculate_angle(hip, knee, ankle)
            
            # Display knee angle on screen
            cv2.putText(image, f'Knee Angle: {angle}°', 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Provide more accurate feedback based on squat depth
            if angle > 170:
                feedback = "Stand Straight (170° - 180°)\n- Keep knees slightly bent.\n- Engage core.\n- Lower yourself slowly."
            elif 140 <= angle <= 170:
                feedback = "Go Lower (140° - 170°)\n- Push hips back.\n- Lower thighs parallel to floor.\n- Maintain a straight spine."
            elif 95 <= angle <= 140:
                feedback = "Good Squat (95° - 140°)\n- Knees aligned with toes.\n- Keep weight on heels.\n- Maintain neutral spine."
            elif 80 <= angle < 95:
                feedback = "Deep Squat (80° - 95°)\n- Ensure knee stability.\n- Maintain strong posture.\n- Only go deep if mobility allows."
            elif angle < 80:
                feedback = "Too Low (Below 80°)\n- Avoid excessive knee stress.\n- Check for balance issues.\n- Limit depth if uncomfortable."
            else:
                feedback = "Adjust Position\n- Check form.\n- Keep spine neutral.\n- Balance weight evenly."
            
            # Display feedback on screen
            y_offset = 100
            for i, line in enumerate(feedback.split("\n")):
                cv2.putText(image, line, (50, y_offset + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Squat Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

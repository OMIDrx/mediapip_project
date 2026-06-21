import cv2
import mediapipe as mp
import math

# راه‌اندازی Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

draw_util = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# محاسبه فاصله بین دو نقطه
def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    emotion = "Normal"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            # ---------- تشخیص لبخند ----------
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]

            mouth_width = distance(left_mouth, right_mouth)
            mouth_height = distance(upper_lip, lower_lip)

            smile_ratio = mouth_width / mouth_height

            # ---------- تشخیص اخم ----------
            left_eyebrow = landmarks[70]
            right_eyebrow = landmarks[300]
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            eyebrow_distance = distance(left_eyebrow, left_eye) + \
                               distance(right_eyebrow, right_eye)

            # ---------- تشخیص تعجب ----------
            eye_top = landmarks[159]
            eye_bottom = landmarks[145]

            eye_open = distance(eye_top, eye_bottom)

            # ---------- تصمیم گیری ----------
            if smile_ratio > 4.5:
                emotion = "Happy :)"

            elif eyebrow_distance < 0.09:
                emotion = "Angry >:("

            elif eye_open > 0.03 and mouth_height > 0.03:
                emotion = "Surprised :O"

            else:
                emotion = "Normal :|"

            # رسم نقاط صورت
            draw_util.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=draw_util.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1,
                    circle_radius=1
                )
            )

    # نمایش احساس
    cv2.putText(frame, emotion, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 255), 3)

    cv2.imshow("Face Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
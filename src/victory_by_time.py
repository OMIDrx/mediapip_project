import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands.Hands(max_num_hands=2)
draw_util = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

countdown_start = False
start_time = 0
countdown_number = 3

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(frameRGB)

    victory = False
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            landmark = hand.landmark
            if (landmark[8].y < landmark[6].y) and (landmark[12].y < landmark[10].y):
                victory = True
                draw_util.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

    if victory and not countdown_start:
        countdown_start = True
        start_time = time.time()

    if countdown_start:
        x = int(time.time() - start_time)
        if x < 5:
            cv2.putText(frame, str(5 - x), (250, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
        else:
            cv2.putText(frame, "done!", (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),3)  

    cv2.imshow('WebCam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
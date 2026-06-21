import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands.Hands(max_num_hands=2)
draw_util = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_count = 0

while True:
    sucsess , frame = cap.read()
    
    if not sucsess:
        break
    
    finger_count = 0

    
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = mp_hands.process(frameRGB)
    
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            
            #draw_util.draw_landmarks(frame,hand,mp.solutions.hands.HAND_CONNECTIONS)
            landmark = hand.landmark
            finger_tip =[8,12,16,20]
            for tip in finger_tip:
                if landmark[tip].y < landmark [tip-2].y:
                    finger_count +=1
            if landmark[12].y < landmark[10].y and landmark[8].y < landmark[6].y :
                filename = './src/images/test1.png'
                cv2.imwrite(filename,frame)
                cv2.putText(frame,str('VICTORY!'),(50,200),
                cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),3)  
            
                    
            
    '''       
    if finger_count == 5:
        cv2.putText(frame,str('OK'),(50,200),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,3,(0,255,0),3)
    if finger_count == 0:
        cv2.putText(frame,str('Not Ok'),(50,200),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,3,(0,255,0),3)
    '''
            
        
    
    cv2.imshow('WebCame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()
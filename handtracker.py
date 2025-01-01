import cv2
import mediapipe as mp
import time

# Initialize the hand tracking module
cap = cv2.VideoCapture(0)

# Create hands object
mpHands = mp.solutions.hands
hands = mpHands.Hands()

ptime = 0 
ctime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert image to RGB
    results = hands.process(imgRGB) # Process the image
    mpDraw = mp.solutions.drawing_utils # Draw the hand landmarks
    
    # Check if there are any hands detected
    if results.multi_hand_landmarks: 
        # For each hand detected, draw the landmarks
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)  
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1 / (cTime - ptime)
    ptime = cTime
    
    img = cv2.flip(img, 1)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow("Image", img)
    
    # Quit loop when the 'esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
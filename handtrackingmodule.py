import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComp=1, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        
    # Find the hands in the image and returns the image with the landmarks drawn
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert image to RGB
        self.results = self.hands.process(imgRGB) # Process the image
        
        # Check if there are any hands detected
        if self.results.multi_hand_landmarks: 
            # For each hand detected, draw the landmarks
            for handLms in self.results.multi_hand_landmarks:
                if draw: 
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                
        return img
            
            
    #Finds the position of the landmarks on a hand, returns a list of the landmarks with their x and y coordinates
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        
        # Check if there are any hands detected
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        return lmList
    
    
# Capture the video feed from the webcam and display the landmarks         
def capture():
    ptime = 0 
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    while True:
        success, img = cap.read()  
        
        if not success: 
            print("Ignoring empty camera frame.")
            continue
        
        img = detector.findHands(img)
        img = cv2.flip(img, 1)
        
        cTime = time.time()
        fps = 1 / (cTime - ptime)
        ptime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        
        # Quit loop when esc key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture()
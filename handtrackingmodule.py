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
        
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert image to RGB
        results = self.hands.process(imgRGB) # Process the image
        
        # Check if there are any hands detected
        if results.multi_hand_landmarks: 
            # For each hand detected, draw the landmarks
            for handLms in results.multi_hand_landmarks:
                if draw: 
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                
                #for id, lm in enumerate(handLms.landmark):
                #    h, w, c = img.shape
                #    cx, cy = int(lm.x * w), int(lm.y * h)
                #    print(id, cx, cy)
                
        return img
                        
        
def capture():
    ptime = 0 
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    while True:
        success, img = cap.read()  
        img = detector.findHands(img, success)
        
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
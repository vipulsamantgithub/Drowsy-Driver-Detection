import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from model import *
import numpy as np
import tensorflow as tf

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    highlight=255;
    shadow=0;
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
            
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
        
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
            
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
    
color = "rgb"
bins = 16
resizeWidth = 0
# Initialize plot.
fig, ax = plt.subplots()
if color == 'rgb':
    ax.set_title('Histogram (RGB)')
else:
    ax.set_title('Histogram (grayscale)')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Frequency')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 6
alpha = 0.5
if color == 'rgb':
    lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha)
    lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha)
    lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha)
    lr, = ax.plot(np.arange(bins), np.zeros((bins,)), c='#654321', lw=lw, alpha=alpha)
    lbr, = ax.plot(np.arange(bins), np.zeros((bins,)), c='#123456', lw=lw, alpha=alpha)
    lg, = ax.plot(np.arange(bins), np.zeros((bins,)), c='#225522', lw=lw, alpha=alpha)
else:
    lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)
ax.set_xlim(0, bins-1)
ax.set_ylim(0, 1)
plt.ion()
#plt.show()
bi=None
gi=None
ri=None

    
#harcascade file to detect Face
facec = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

#loading emaotion Detection Model
model = DowsinessDetectionModel("models/eyes.json", "models/Eyes.h5")
font = cv2.FONT_HERSHEY_DUPLEX


flag=0;
demoimg=storage=0;
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    #image=cv2.resize(image,(int(image.shape[0]*1.5),int(image.shape[1]*2)))
    numPixels = np.prod(image.shape[:2])
    demoimg= np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    

    
    gray_fr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img=apply_brightness_contrast(gray_fr, 64, 64)#for gray image
    faces = facec.detectMultiScale(img, 1.3, 5)
    count=0;
    flag=1
    
    
    for (x, y, w, h) in faces:
            fc = img[y-10:y+h+10, x-10:x+w+10]
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_color = image[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(fc)
            state=[]
            for (ex,ey,ew,eh) in eyes:
                eye_detect=fc[ey:ey+eh, ex:ex+ew]
                roi=cv2.resize(eye_detect, (86, 86))
                pred = model.predict_state(roi[np.newaxis, :, :])
                state.append(pred)
                cv2.rectangle(image,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)
            print(state)
            if len(state)!=2:
                state="Eyes closed(100%)"
            else:
                if state[0][0]==state[1][0]:
                    state=state[0][0]+"("+str((state[0][1]+state[1][1])/2)
                else:
                    state="Eyes Open("+str((state[0][1]+state[1][1])/2)+")"
            if flag:
                cv2.putText(demoimg, "state: "+str(state), (x+10, y+w+10),
                            font, 1*(abs(w)/240), (255, 255, 110), 1)
                
                
                
                cv2.rectangle(demoimg,(x,y),(x+w+10,y+h+40),(255,0,0),2)
                flag=0;
            else:
                cv2.putText(demoimg, "state: "+str(state), (x+10, y-35),
                            font, 1*(abs(w-x)/240), (255, 255, 110), 1)
                
                
                
                cv2.rectangle(demoimg,(x-10,y-20),(x+w+10,y+h+40),(255,0,0),1)
                flag=1;
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    
            
    (b, g, r) = cv2.split(image)
    
    if bi==0 or gi==0 or ri==0:
        bi=b
        gi=g
        ri=r
        histogramRi = cv2.calcHist([ri], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramGi = cv2.calcHist([gi], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramBi = cv2.calcHist([bi], [0], None, [bins], [0, 255]) / (numPixels/4)
        lr.set_ydata(histogramRi)
        lg.set_ydata(histogramGi)
        lbr.set_ydata(histogramBi)
    else:
        histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramRi = cv2.calcHist([ri], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramGi = cv2.calcHist([gi], [0], None, [bins], [0, 255]) / (numPixels/4)
        histogramBi = cv2.calcHist([bi], [0], None, [bins], [0, 255]) / (numPixels/4)
        lineR.set_ydata(histogramR)
        lineG.set_ydata(histogramG)
        lineB.set_ydata(histogramB)
        fig.canvas.draw()
        lr.set_ydata(histogramRi)
        lg.set_ydata(histogramGi)
        lbr.set_ydata(histogramBi)
    fig.canvas.draw()
    fig.savefig('plot.jpg', bbox_inches='tight', dpi=150)
    plotteddata=cv2.resize(cv2.imread("plot.jpg"),(image.shape[1],image.shape[0]))
    halfFrame=np.concatenate((demoimg,plotteddata),axis=0)
    halfFrame=cv2.resize(halfFrame,(halfFrame.shape[1]//2,halfFrame.shape[0]//2))
    #cv2.imshow("halfFrame",halfFrame)
    fullFrame=np.column_stack((image,halfFrame))
    cv2.imshow("Data Visualization",fullFrame);
    if cv2.waitKey(1) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()


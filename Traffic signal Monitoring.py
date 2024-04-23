import numpy as np
import argparse
import imutils
import time
import cv2
import os

confidence_TH=0.5
threshold_TH=0.1
labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
weightsPath = os.path.sep.join(['yolo-coco', "yolov3.weights"])
configPath = os.path.sep.join(['yolo-coco', "yolov3.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i- 1] for i in net.getUnconnectedOutLayers()]

def CALC_VEHC(frame1,net,COLORS):
    xds=np.shape(frame1)
    print('FRMA',xds)
    (H, W) = frame1.shape[:2]
    blob = cv2.dnn.blobFromImage(frame1, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_TH:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences,confidence_TH,threshold_TH)
    LV=0;HV=0;BK=0;BS=0
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(LABELS[classIDs[i]])
            cv2.putText(frame1, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if text=='car':
                LV=LV+1
            if text=='person':
                BK=BK+1
            if text=='motorbike':
                BK=BK+1
            if text=='bus':
                BS=BS+1
            if text=='truck':
                HV=HV+1
        TV=2*LV+BK+4*HV;
        print('COUNT:',LV+BK+HV,'\n')
        cv2.putText(frame1, 'SIGNAL STATUS:OFF', (0,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,[1,1,1],2)
        cv2.putText(frame1, 'EMERGENCY VEHICLE:'+str(BS), (0,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,[1,0,1],2)
        cv2.putText(frame1, 'LIGHT VEHICLE:'+str(LV), (0,90),cv2.FONT_HERSHEY_SIMPLEX,0.5,[1,0,1],2)
        cv2.putText(frame1, 'HEAVY VEHICLE:'+str(HV), (0,120),cv2.FONT_HERSHEY_SIMPLEX,0.5,[1,0,1],2)
        cv2.putText(frame1, 'BIKE:'+str(BK), (0,150),cv2.FONT_HERSHEY_SIMPLEX,0.5,[1,0,1],2)
        return frame1,TV,BS


RD=cv2.imread('A.png');
RD=cv2.resize(RD,(1000,1000))

Vidname1='H.mp4'
vs1=cv2.VideoCapture(Vidname1)
(grabbed, frame1) = vs1.read()
frame1,TV1,BS1=CALC_VEHC(frame1,net,COLORS)
print('TOTAL VEHICLE at Road1:','\n')

Vidname2='E.mp4'
vs2=cv2.VideoCapture(Vidname2)
(grabbed, frame2) = vs2.read()
frame2,TV2,BS2=CALC_VEHC(frame2,net,COLORS)
print('TOTAL VEHICLE at Road2:','\n')

Vidname3='G.mp4'
vs3=cv2.VideoCapture(Vidname3)
(grabbed, frame3) = vs3.read()
frame3,TV3,BS3=CALC_VEHC(frame3,net,COLORS)
print('TOTAL VEHICLE at Road3:','\n')

Vidname4='F.mp4'
vs4=cv2.VideoCapture(Vidname4)
(grabbed, frame4) = vs4.read()
frame4,TV4,BS4=CALC_VEHC(frame4,net,COLORS)
print('TOTAL VEHICLE at Road4:','\n')


print(TV1,TV2,TV3,TV4)
TV=[TV1,TV2,TV3,TV4];
sort_index = np.argsort(TV) 
print('ADJUSTING SIGNAL ORDER according to Vehicle count',sort_index)


x = range(1,5)
if BS1>=1:
    for n in x:
        (grabbed, frame1) = vs1.read()
        frame1,TV1,BS1=CALC_VEHC(frame1,net,COLORS)
        RD[0:500,0:500]=cv2.resize(frame1,(500,500));
        RD[500:5000,0:500]=cv2.resize(frame2,(500,500));
        RD[0:500,500:5000]=cv2.resize(frame3,(500,500));
        RD[500:5000,500:5000]=cv2.resize(frame4,(500,500));
        print('EMERGENCY VEHICLE FOUND at road 1\n')
        cv2.imshow('',RD);
        cv2.waitKey(0);cv2.destroyAllWindows()
        
        
elif BS2>=1:
    for n in x:
        (grabbed, frame2) = vs2.read()
        frame2,TV1,BS1=CALC_VEHC(frame2,net,COLORS)
        RD[0:500,0:500]=cv2.resize(frame1,(500,500));
        RD[500:5000,0:500]=cv2.resize(frame2,(500,500));
        RD[0:500,500:5000]=cv2.resize(frame3,(500,500));
        RD[500:5000,500:5000]=cv2.resize(frame4,(500,500));
        print('EMERGENCY VEHICLE FOUND at road 2\n')
        cv2.imshow('',RD);
        cv2.waitKey(0);cv2.destroyAllWindows()
        
elif BS3>=1:
    for n in x:
        (grabbed, frame3) = vs3.read()
        frame3,TV1,BS1=CALC_VEHC(frame3,net,COLORS)
        RD[0:500,0:500]=cv2.resize(frame1,(500,500));
        RD[500:5000,0:500]=cv2.resize(frame2,(500,500));
        RD[0:500,500:5000]=cv2.resize(frame3,(500,500));
        RD[500:5000,500:5000]=cv2.resize(frame4,(500,500));
        print('EMERGENCY VEHICLE FOUND at road 3\n')
        cv2.imshow('',RD);
        cv2.waitKey(0);cv2.destroyAllWindows()

elif BS4>=1:
    for n in x:
        (grabbed, frame4) = vs4.read()
        frame4,TV1,BS1=CALC_VEHC(frame4,net,COLORS)
        RD[0:500,0:500]=cv2.resize(frame1,(500,500));
        RD[500:5000,0:500]=cv2.resize(frame2,(500,500));
        RD[0:500,500:5000]=cv2.resize(frame3,(500,500));
        RD[500:5000,500:5000]=cv2.resize(frame4,(500,500));
        print('EMERGENCY VEHICLE FOUND at road 4\n')
        cv2.imshow('',RD);
        cv2.waitKey(0);cv2.destroyAllWindows()

else:
    print('NO EMERGENCY VEHICLE FOUND \n')
    for n in range(1,4):
        ttp=sort_index[n];
        print(ttp)
        print('----------------------------------------\n')
        if ttp==0:
            x = range(1,TV1)
            print('SIGNAL 1 WILL ON For ',TV1*10,'Seconds \n')
            for n in x:
                (grabbed, frame1) = vs1.read()
                frame1,TV1,BS1=CALC_VEHC(frame1,net,COLORS)
                RD[0:500,0:500]=cv2.resize(frame1,(500,500));
                RD[500:5000,0:500]=cv2.resize(frame2,(500,500));
                RD[0:500,500:5000]=cv2.resize(frame3,(500,500));
                RD[500:5000,500:5000]=cv2.resize(frame4,(500,500));
                cv2.imshow('',RD);
                cv2.waitKey(1000);#cv2.destroyAllWindows()
        elif ttp==1:
            x = range(1,TV2)
            print('SIGNAL 1 WILL ON For ',TV2*10,'Seconds \n')
            for n in x:
                (grabbed, frame2) = vs2.read()
                frame2,TV1,BS1=CALC_VEHC(frame2,net,COLORS)
                RD[0:500,0:500]=cv2.resize(frame1,(500,500));
                RD[500:5000,0:500]=cv2.resize(frame2,(500,500));
                RD[0:500,500:5000]=cv2.resize(frame3,(500,500));
                RD[500:5000,500:5000]=cv2.resize(frame4,(500,500));
                cv2.imshow('',RD);
                cv2.waitKey(1000);#cv2.destroyAllWindows()
        elif ttp==2:
            x = range(1,TV3)
            print('SIGNAL 1 WILL ON For ',TV3*10,'Seconds \n')
            for n in x:
                (grabbed, frame3) = vs3.read()
                frame3,TV1,BS1=CALC_VEHC(frame3,net,COLORS)
                RD[0:500,0:500]=cv2.resize(frame1,(500,500));
                RD[500:5000,0:500]=cv2.resize(frame2,(500,500));
                RD[0:500,500:5000]=cv2.resize(frame3,(500,500));
                RD[500:5000,500:5000]=cv2.resize(frame4,(500,500));
                cv2.imshow('',RD);
                cv2.waitKey(1000);#cv2.destroyAllWindows()
        elif ttp==3:
            x = range(1,TV4)
            print('SIGNAL 1 WILL ON For ',TV4*10,'Seconds \n')
            for n in x:
                (grabbed, frame4) = vs4.read()
                frame4,TV1,BS1=CALC_VEHC(frame4,net,COLORS)
                RD[0:500,0:500]=cv2.resize(frame1,(500,500));
                RD[500:5000,0:500]=cv2.resize(frame2,(500,500));
                RD[0:500,500:5000]=cv2.resize(frame3,(500,500));
                RD[500:5000,500:5000]=cv2.resize(frame4,(500,500));
                cv2.imshow('',RD);
                cv2.waitKey(1000);#cv2.destroyAllWindows()
            
            
        

#frame1,TV1,BS1=CALC_VEHC(frame1,net,COLORS)

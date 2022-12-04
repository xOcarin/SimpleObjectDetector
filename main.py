import cv2

#img = cv2.imread('lena.png')

thres = 0.5 #threshold to detect objects

cap = cv2.VideoCapture(1) #number changes which camera is being used. Mine is 0 for switch capture, 1 for webcam.
cap.set(3, 1920)
cap.set(4, 1080)

#goofy ahh array
classNames= []
classFile = 'coco.names lmaooooooooooooooooooooooooooooo'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#adsjfasdjfkasjdfa

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(),bbox):
            if classId != 1:
                cv2.rectangle(img, box, color=(0,225,0), thickness = 2)
                cv2.putText(img, classNames[classId - 1].upper(),(box[0]+10, box[0]+30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.putText(img, str(round(confidence * 100,2)), (box[0] + 200, box[0] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)





    cv2.imshow("Output", img)
    cv2.waitKey(1)
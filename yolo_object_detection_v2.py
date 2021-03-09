import cv2
import numpy as np

# Load Yolo
#net = cv2.dnn.readNet("yolov3_EmrataGal.cfg", "yolov3_EmrataGal_last.weights")
#net = cv2.dnn.readNet("weights\\yolov3\\yolov3.cfg", "weights\\yolov3\\yolov3.weights")
net = cv2.dnn.readNet("weights\\yolov3_Faces\\yolov3_custom.cfg", "weights\\yolov3_Faces\\yolov3_custom_final.weights")
classes = []
with open("weights\\yolov3_Faces\\classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
#img = cv2.imread("D:\\Wallpaper\\tom-cruise-as-ethan-hunt-in-mission-impossible-fallout-2018-l2-3840x2400.jpg")
#img = cv2.imread("D:\\Train_Images\\Celebrities\\Kendall Jenner\\3c12411-01.jpeg")
#img = cv2.resize(img, None, fx=0.8, fy=0.8)
#height, width, channels = img.shape

#Loading videos
cap = cv2.VideoCapture('D:\\Train_Images\\Celebrities\\Jennifer Lopez\\JLo.mp4')

while(True):
    ret, img = cap.read()
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height,width, channel = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.15:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3]* height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)



    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,label + " " + confidence, (x,y+400),font,2,color,2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
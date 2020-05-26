import numpy
import cv2
import time


net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layernames=net.getLayerNames()
output_layers=[layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = numpy.random.uniform(0, 255 , size=(len(classes), 3))

#video capturing
cap = cv2.VideoCapture(0)
start_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape

    #detecting object-blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0))

    #in case to see th blobs
    # for b in blob:
    #     for n, imgBlob in enumerate(b):
    #         cv2.imshow(str(n), imgBlob)

    #detecting object-extracting info
    net.setInput(blob)
    outs = net.forward(output_layers)

    #declaring empty arrays to store details of detected objects
    confidences = []
    class_ids = []
    boxes =[]
    #show-on-screen
    for out in outs:
        for detection in out:
            scores = detection[5:]
            #assigning predominant class
            class_id = numpy.argmax(scores)
            #extracting confidence
            confidence = scores[class_id]
            if confidence>0.2:
                #object found
                centre_x = int(detection[0]*width)
                centre_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #rectangle cordinates

                x = int(centre_x-w/2)
                y = int(centre_y-h/2)

                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                class_ids.append(class_id)

    n = len(boxes)
    #to prevent repetitive boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_COMPLEX
    for i in range(n):
        if i in indexes:
            x, y, w, h = boxes[i]
            item_name = str(classes[class_ids[i]])
            color = colors[i]
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, item_name + " " + str(round(confidence, 2)), (x, y+30), font, 0.75, color, 3)

    elapsed = time.time() - start_time
    fps = elapsed/frame_id
    cv2.putText(frame, "fps:" + str(fps), (10, 10), font, 0.5, (0, 0, 0), 2)
    cv2.imshow("image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()






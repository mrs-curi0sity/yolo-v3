##https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

## yolo-coco/ : The YOLOv3 object detector pre-trained (on the COCO dataset) model files. 
# These were trained by the Darknet team.

## to run script
# $ python3 yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

import numpy as np 
import argparse
import time
import cv2
import os


def main():
    print('[INFO] starting yolo')

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='path to the input image' )
    ap.add_argument('-y', '--yolo', required=True, help='base path to YOLO directory')
    ap.add_argument('-c', '--confidence', type=float, default=0.5, help='min probability to filter weak positive detections')
    ap.add_argument('-t', '--threshold', type=float, default=0.3, help = 'thresh wen applying non-maxima suppression')
    args = vars(ap.parse_args())

    labelsPath = os.path.sep.join([args['yolo'], 'coco.names'])
    print(f'labelsPath: {labelsPath}')
    LABELS = open(labelsPath).read().strip().split('\n')
    print(f'some labels: {LABELS[0:10]}')

    # initialize a list of colors
    np.random.seed(99)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    #get paths for weights and config
    weightsPath = os.path.sep.join([args['yolo'], 'yolov3.weights'])
    configPath = os.path.sep.join([args['yolo'], 'yolov3.cfg'])

    # load YOLO object detector
    print('\n [INFO] loading YOLO object from disc')
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load input image to predict
    image = cv2.imread(args['image'])
    (H,W) = image.shape[:2]

    # determine output layer names
    ln = net.getLayerNames()
    ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

    # construct a blob of input image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB = True, crop = False)

    # forward pass
    net.setInput(blob)
    start = time.time()

    # hier ist jetzt der output enthalten
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing info
    print(f'[INFO] Yolo took {(end -start):.3f} seconds')

    # now visualize object detection
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:

            # why ab 5?
            # weil die ersten 4 Elemente die Bounding Box definieren
            boxes, confidences, classIDs = extractDetection(detection, args, W, H, boxes, confidences, classIDs)

    print(f'\nboxes: {boxes}')
    print(f'\nconfidences: {confidences}')
    print(f'\nclassIDs: {classIDs}')
    print(f'\nLabels: {[LABELS[classID] for classID in classIDs]}')



    # nms =  non-maxima suppression to suppress weak, overlapping bounding boxes
    # also ensures non-redundancy
    idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.5, 0.5)

    # now draw the bounding boxes
    print(f'we kept {len(idxs)} of {len(classIDs)} many inputs')
    print(type(idxs))


    paintBoxes(idxs, classIDs, LABELS, confidences, boxes, COLORS, image)

    # now show the image
    cv2.imshow('predicted objects', image)
    to_path = 'output/' + args['image'].split('/')[1]
    print(f'[INFO] writing file to {to_path}')
    cv2.imwrite(to_path, image)
    cv2.waitKey(0)




def paintBoxes(idxs, classIDs, LABELS, confidences, boxes, COLORS, image):
    if len(idxs) == 0:
        print('Sorry, could not detect anyting')
    else:
        flat_indices = idxs.flatten()
        for index in flat_indices:
            # extract confidence and label
            classID = classIDs[index]
            label = LABELS[classID]
            confidence = confidences[index]

            # extract coordinates
            box = boxes[index]
            (x, y, w, h) = box

            # draw bounding box
            color = [int(c) for c in COLORS[classID]]

            # what does the 2 mean?
            cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)
            text = "{}: {:.4f}".format(label, confidence)
            cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def extractDetection(detection, args, W, H, boxes, confidences, classIDs):
    # why ab 5?
    # weil die ersten 4 Elemente die Bounding Box definieren
    scores = detection[5:]
    classID = np.argmax(scores)
    confidence = scores[classID]

    if confidence > args['confidence']:
        # YOLO returns center of Bounding Box + X/y coord
        # recalculate proportion => abs
        box = detection[0:4] * np.array([W,H, W, H])
        (centerX, centerY, width, height) = box.astype(int)

        x = int(centerX - (width/2))
        y = int(centerY - (height/2))

        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        classIDs.append(classID)
    return boxes, confidences, classIDs


if __name__ == '__main__':
    main()
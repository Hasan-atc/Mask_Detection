import cv2
import numpy as np

vid = cv2.VideoCapture(0)  

while True:
    x, frame = vid.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (960, 720))

    en = frame.shape[1]
    boy = frame.shape[0]
    frame_blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    classes = ["Mask", "NONE", "No Mask"]

    colors = ["0,255,0", "0,255, 255", "255, 0, 255"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)

    model = cv2.dnn.readNetFromDarknet("yolov3_mask.cfg", "yolov3_mask.weights")
    layers = model.getLayerNames()
    output_layers = [layers[layer[0] - 1] for layer in model.getUnconnectedOutLayers()]

    model.setInput(frame_blob)
    detects_layers = model.forward(output_layers)

    """NON MAXIMUM SUPPRESSION"""
    ids_list = []
    boxes_list = []
    confidence_list = []
    """     END OF OPERATION  """

    for detect_layer in detects_layers:
        for o_detection in detect_layer:

            scores = o_detection[5:]
            p_id = np.argmax(scores)
            confidence = scores[p_id]

            if confidence > 0.90:
                label = classes[p_id]
                box = o_detection[0:4] * np.array([en, boy, en, boy])
                (box_center_x, box_center_y, box_en, box_boy) = box.astype("int")

                start_x = int(box_center_x - (box_en / 2))
                start_y = int(box_center_y - (box_boy / 2))

                """ SUPRESSION 2 """
                ids_list.append(p_id)
                confidence_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_en), int(box_boy)])
                """ END OF 2"""

                """ SUPRESSION 3 """

                max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.5, 0.4)

                for max_id in max_ids:
                    max_class_id = max_id[0]
                    box = boxes_list[max_class_id]

                    start_x = box[0]
                    start_y = box[1]
                    box_en = box[2]
                    box_boy = box[3]

                    p_id = ids_list[max_class_id]
                    label = classes[p_id]
                    confidence = confidence_list[max_class_id]
                """ END OF 3"""

                end_x = start_x + box_en
                end_y = start_y + box_boy

                b_color = colors[p_id]
                b_color = [int(kadr) for kadr in b_color]

                label = "{}: {:.2f}%".format(label, confidence * 100)
                print("Oran {}".format(label))

                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), b_color, 2)
                cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 2)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

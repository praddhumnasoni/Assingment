import torch
import cv2
import itertools
import time


def distancing(coords):
    centers=[]
    for i in coords:
        centers.append(((int(i[2]) + int(i[0])) // 2, (int(i[3]) + int(i[1])) // 2))
    x_combs = list(itertools.combinations(coords, 2))
    radius=10
    thickness=5

    for x in x_combs:
        xyxy1, xyxy2 = x[0], x[1]
        cntr1 = ((int(xyxy1[2]) + int(xyxy1[0])) // 2, (int(xyxy1[3]) + int(xyxy1[1])) // 2)
        cntr2 = ((int(xyxy2[2]) + int(xyxy2[0])) // 2, (int(xyxy2[3]) + int(xyxy2[1])) // 2)
        dist = ((cntr2[0] - cntr1[0]) ** 2 + (cntr2[1] - cntr1[1]) ** 2) ** 0.5
        color = (0, 255, 255)
        cv2.line(img, cntr1, cntr2, color, thickness)
        mid = ((int(cntr1[0]) + int(cntr2[0])) // 2, (int(cntr1[1]) + int(cntr2[1])) // 2)
        cm = round(dist) / 100
        cv2.putText(img, str(cm) + " cm", mid, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)





trained_model_path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=trained_model_path, force_reload=True)

def detectObject(image_path,name):
    img=cv2.imread(image_path)
    while True:
        if img is None:
            break
        else:
            results = model(img)
            labels = results.xyxyn[0][:, -1].cpu().numpy()
            cord = results.xyxyn[0][:, :-1].cpu().numpy()
            n = len(labels)
            coords = []
            x_shape, y_shape = img.shape[1], img.shape[0]
            for i in range(n):
                row = cord[i]
                if row[4] < 0.4:
                    continue
                x1 = int(row[0] * x_shape)
                y1 = int(row[1] * y_shape)
                x2 = int(row[2] * x_shape)
                y2 = int(row[3] * y_shape)
                coords.append(row)
                bgr = (0, 255, 0)  # color of the box
                classes = model.names  # Get the name of label index
                label_font = cv2.FONT_HERSHEY_COMPLEX  # Font for the label.
                cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)  # Plot the boxes
                cv2.putText(img, classes[int(labels[i])], (x1, y1), label_font, 2, bgr, 2)

        distancing(coords)
        img = cv2.resize(img, (500, 500))
        cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    detectObject("D:/HappyMonk.ai/DataSet/Training_Happy/0001040.jpg", "ok")


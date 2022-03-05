from facenet_pytorch import InceptionResnetV1, MTCNN
from MyMTCNN import MyMTCNN
import torch
import os
from scipy.spatial import distance
import numpy as np
import cv2 as cv
import joblib
from ImageTransform import ImageTransform
from utils import*

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# mtcnn model
mtcnn = MyMTCNN(image_size=160, margin=20, keep_all=False, device=device)

# InceptionResnet module
Restnet = InceptionResnetV1(classify=False, pretrained='vggface2')
Restnet.to(device).eval()

#SVM model
SVM_face = joblib.load('./Module/face_recognition.joblib')

cap = cv.VideoCapture(0)
while cap.isOpened():
    since = cv.getTickCount()
    ret, frame = cap.read()
    if not ret:
        continue

    # crop face
    crop_faces = mtcnn(frame)
    transform = ImageTransform()

    if crop_faces[0] is not None:
        tran_crop_face = transform(cv2pil(crop_faces[0]))

        # creat embed
        with torch.no_grad():
            crop_embed = Restnet(tran_crop_face.to(device).unsqueeze(0)).squeeze(0)
        np_crop_embed = crop_embed.numpy()

        # SVM predict
        SVM_model = joblib.load('./Module/face_recognition.joblib')
        pre = SVM_model.predict(np.array([np_crop_embed]))
        pre_acc = SVM_model.predict_proba(np.array([np_crop_embed]))

        list_name = os.listdir('./data')
        pre_name = list_name[pre[0]]
        accuracy = max(pre_acc[0])

        # # mapping with faces embedding in facelist
        # list_sample_face, list_username = load_data_face()
        # distance_matrix = np.zeros_like(list_username)
        #
        # for i in range(list_sample_face.shape[0]):
        #     distance_matrix[i] = distance.euclidean(list_sample_face[i], np_crop_embed)
        # pre = list_username[np.argmin(distance_matrix)]

    bbox, prof, points = mtcnn.detect(frame, landmarks=True)
    if bbox is not None:
        for idx, (box, point) in enumerate(zip(bbox, points)):
            cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
            cv.putText(frame, '{} ac:{:.2f}'.format(pre_name, accuracy), (int(box[0]), int(box[1])-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

    fps = cv.getTickFrequency()/(cv.getTickCount()-since)
    cv.putText(frame, str(int(fps)), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv.imshow('anh', frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()






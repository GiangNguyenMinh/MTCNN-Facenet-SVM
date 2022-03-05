import torch
import glob
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from ImageTransform import ImageTransform
import argparse

def main_run(args):
    # create a dictionary: {label: [path to image]
    list_dirt = {}
    root_path = './data'
    for middle_path in os.listdir(root_path):
        list_dirt[middle_path] = glob.glob(os.path.join(root_path, middle_path, '*.jpg'))

    # Init module
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = InceptionResnetV1(classify=False, pretrained='vggface2')
    model.to(device).eval()

    # Create array of labels and emb
    labels = []
    embedding = []

    for label in list_dirt:
        for path in list_dirt[label]:
            img = Image.open(path)
            img_transform = ImageTransform()

            # input of model include 4 dimension
            # embed has 1 dimension
            with torch.no_grad():
                embed = model(img_transform(img).to(device).unsqueeze(0)).squeeze(0)
            n_embed = embed.numpy()
            embedding.append(n_embed)
            labels.append(label)

        # embedding_mean = torch.cat(embedding_label).mean(0, keepdim=True)
        # np_embedding_mean = embedding_mean.numpy()
        # embedding.append(np_embedding_mean)
        # labels.append(label)

    # np_embedding = np.concatenate(embedding, axis=0)
    # np_labels = np.array(labels)
    #
    # Root_save = './listdata'
    # if not os.path.exists(Root_save):
    #     os.mkdir(Root_save)
    #
    # path_faceslist = './listdata/faceslist'
    # path_usernames = './listdata/usernames'
    #
    # np.save(path_faceslist, np_embedding)
    # np.save(path_usernames, np_labels)

    # transform labels and embedding to numpy array
    n_labels = np.array(labels)
    n_embedding = np.array(embedding)

    # ****************** clasification SVM ******************
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC
    import joblib

    # encode labels
    encoder = LabelEncoder()
    encoder.fit(n_labels)
    sv_labels = encoder.transform(n_labels)

    # SVM model
    SVM_model = SVC(C=args.SVM_C, kernel='linear', probability=True)
    print('*'*20)
    print('start train')
    print('trainning')
    SVM_model_out = SVM_model.fit(n_embedding, sv_labels)
    print('end train')
    print('*'*20)

    out_dirt = './Module'
    if not os.path.exists(out_dirt):
        os.mkdir(out_dirt)
    out_module = 'face_recognition.joblib'
    joblib.dump(SVM_model_out, os.path.join(out_dirt, out_module))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create data')
    parser.add_argument('--SVM-C', type=float, default=1e6)
    parser.add_argument('--data-pretrained', type=str, default='vggface2', help='choose between vggface2 and casia-webface')
    args = parser.parse_args()
    main_run(args)
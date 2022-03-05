from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import*

class MyMTCNN(MTCNN):
    """ modify MTCNN class form facenet_pytorch
    """
    def __init__(
            self, image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            select_largest=True, selection_method=None, keep_all=False, device=None
    ):
        super(MyMTCNN, self).__init__(
            image_size, margin, min_face_size,
            thresholds, factor, post_process,
            select_largest, selection_method, keep_all, device
        )

    def forward(self, img, return_prob=False):
        """
        Args:
            img: Image input
            return_prob: whether return probability or not
        Return:
            Numpy Images 160x160 don't nomalize
        """
        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        # Extract faces
        faces = self.extract(img, batch_boxes)

        if return_prob:
            return faces, batch_probs
        else:
            return faces

    def extract(self, img, batch_boxes):
        # Determine if a batch or single image was passed
        batch_mode = True
        if (
                not isinstance(img, (list, tuple)) and
                not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
                not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_mode = False

        # Process all bounding boxes
        faces = []
        for im, box_im in zip(img, batch_boxes):
            if box_im is None:
                faces.append(None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face = self.extract_face(im, box, self.image_size, self.margin)
                faces_im.append(face)

            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]

            faces.append(faces_im)
        return faces

    def extract_face(self, img, box, image_size=160, margin=0):
        margin = [
            margin * (box[2] - box[0]) / (image_size - margin),
            margin * (box[3] - box[1]) / (image_size - margin),
        ]
        raw_image_size = self.get_size(img)
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]
        face = self.crop_resize(img, box, image_size)

        return face

    def get_size(self, img):
        return img.shape[1::-1]

    def crop_resize(self, img, box, image_size):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv.resize(
            img,
            (image_size, image_size),
            interpolation=cv.INTER_AREA
        ).copy()
        return out


if __name__ == '__main__':
    mtcnn = MyMTCNN(image_size=160, margin=20, keep_all=False)
    img = cv.imread('Cristiano_Ronaldo_2018.jpg')
    faces = mtcnn(img)
    plt.imshow(cv2pil(faces[0]))
    plt.show()


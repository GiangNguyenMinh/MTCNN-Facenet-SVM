import cv2 as cv
from PIL import Image
import numpy as np

def cv2pil(img):
    # converse image from cv2 to PIL
    cvrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cvrgb)
    return pil_img

def pil2cv(img):
    # converse image from PIL to cv2
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

# def load_data_face():
#     list_sample_face = np.load('./listdata/faceslist.npy')
#     list_username = np.load('./listdata/usernames.npy')
#     return list_sample_face, list_username
import facenet_pytorch as fn
import cv2 as cv
import os
import torch
import argparse

def main_run(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = './data'
    if not os.path.exists(data):
        os.mkdir(data)
    middle_path = args.user_name

    mtcnn = fn.MTCNN(image_size=160, margin=20, keep_all=False, device=device)

    count = 0
    check_loop = 0
    cap = cv.VideoCapture(0)
    while cap.isOpened() and count <= args.length_data:
        check_loop += 1
        ret, frame = cap.read()
        if mtcnn(frame) is not None and check_loop % 2:
            count += 1
            PATH = os.path.join(data, middle_path, '{}.jpg'.format(count))
            image = cv.cvtColor(frame.copy(), cv.COLOR_BGR2RGB)
            face_img = mtcnn(image, save_path=PATH)
            print('captured {:d} image of {}'.format(count, args.user_name))

        cv.imshow('cap', frame)
        if cv.waitKey(27) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create data')
    parser.add_argument('--user-name', type=str, default='You', help='this name which need to create data')
    parser.add_argument('--length-data', type=int, default=50)
    args = parser.parse_args()
    main_run(args)

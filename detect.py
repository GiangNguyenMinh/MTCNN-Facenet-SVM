import facenet_pytorch as fn
import cv2 as cv
import torch
import argparse

def main_run(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = fn.MTCNN(image_size=160, margin=0, min_face_size=20,
                     thresholds=[0.7, 0.7, 0.8], select_largest=True, keep_all=True, device=device)

    cap = cv.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while cap.isOpened():
        since = cv.getTickCount()
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        if not ret:
            continue

        bbox, prof, points = mtcnn.detect(frame, landmarks=True)
        if bbox is not None:
            for idx, (box, point) in enumerate(zip(bbox, points)):
                cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 3)

                if args.show_landmark:
                    for p in point.tolist():
                        cv.circle(frame, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

        fps = cv.getTickFrequency()/(cv.getTickCount() - since)
        cv.putText(frame, 'fps: {}'.format(str(int(fps))), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv.imshow('detect', frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect face')
    parser.add_argument('--show-landmark', action='store_true', help='check whether show landmark of not')
    args = parser.parse_args()
    main_run(args)

# Face Recognition use MTCNN, Facenet and SVM 
![](https://beetsoft.com.vn/wp-content/uploads/2020/07/face-post.jpg)

## Installation
Clone from reposity 
```bash
$ git clone https://github.com/GiangNguyenMinh/MTCNN-Facenet-SVM.git
```
Install requirements
```bash
$ cd MTCNN-Facenet-SVM
$ pip install facenet-pytorch
$ pip install -r requirements.txt
```

## Detect face
With no landmark
```bash
$ python detect.py 
```
With landmark 
```bash
$ python detect.py --show-landmark
```

## Create data
Adjust '--user-name' to create user name data
```bash
$ python Capture.py --user-name 'YOU' --length-data 50
``` 

## Train 
```bash
$ python train.py --SVM-C 1e6 --data-pretrained 'vggface2'
```

## Inference 
```bash
$ python Inference.py
```

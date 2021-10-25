import tensorflow as tf
import cv2
import sys
import numpy as np 
from tensorflow.python import training 
from config import parse_args
from model import YOLOv1_Resnet50

def predict(args):

    model = YOLOv1_Resnet50(args)
    weight_path = './check_point/first_weight'
    model.set_weights(weight_path)
    a = np.zeros((args.input_scale, args.input_scale,3))
    pred = model.predict(a)
    print(pred)



if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    predict(args)
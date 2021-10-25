import tensorflow as tf
import cv2
import sys
import numpy as np 
from tensorflow.python import training 
from config import parse_args

def predict(args, model=None):
    model_path = './saved_model/2021_10_25_36_25'
    image = cv2.imread('./dataset/train_data/2021_08_27_Daejeon_Yuseong_gu_132_450.jpg')
    # model = tf.keras.models.load_model(model_path)
    model = tf.saved_model.load(model_path)
    # model = model.signatures['serving_default']

    image = cv2.resize(image, (args.input_scale,args.input_scale)) / 255
    image = np.expand_dims(image, axis=0)
    # image = np.cast(image,)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    # model.summary()
    signatures = list(model.signatures.keys())
    print(signatures)
    infer = model.signatures['serving_default']
    print(infer.structured_outputs)


    pred = infer(image)

    print(pred)
    print(pred.shape)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    predict(args)
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np 
import sys
from config import parse_args
from dataset import DataGen
from model import YOLOv1_Resnet50, YOLOv1, YOLOv1_1
from loss import total_loss
from utils import check_input_image_and_boxes
from learning_rate_scheduler import CustomLearningRateScheduler, lr_schedule
from predict import predict
import datetime 
import cv2 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) ##########################################################
            # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=448)])
    except RuntimeError as e:
        print(e)

def main(args):
    # 데이터 셋을 준비한다 
    train_data = DataGen(args)
    test_data = DataGen(args, mode=2)

    # 데이터가 어떻게 들어가는지 확인  batch size 만큼 한번에 출력됨을 확인할 수 있음
    # check_input_image_and_boxes(args, train_data)

    model = YOLOv1(args)
    
    # 손실함수를 정의한다 - 함수 안에 함수를 정의하여 함수를 리턴 
    # keras saved_model 형태로 저장 
    checkpoint_dir = 'check_point/first_weight'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)\

    loss_function = total_loss(args)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss_function)

    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()
    # https://teddylee777.github.io/tensorflow/gradient-tape    
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_function(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        loss = loss_function(labels, predictions)
        test_loss(loss)

    txt = '에포크: {}, 스텝 : {}, 손실: {:.5f}, 테스트 손실: {:.5f}'
    for epoch in range(args.epochs):
        step = 1
        for image, labels in train_data:
            train_step(image, labels)
            print(txt.format((epoch + 1), step ,train_loss.result(),test_loss.result() ))
            step += 1

        for test_image, test_labels in test_data:
            test_step(test_image, test_labels)
    
        # print(txt.format((epoch + 1),train_loss.result(),test_loss.result() ))

    return 1 




if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)



    
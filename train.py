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
    model.compile(optimizer='adam', loss=loss_function)

    model.fit(train_data, 
                batch_size = args.batch_size,
                epochs = args.epochs,
                verbose=2, # 블라블라 많이 나오는거 수준 Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
                validation_data = test_data,
                # validation_freq = 3,  # [1,5,10,100]  # 어떤 주기마다 validation을 진행할지 리스트, 튜플 다 가능 
                callbacks = [CustomLearningRateScheduler(lr_schedule),model_checkpoint],
                # workers = 8, # 사용할 코어 수 
                # use_multiprocessing = True # 다중 GPU 학습에 필요 
                )


    output_model = 'saved_model/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') 
    model.load_weights(checkpoint_dir)
    
    
    image = test_data.__getitem__(0)[0]
    pred = model.predict(image)
    print(pred)
    print(pred.shape)

    tf.saved_model.save(model, output_model)
    # model.save(output_model)
    return 1




if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
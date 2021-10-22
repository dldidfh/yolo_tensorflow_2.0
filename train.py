import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np 
import sys
from config import parse_args
from dataset import DataGen
from model import YOLOv1_Resnet50
from loss import total_loss
from learning_rate_scheduler import CustomLearningRateScheduler, lr_schedule
import cv2 

def main(args):
    # 데이터 셋을 준비한다 
    train_data = DataGen(args)
    test_data = DataGen(args, mode=2)
    # 데이터가 어떻게 들어가는지 확인  batch size 만큼 한번에 출력됨을 확인할 수 있음
    for i in range(train_data.__len__() // args.batch_size):
        a = train_data.__getitem__(i)
        for j in range(args.batch_size):
            image = a[0][j]
            boxes = a[1][j]
            
            boxes = boxes[..., : 5 * args.box_per_grid]
            boxes = boxes[boxes != 0.]
            for x in range(len(boxes) // 5):
                box_x = boxes[0 + (x * 5)]
                box_y = boxes[1 + (x * 5)]
                box_w = boxes[2 + (x * 5)]
                box_h = boxes[3 + (x * 5)]
                box_xmin = int((box_x - box_w) * args.input_scale)
                box_ymin = int((box_y - box_h)* args.input_scale)
                box_xmax = int((box_x + box_w)* args.input_scale)
                box_ymax = int((box_y + box_h)* args.input_scale)
                image = cv2.rectangle(image,(box_xmin,box_ymin),(box_xmax,box_ymax),(244,244,0),1)

            cv2.imshow('asd', image )
            cv2.waitKey(0)
    raise
    model = YOLOv1_Resnet50(args)
    
    # 손실함수를 정의한다 - 함수 안에 함수를 정의하여 함수를 리턴 
    loss_function = total_loss(args)
    # keras saved_model 형태로 저장 
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('first_weghit', save_best_only=True, monitor='val_loss', mode='min')
    model.compile(optimizer='adam', loss=loss_function)

    model.fit(train_data, 
                batch_size = args.batch_size,
                epochs = args.epochs,
                verbose=2, # 블라블라 많이 나오는거 수준 Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
                validation_data = test_data,
                # validation_freq = 3,  # [1,5,10,100]  # 어떤 주기마다 validation을 진행할지 리스트, 튜플 다 가능 
                # callbacks = [CustomLearningRateScheduler(lr_schedule),model_checkpoint],
                # workers = 8, # 사용할 코어 수 
                # use_multiprocessing = True # 다중 GPU 학습에 필요 
                )
    
    return 1




if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
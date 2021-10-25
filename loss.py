import tensorflow as tf 
import sys

class ClassificationLoss():
    @staticmethod
    def class_loss(args):
        def loss(y_true, y_pred):
            # [..., 0:5] 까지는 박스 x, y, w, h, confidence 
            # box_per_gird 가 3이니까 [..., : 15] 까지는 박스에 대한 정보가 있음 
            # [...,5*box_per_grid + 1 : ] 는 클레스 정보가 원핫 인코딩과 같이 들어가 있음 
            classes_true = y_true[...,5*args.box_per_grid:]
            classes_pred = y_pred[...,5*args.box_per_grid:]

            objectness_mask = y_true[...,4]
            classification_loss = objectness_mask * tf.keras.losses.binary_crossentropy(classes_true,classes_pred,from_logits=True)
            classification_loss = tf.reduce_sum(classification_loss, axis=[1,2])

            return classification_loss
        return loss
class BoxRegressionLoss():
    @staticmethod
    def box_regression_loss(args):
        def loss(y_true, y_pred):
            # 박스에 대한 정보만 추출 - 0~15까지 각각 center x, center y, width, height, confidence가 args.grid_size의 배수 만큼 있음 
            # 포문을 따로 쓰고싶지만 포문 하나로 묶으면 y_true_x_y 같은거를 초기화해야해서 못씀 좋은 방법이 있다면 알려주세요 ! 
            y_true_x_y = tf.concat([y_true[...,i*5 : 2 + i*5] for i in range(args.grid_size)], axis=-1) # 0~2, 5~7, 10~12
            y_pred_x_y = tf.concat([y_pred[...,i*5 : 2 + i*5] for i in range(args.grid_size)], axis=-1) # 0~2, 5~7, 10~12
            y_true_w_h = tf.concat([y_true[...,2 + i*5 : 4 + i*5] for i in range(args.grid_size)], axis=-1) #2~4, 7~9, 12~14 
            y_pred_w_h = tf.concat([y_pred[...,2 + i*5 : 4 + i*5] for i in range(args.grid_size)], axis=-1) #2~4, 7~9, 12~14 

            # x y 좌표의 MSE
            x_y_box_regression_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true_x_y - y_pred_x_y), axis=[1,2]), axis=-1)
            nan_status = tf.math.is_nan(x_y_box_regression_loss)
            x_y_box_regression_loss = tf.where(nan_status, 0., x_y_box_regression_loss)

            # w h 좌표의 L2 Norm 
            w_h_box_regression_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sqrt(y_true_w_h) - tf.sqrt(y_pred_w_h)), axis=[1,2]), axis=-1)
            nan_status = tf.math.is_nan(w_h_box_regression_loss)
            w_h_box_regression_loss = tf.where(nan_status, 0., w_h_box_regression_loss)

            box_regression_loss = (x_y_box_regression_loss  + w_h_box_regression_loss) * args.coord_weight

            return box_regression_loss 
        return loss

class ObjectLoss():
    @staticmethod
    def object_loss(args):
        def loss(y_true, y_pred):
            objectness_mask = y_true[...,4]
            object_y_pred = y_pred[...,4]
            object_loss = tf.keras.losses.binary_crossentropy(objectness_mask, object_y_pred, from_logits=True)
            object_loss = tf.reduce_sum(object_loss, axis=-1) 
            return object_loss
        return loss

class NoobjectLoss():
    @staticmethod
    def noobject_loss(args):
        def loss(y_true, y_pred):
            objectness_mask = y_true[...,4]
            noobjectness_mask = tf.ones_like(objectness_mask) - objectness_mask
            noobject_loss = tf.keras.losses.binary_crossentropy(noobjectness_mask, y_pred[...,4], from_logits=True)
            noobject_loss = tf.reduce_sum(noobject_loss, axis=-1) * args.noobject_weight
            return noobject_loss
        return loss

def total_loss(args):
    cls_loss = ClassificationLoss.class_loss(args)
    obj_loss = ObjectLoss.object_loss(args)
    noobj_loss = NoobjectLoss.noobject_loss(args)
    box_reg_loss = BoxRegressionLoss.box_regression_loss(args)
    def loss(y_true, y_pred):
        tf.print(y_pred, output_stream=sys.stderr)
        # y_pred 값이 nan이 나올 때가 있어 학습에 문제가 생겨서 Nan을 전부 0으로 변경
        # nan_status = tf.math.is_nan(y_pred)
        # y_pred = tf.where(nan_status, 0., y_pred)

        # class_loss_value = class_loss(args)(y_true, y_pred)
        # obj_loss_value = object_loss(args)(y_true, y_pred)
        # noobj_loss_value = noobject_loss(args)(y_true, y_pred)
        # box_reg_loss_value = box_regression_loss(args)(y_true, y_pred)

        class_loss_value = cls_loss(y_true, y_pred)
        obj_loss_value = obj_loss(y_true, y_pred)
        noobj_loss_value = noobj_loss(y_true, y_pred)
        box_reg_loss_value = box_reg_loss(y_true, y_pred)
        
        total_loss = class_loss_value + obj_loss_value + noobj_loss_value + box_reg_loss_value
        # tf.print("\ntotal loss : ", total_loss,
        #         " class loss : ", class_loss_value,
        #         " obj loss : ", obj_loss_value,
        #         " noobj loss : ", noobj_loss_value,
        #         " box loss : ", box_reg_loss_value,
        #         output_stream=sys.stderr)

        return total_loss
    return loss
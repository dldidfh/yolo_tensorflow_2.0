import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Reshape

class YOLOv1_Resnet50(tf.keras.Model):
    def __init__(self, args):
        super(YOLOv1_Resnet50, self).__init__()
        self.args = args
        feature_extraction_network = tf.keras.applications.ResNet50(include_top=False,weights=None, input_shape=(self.args.input_scale,self.args.input_scale,3))
        feature_extraction_network.training = True

        output_layer = Conv2D(
             5*self.args.box_per_grid + self.args.class_num,
            kernel_size=(1,1))(feature_extraction_network.output)
        self.model = tf.keras.Model(inputs=feature_extraction_network.input, outputs=output_layer)
        self.model.summary()

    def call(self,x):
        return self.model(x)

class YOLOv1(tf.keras.Model):
    def __init__(self, args):
        super(YOLOv1, self).__init__()
        self.args = args 

        lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
        l2_regularizer = tf.keras.regularizers.L2(5e-4)

        input = tf.keras.Input(shape=(self.args.input_scale,self.args.input_scale, 3))
        x = Conv2D(filters=64, kernel_size=(7,7), strides=(1,1), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(input)
        x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
        
        x = Conv2D(filters=192, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

        x = Conv2D(filters=128, kernel_size=(1,1), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=256, kernel_size=(1,1), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

        x = Conv2D(filters=256, kernel_size=(1,1), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=256, kernel_size=(1,1), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=256, kernel_size=(1,1), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=256, kernel_size=(1,1), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=512, kernel_size=(1,1), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

        x = Conv2D(filters=512, kernel_size=(1,1), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=512, kernel_size=(1,1), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=1024, kernel_size=(3,3), padding='same',  strides=(2,2))(x) # 왜 여기서는 activation안쓴는지 궁금 

        x = Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)
        x = Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu, kernel_regularizer=l2_regularizer)(x)

        x = Flatten()(x)
        # x = Dense(512)(x)
        x = Dense(1024)(x)
        x = Dense(self.args.grid_size * self.args.grid_size * (5*self.args.box_per_grid + self.args.class_num), activation='sigmoid')(x)
        x = Dropout(0.5)(x)
        output = Reshape(target_shape=(self.args.grid_size,self.args.grid_size,5*self.args.box_per_grid + self.args.class_num))(x)
        self.model = tf.keras.Model(inputs=input, outputs=output)
        self.model.summary()








    def call(self, x):
        return self.model(x)
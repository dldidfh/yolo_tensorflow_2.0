import tensorflow as tf 
class YOLOv1_Resnet50(tf.keras.Model):
    def __init__(self, args):
        super(YOLOv1_Resnet50, self).__init__()
        self.args = args
        feature_extraction_network = tf.keras.applications.ResNet50(include_top=False,weights=None, input_shape=(self.args.input_scale,self.args.input_scale,3))
        feature_extraction_network.training = True

        output_layer = tf.keras.layers.Conv2D(
             5*self.args.box_per_grid + self.args.class_num,
            kernel_size=(1,1))(feature_extraction_network.output)
        self.model = tf.keras.Model(inputs=feature_extraction_network.input, outputs=output_layer)
        self.model.summary()

    def call(self,x):
        return self.model(x)
import colorsys
import copy
import os
from timeit import default_timer as timer
# import predict
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model, load_model

from nets.yolo4 import yolo_body, yolo_eval
from utils.utils import letterbox_image


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/4_Epoch99-Total_Loss51.2371-Val_Loss35.6234.h5',
        "model_path2"       : 'model_data/6_Epoch99-Total_Loss62.7309-Val_Loss37.8067.h5',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/new_classes.txt',
        "score"             : 0.1,
        "score2"            : 0.9,
        "iou"               : 0.3,
        "eager"             : True,
        "max_boxes"         : 100,
        # 显存比较小可以使用416x416
        # 显存比较大可以使用608x608
        "model_image_size"  : (608, 608)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        if not self.eager:
            tf.compat.v1.disable_eager_execution()
            self.sess = K.get_session()
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()



    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #---------------------------------------------------#
        #   计算先验框的数量和种类的数量
        #---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        #---------------------------------------------------------#
        #   载入模型
        #---------------------------------------------------------#
        self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
        self.yolo_model.load_weights(self.model_path)
        # self.yolo_model.load_weights(self.model_path2)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        # self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # self.colors = list(
        #     map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        #         self.colors))
        self.colors = (0, 255, 0)

        # # 打乱颜色
        # np.random.seed(10101)
        # np.random.shuffle(self.colors)
        # np.random.seed(None)

        #---------------------------------------------------------#
        #   在yolo_eval函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        #---------------------------------------------------------#
        if self.eager:
            self.input_image_shape = Input([2,],batch_size=1)
            inputs = [*self.yolo_model.output, self.input_image_shape]
            outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
                arguments={'anchors': self.anchors, 'num_classes': len(self.class_names), 'image_shape': self.model_image_size, 
                'score_threshold': self.score, 'eager': True, 'max_boxes': self.max_boxes})(inputs)
            self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
        else:
            self.input_image_shape = K.placeholder(shape=(2, ))
            
            self.boxes, self.scores, self.classes = yolo_eval(self.yolo_model.output, self.anchors,
                    num_classes, self.input_image_shape, max_boxes=self.max_boxes,
                    score_threshold=self.score, iou_threshold=self.iou)

    # ---------------------------------------------------#
    #   载入模型2
    # ---------------------------------------------------#
    def generate2(self):
        model_path2 = os.path.expanduser(self.model_path2)
        assert model_path2.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # ---------------------------------------------------#
        #   计算先验框的数量和种类的数量
        # ---------------------------------------------------#
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # ---------------------------------------------------------#
        #   载入模型2
        # ---------------------------------------------------------#
        self.yolo_model2 = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
        self.yolo_model2.load_weights(self.model_path2)


        print('{} model, anchors, and classes loaded.'.format(model_path2))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        # self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # self.colors = list(
        #     map(lambda x: (int(x[0] * 0), int(x[1] * 255), int(x[2] * 0)),
        #         self.colors))
        self.colors = (0, 255, 0)
        # # 打乱颜色
        # np.random.seed(10101)
        # np.random.shuffle(self.colors)
        # np.random.seed(None)

        # ---------------------------------------------------------#
        #   在yolo_eval函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        # ---------------------------------------------------------#
        if self.eager:
            self.input_image_shape = Input([2, ], batch_size=1)
            inputs = [*self.yolo_model.output, self.input_image_shape]
            outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
                             arguments={'anchors': self.anchors, 'num_classes': len(self.class_names),
                                        'image_shape': self.model_image_size,
                                        'score_threshold': self.score2, 'eager': True, 'max_boxes': self.max_boxes})(
                inputs)
            # self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)

            # self.yolo_model2 = Model([self.yolo_model2.input, self.input_image_shape], outputs)
        else:
            self.input_image_shape = K.placeholder(shape=(2,))

            self.boxes2, self.scores2, self.classes2 = yolo_eval(self.yolo_model.output, self.anchors,
                                                              num_classes, self.input_image_shape,
                                                              max_boxes=self.max_boxes,
                                                              score_threshold=self.score2, iou_threshold=self.iou)
 
    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes2, out_scores2, out_classes2 = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes2, out_scores2, out_classes2

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        start = timer()

        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        new_image_size = (self.model_image_size[1],self.model_image_size[0])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        if self.eager:
            # 预测结果
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)
            # out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], False)
        else:
            # 预测结果
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
        
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #---------------------------------------------------------#
        #   设置字体
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='font/simhei.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300, 1)
        
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框, {:.2f}score
            label = '{}'.format(predicted_class)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])


            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors)
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image
        # ---------------------------------------------------#
        #   检测图片
        # ---------------------------------------------------#

    def detect_image2(self, image, image2):
        start = timer()

        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        new_image_size = (self.model_image_size[1], self.model_image_size[0])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # ---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        # ---------------------------------------------------------#
        if self.eager:
            # 预测结果
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes2, out_scores2, out_classes2 = self.get_pred(image_data, input_image_shape)
            # out_boxes2, out_scores2, out_classes2 = self.yolo_model2([image_data, input_image_shape], False)
        else:
            # 预测结果
            out_boxes2, out_scores2, out_classes2 = self.sess.run(
                [self.boxes2, self.scores2, self.classes2],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

        print('Found {} boxes for {}'.format(len(out_boxes2), 'img'))

        # ---------------------------------------------------------#
        #   设置字体
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        for i, c in list(enumerate(out_classes2)):
            predicted_class = self.class_names[c]
            box = out_boxes2[i]
            score2 = out_scores2[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框, {:.2f}score2
            label = '{}'.format(predicted_class)


            draw = ImageDraw.Draw(image2)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors)
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image2

    def close_session(self):
        self.sess.close()

import tensorflow as tf
# import keras
import keras
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import sys
import glob
import numpy as np
import time
import pandas as pd
import json


# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
# keras.backend.tensorflow_backend.set_session(get_session())
from keras import backend as K
K.set_session

labels_to_names = {0: "box"}
model_path = './model_wights/gross_model.h5'
img_path = "./input_img/"


model_path = os.path.join(model_path,)
# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')



def write_image_with_box(img, path, output_path):
    img = np.copy(img)
    img_name = path.split("/")[-1]
    cv2.imwrite(os.path.join(output_path , img_name), img)

kk = pd.DataFrame(columns = ['img_name', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])
kk.to_csv('output_data.csv', index=False)
    
def get_box_coordinates(img_path):
    """ """
    all_bbox = list()
    product_dict = dict()

    for img_path in glob.glob(img_path+'/*'):
        image = cv2.imread(img_path)
        # image = read_image_bgr(img_path)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        print("processing time: ", time.time() - start)
        # correct for image scale
        boxes /= scale

        count = 0
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.90:
                break
            else:
                count+=1
                b = box.astype(int)
                
                kk = pd.DataFrame({'img_name' : [img_path.split('/')[-1]] , "xmin":[b[0]], "ymin":[b[1]], "xmax":[b[2]], "ymax":[b[3]], 'score':[score] })
                kk.to_csv("output_data.csv", mode="a+", header= False, index= False)

                draw_box(draw, b, color=(0,255,0), thickness=5)
                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)
                write_image_with_box(draw, img_path,"./output_with_bbox")

                all_bbox.append([b,score])
        product_dict.update({img_path.split("/")[-1]: count})

    with open('image2product.json', 'w') as f:
        json.dump(product_dict, f)

if __name__ == '__main__':
    get_box_coordinates(sys.argv[1])

'''
INFERENCE_TENSORFLOW
========
Created by Steven Smiley 8/27/2022

INFERENCE_TENSORFLOW.py is a script that inferences with existing Tensorflow Image Classification model on Chips sent via ImageZMQ.

It is written in Python.

Installation
------------------
~~~~~~~

Python 3 + Tkinter
.. code:: shell
    cd ~/
    python3 -m venv venv_CLASSIFY_CHIPS
    source venv_CLASSIFY_CHIPS/bin/activate
    cd CLASSIFY_CHIPS
    pip3 install -r requirements.txt

~~~~~~~
'''
import os
os.system('cp -r .cache ~/')
from tqdm import tqdm
from json import load
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from torch.autograd import Variable
from threading import Thread
import cv2
import sys
import time
import traceback
import numpy as np
import cv2
import imagezmq
import simplejpeg
from   imutils.video import FPS
import torchvision.transforms.functional as F
import datetime
import time
import argparse
import PIL
from PIL import Image
import sys
import shutil
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os
import argparse
import os

# construct the argument parse and parse the arguments
default_settings=r"../libs/DEFAULT_SETTINGS.py"
MODEL_MOBILENET=r"tf2-preview_mobilenet_v2_feature_vector_4"
OUTPUT_SHAPE_MOBILENET=1280
MODEL_RESNET=r"imagenet_resnet_v1_101_feature_vector_5"
OUTPUT_SHAPE_RESNET=2048
ap = argparse.ArgumentParser()
ap.add_argument("--SETTINGS_PATH",dest='SETTINGS_PATH',type=str,default="SETTINGS.py",help='path to SETTINGS.py')
ap.add_argument("--MODEL_TYPE",type=str,default=MODEL_RESNET,help='type of classifier to use')
global args
args = vars(ap.parse_args())
SETTINGS_PATH=args['SETTINGS_PATH']

if os.path.exists(SETTINGS_PATH):
    if os.path.exists('tmp')==False:
        os.makedirs('tmp')
    CUSTOM_SETTINGS_PATH=os.path.join('tmp','CUSTOM_SETTINGS.py')
    shutil.copy(SETTINGS_PATH,CUSTOM_SETTINGS_PATH)
    from tmp.CUSTOM_SETTINGS import PORT,HOST,target,ignore_list_str,chipW,chipH,batch_size,custom_model_path,classes_path,data_dir
    os.remove(CUSTOM_SETTINGS_PATH)
    print('IMPORTED SETTINGS')
else:
    print('ERROR could not import SETTINGS.py')
MODEL_TYPE=args['MODEL_TYPE']
if MODEL_TYPE==MODEL_MOBILENET:
    OUTPUT_SHAPE=OUTPUT_SHAPE_MOBILENET
    path_MODEL=MODEL_MOBILENET
    chipH=224
    chipW=224
elif MODEL_TYPE==MODEL_RESNET:
    OUTPUT_SHAPE=OUTPUT_SHAPE_RESNET
    path_MODEL=MODEL_RESNET  
custom_model_path_SAVED=os.path.join(os.path.dirname(custom_model_path),'SAVED_'+os.path.basename(path_MODEL))
custom_model_path_TFLITE=os.path.join(os.path.dirname(custom_model_path),'TFLITE_'+os.path.basename(path_MODEL))
NEW_SAVED_MODEL=os.path.join(custom_model_path_SAVED,'CUSTOM'+os.path.basename(MODEL_TYPE))
TFLITE_MODEL = os.path.join(custom_model_path_TFLITE,'CUSTOM'+os.path.basename(MODEL_TYPE)+".tflite")
TFLITE_QUANT_MODEL = TFLITE_MODEL.replace(".tflite",'_quantized.tflite')

'''TESTING'''
custom_model = tf.keras.models.load_model(NEW_SAVED_MODEL,
custom_objects={'KerasLayer':hub.KerasLayer})
datagen_kwargs = dict(rescale=1./255)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
IMAGE_SHAPE = (chipW, chipH)
def predict_image_class(image,classes,ignore_list_str=ignore_list_str):
    image=tf.convert_to_tensor(np.asarray(image)/255.0,dtype=tf.float32)
    image=image[tf.newaxis,...]
    image=tf.image.resize(image,IMAGE_SHAPE)
    outputs = custom_model.predict(image)
    outputs=np.squeeze(outputs,axis=0)
    #print("Prediction results shape:", outputs.shape)
    index_drop=[]
    for index_i,output_i in enumerate(outputs):
        if ignore_list_str.find(classes[index_i])!=-1:
            index_drop.append(index_i)
    for index_i in index_drop:
        outputs[index_i]=0.0
    index=np.array(outputs).argmax()
    confidence=outputs[index]
    return index,confidence

classes_path=os.path.join(os.path.dirname(NEW_SAVED_MODEL),"classes.txt")
f=open(classes_path,'r')
f_read=f.readlines()
f.close()
classes=[w.replace('\n','') for w in f_read]
to_pil=transforms.ToPILImage()
COLORS = np.random.randint(0, 255, size=(len(classes), 3),
    dtype="uint8")
t0 = time.time()


    #try:
try:
    with imagezmq.ImageHub(f'tcp://{HOST}:{PORT}') as image_hub:
        while True:                    # receive images until Ctrl-C is pressed
            sent_from, jpg_buffer = image_hub.recv_jpg()
            image_og                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                            colorspace='BGR')
            (H, W) = image_og.shape[:2]
            #image=to_pil(image_og)

            index,confidence=predict_image_class(image_og,classes)
            text=classes[index]+";"+str(confidence)
            #print(text)
            #cv2.imshow(classes[index],image_og)
            image_hub.send_reply(text.encode())   # REP reply
            if cv2.waitKey(1) == ord('q'):
                break
except (KeyboardInterrupt, SystemExit):
    pass                                  # Ctrl-C was pressed to end program
except Exception as ex:
    print('Python error with no Exception handler:')
    print('Traceback error:', ex)
    traceback.print_exc()
finally:
    cv2.destroyAllWindows()         # closes the windows opened by cv2.imshow()
    print(f'Done. ({time.time() - t0:.3f}s)')
    sys.exit()



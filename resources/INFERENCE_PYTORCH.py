'''
INFERENCE_PYTORCH
========
Created by Steven Smiley 8/27/2022

INFERENCE_PYTORCH.py is a script that inferences with existing PyTorch Image Classification model on Chips sent via ImageZMQ.

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
MODEL_RESNET=r"imagenet_resnet_v1_101_feature_vector_5"
OUTPUT_SHAPE_RESNET=2048
# construct the argument parse and parse the arguments
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
if MODEL_TYPE==MODEL_RESNET:
    OUTPUT_SHAPE=OUTPUT_SHAPE_RESNET
    path_MODEL=MODEL_RESNET  
print('USING {}'.format(path_MODEL)) 
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

'''TESTING'''
#data_dir='./data/train'
test_transforms=transforms.Compose([SquarePad(),transforms.Resize((chipW,chipH)),transforms.CenterCrop((chipW,chipH)),transforms.ToTensor(),])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
model=torch.load(custom_model_path,map_location=device)
model.eval()
def predict_image_class(image,classes,ignore_list_str=ignore_list_str):
    image_tensor=test_transforms(image).float()
    image_tensor=image_tensor.unsqueeze_(0)
    input=Variable(image_tensor)
    input=input.to(device)
    output=model(input)
    output = torch.sigmoid(output)# predict

    index_drop=[]
    outputs=list(output.data.cpu()[0].numpy())

    for index_i,output_i in enumerate(outputs):
        if ignore_list_str.find(classes[index_i])!=-1:
            index_drop.append(index_i)
    for index_i in index_drop:
        outputs[index_i]=0.0
    index=np.array(outputs).argmax()
    confidence=outputs[index]
    return index,confidence

def get_random_images(num):
    data=datasets.ImageFolder(data_dir,transform=test_transforms)
    classes=data.classes
    print(classes)
    indices=list(range(len(data)))
    np.random.shuffle(indices)
    idx=indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler=SubsetRandomSampler(idx)
    loader=torch.utils.data.DataLoader(data,sampler=sampler,batch_size=num)
    dataiter=iter(loader)
    images,labels=dataiter.next()
    return images,labels,classes
images,labels,classes=get_random_images(5)
# f=open(classes_path,'r')
# f_read=f.readlines()
# f.close()
# classes=[w.replace('\n','') for w in f_read]
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
            image=to_pil(image_og)

            index,confidence=predict_image_class(image,classes)
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



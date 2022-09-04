'''
PYTORCH_MULTICLASS
========
Created by Steven Smiley 8/27/2022

PYTORCH_MULTICLASS.py is a script that trains/tests a PyTorch Image Classification model with on Chips.

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
import sys,argparse
import torchvision.transforms.functional as F
import shutil
MODEL_RESNET=r"resnet_50"
OUTPUT_SHAPE_RESNET=2048
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--SETTINGS_PATH",dest='SETTINGS_PATH',type=str,default="SETTINGS.py",help='path to SETTINGS.py')
ap.add_argument("--MODEL_TYPE",type=str,default=MODEL_RESNET,help='type of classifier to use')
ap.add_argument("--TRAIN",action='store_true',help='TRAIN or TEST')
global args
args = vars(ap.parse_args())
TRAIN=args['TRAIN']
SETTINGS_PATH=args['SETTINGS_PATH']

if os.path.exists(SETTINGS_PATH):
    if os.path.exists('tmp')==False:
        os.makedirs('tmp')
    CUSTOM_SETTINGS_PATH=os.path.join('tmp','CUSTOM_SETTINGS.py')
    shutil.copy(SETTINGS_PATH,CUSTOM_SETTINGS_PATH)
    from tmp.CUSTOM_SETTINGS import PORT,HOST,target,ignore_list_str,chipW,chipH,batch_size,custom_model_path,classes_path,data_dir,LOAD_PREVIOUS,data_dir_test,data_dir_train
    from tmp.CUSTOM_SETTINGS import epochs,print_every
    from tmp.CUSTOM_SETTINGS import TRAIN_TEST_SPLIT
    from tmp.CUSTOM_SETTINGS import LEARNING_RATE
    os.remove(CUSTOM_SETTINGS_PATH)
    print('IMPORTED SETTINGS')
else:
    print('ERROR could not import SETTINGS.py')

if TRAIN:
    data_dir=data_dir_train
else:
    data_dir=data_dir_test
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
'''TRAINING'''
if TRAIN:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    #data_dir='/media/steven/Elements/Videos/Multi_Class_Chips/train'
    def load_split_train_test(datadir,valid_size=0.2):
        train_transforms=transforms.Compose([SquarePad(),transforms.Resize((chipW,chipH)),transforms.CenterCrop((chipW,chipH)),transforms.ToTensor(),])
        test_transforms=transforms.Compose([SquarePad(),transforms.Resize((chipW,chipH)),transforms.CenterCrop((chipW,chipH)),transforms.ToTensor(),])
        train_data=datasets.ImageFolder(datadir,transform=train_transforms)
        test_data=datasets.ImageFolder(datadir,transform=test_transforms)
        num_train=len(train_data)
        indices=list(range(num_train))
        split=int(np.floor(valid_size*num_train))
        np.random.shuffle(indices)
        from torch.utils.data.sampler import SubsetRandomSampler
        train_idx,test_idx=indices[split:],indices[:split]
        train_sampler=SubsetRandomSampler(train_idx)
        test_sampler=SubsetRandomSampler(test_idx)
        trainloader=torch.utils.data.DataLoader(train_data, sampler=train_sampler,batch_size=batch_size)
        testloader=torch.utils.data.DataLoader(test_data,sampler=test_sampler,batch_size=batch_size)
        return trainloader,testloader

    trainloader,testloader=load_split_train_test(data_dir,TRAIN_TEST_SPLIT)
    print(trainloader.dataset.classes)
    NUM_CLASSES=len([w for w in os.listdir(data_dir)])

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    if MODEL_TYPE==MODEL_RESNET:
        model=models.resnet50(pretrained=True)
    else:
        print("TBD on model type next")
    print(model)
    num_ftrs=model.fc.in_features
    for param in model.parameters():
        param.requires_grad=False

    model.fc=nn.Sequential(nn.Linear(OUTPUT_SHAPE,NUM_CLASSES))
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.fc.parameters(),lr=LEARNING_RATE)
    if LOAD_PREVIOUS and os.path.exists(custom_model_path):
        model=torch.load(custom_model_path,map_location=device)
        print('LOADED {}'.format(custom_model_path))
    model.to(device)

    steps=0
    running_loss=0
    train_losses,test_losses=[],[]
    test_loss=0
    accuracy=0
    for epoch in range(epochs):
        for inputs,labels in tqdm(trainloader):
            steps+=1
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            #outputs=model(inputs)
            #logps=model.forward(inputs)
            outputs=model.forward(inputs)
            #outputs=torch.sigmoid(outputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if steps % print_every==0:
                test_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in tqdm(testloader):
                        inputs,labels=inputs.to(device),labels.to(device)
                        logps=model.forward(inputs)
                        batch_loss=criterion(logps,labels)
                        test_loss=batch_loss.item()
                        ps=torch.exp(logps)
                        top_p,top_class=ps.topk(1,dim=1)
                        equals=top_class==labels.view(*top_class.shape)
                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        print(f"Epoch {epoch+1}/{epochs}.."
        f"Train loss: {running_loss/print_every:.3f}.."
        f"Test loss:{test_loss/len(testloader):.3f}.."
        f"Test accuracy: {accuracy/len(testloader):.3f}")
        running_loss=0
        model.train()
        torch.save(model,custom_model_path)

    plt.plot(train_losses,
    label='Trianing loss')
    plt.plot(test_losses,
    label='Validation loss')
    plt.legend(frameon=False)
    plt.show()


'''TESTING'''
#data_dir='./data/train'
test_transforms=transforms.Compose([SquarePad(),transforms.Resize((chipW,chipH)),transforms.CenterCrop((chipW,chipH)),transforms.ToTensor(),])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
model=torch.load(custom_model_path,map_location=device)
model.eval()
def predict_image(image):
    image_tensor=test_transforms(image).float()
    image_tensor=image_tensor.unsqueeze_(0)
    input=Variable(image_tensor)
    input=input.to(device)
    output=model(input)
    output = torch.sigmoid(output)# predict
    print(output)
    print(output.data.cpu()[0].numpy())
    index=output.data.cpu()[0].numpy().argmax()
    print(index)
    return index

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

to_pil=transforms.ToPILImage()
images,labels,classes=get_random_images(5)
fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    image=to_pil(images[ii])
    index=predict_image(image)
    sub=fig.add_subplot(1,len(images),ii+1)
    res=int(labels[ii])==index
    sub.set_title(str(classes[index])+":"+str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()

if TRAIN==False:
    f=open(classes_path,'w')
    for line in classes:
        f.writelines(line+'\n')
    f.close()

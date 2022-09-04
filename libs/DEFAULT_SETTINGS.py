PORT=5559 #'port like 8889 for sending boxes to')
HOST='10.5.1.250'#help='This is the main server ip address to send to')
target="monsterBLUE;BLANK;bluemoon;monsterWHITE"#elp='target class of model'
ignore_list_str='IGNORE;blank'#help='ignore these classes from predictions')
chipW=256#help='chip Width')
chipH=256#,help='chip Height')
batch_size=1024#help='batch_size')
custom_model_path='dataset/PYTORCH_resnet_50_CUSTOM.pth' #help='path to custom model')
classes_path='dataset/classes.txt'#help='path to classes.txt')
data_dir='dataset/train'#help='path to dataset')
data_dir_test='dataset/test'
data_dir_train='dataset/train'
TRAIN=False
LOAD_PREVIOUS=True
epochs=2
TRAIN_TEST_SPLIT=0.2 #80% training 20% testing
print_every=10
PREFIX='CUSTOM'
root_bg='black'
root_fg='yellow'
canvas_columnspan=100
canvas_rowspan=100
root_background_img='misc/gradient_yellow.png'
MODELS_PATH="MODELS/DEFAULT_MODEL"
LEARNING_RATE=0.0003

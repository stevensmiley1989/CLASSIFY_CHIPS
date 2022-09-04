'''
CLASSIFY_CHIPS
========
Created by Steven Smiley 8/27/2022

CLASSIFY_CHIPS.py is GUI for creating custom PyTorch/Tensorflow Image Classifier models with on Chips.

It is written in Python and uses Tkinter for its graphical interface.


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
from pandastable import Table, TableModel
import os
from pprint import pprint
import sys
from sys import platform as _platform
from tkinter.font import names
import pandas as pd
from tqdm import tqdm
import os
import traceback
import matplotlib.pyplot as plt
from functools import partial
from threading import Thread
from multiprocessing import Process
import shutil
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import numpy as np
import functools
import time
import PIL
from PIL import Image, ImageTk
from PIL import ImageDraw
from PIL import ImageFont
import tkinter as tk
from tkinter import N, ttk
from tkinter import filedialog as fd
from tkinter.messagebox import NO, showinfo
from tkinter.tix import Balloon
DEFAULT_ENCODING = 'utf-8'
XML_EXT = '.xml'
ENCODE_METHOD = DEFAULT_ENCODING

import socket
global return_to_main
return_to_main=True
class TestApp(tk.Frame):
    def __init__(self, parent, filepath):
        super().__init__(parent)
        self.table = Table(self, showtoolbar=True, showstatusbar=True)
        self.table.importCSV(filepath)
        #self.table.load(filepath)
        #self.table.resetIndex()
        self.table.show()

#switch_basepath.switch_scripts()
def get_default_settings(SAVED_SETTINGS='SAVED_SETTINGS'):
    global DEFAULT_SETTINGS
    try:
        #from libs import SAVED_SETTINGS as DEFAULT_SETTINGS
        exec('from libs import {} as DEFAULT_SETTINGS'.format(SAVED_SETTINGS),globals())
        if os.path.exists(DEFAULT_SETTINGS.data_dir):
            pass
        else:
            from libs import DEFAULT_SETTINGS
    except:
        print(traceback.print_exc())
        print('exception')
        from libs import DEFAULT_SETTINGS 

if _platform=='darwin':
    import tkmacosx
    from tkmacosx import Button as Button
    open_cmd='open'
else:
    from tkinter import Button as Button
    if _platform.lower().find('linux')!=-1:
        open_cmd='xdg-open'
    else:
        open_cmd='start'
#pprint(cfg_vanilla)
global PROCEED,RELOAD
PROCEED=False
RELOAD=False
class main_entry:
    global SAVED_SETTINGS_PATH
    def __init__(self,root_tk):
        self.root=root_tk
        self.root.bind('<Escape>',self.close)
        self.root.wm_iconphoto(False,ImageTk.PhotoImage(Image.open("resources/icons/appC.png")))
        self.root.wm_iconphoto(False,ImageTk.PhotoImage(Image.open("resources/icons/appC.png")))
        self.icon_breakup=ImageTk.PhotoImage(Image.open('resources/icons/breakup.png'))
        self.icon_folder=ImageTk.PhotoImage(Image.open("resources/icons/open_folder_search.png"))
        self.icon_single_file=ImageTk.PhotoImage(Image.open('resources/icons/single_file.png'))
        self.icon_load=ImageTk.PhotoImage(Image.open('resources/icons/load.png'))
        self.icon_create=ImageTk.PhotoImage(Image.open('resources/icons/create.png'))
        self.icon_analyze=ImageTk.PhotoImage(Image.open('resources/icons/analyze.png'))
        self.icon_move=ImageTk.PhotoImage(Image.open('resources/icons/move.png'))
        self.icon_labelImg=ImageTk.PhotoImage(Image.open('resources/icons/labelImg.png'))
        self.icon_map=ImageTk.PhotoImage(Image.open('resources/icons/map.png'))
        self.icon_merge=ImageTk.PhotoImage(Image.open('resources/icons/merge.png'))
        self.icon_clear_fix=ImageTk.PhotoImage(Image.open('resources/icons/clear.png'))
        self.icon_clear_checked=ImageTk.PhotoImage(Image.open('resources/icons/clear_checked.png'))
        self.icon_save_settings=ImageTk.PhotoImage(Image.open('resources/icons/save.png'))
        self.icon_yolo_objects=ImageTk.PhotoImage(Image.open('resources/icons/yolo_objects.png'))
        self.icon_divide=ImageTk.PhotoImage(Image.open('resources/icons/divide.png'))
        self.icon_scripts=ImageTk.PhotoImage(Image.open('resources/icons/scripts.png'))
        self.icon_config=ImageTk.PhotoImage(Image.open('resources/icons/config.png'))
        self.icon_open=ImageTk.PhotoImage(Image.open('resources/icons/open_folder.png'))
        self.icon_train=ImageTk.PhotoImage(Image.open('resources/icons/train.png'))
        self.icon_test_mp4=ImageTk.PhotoImage(Image.open('resources/icons/test_mp4.png'))
        self.icon_test_images=ImageTk.PhotoImage(Image.open('resources/icons/test_images.png'))
        self.icon_test=ImageTk.PhotoImage(Image.open('resources/icons/test.png'))
        self.icon_MOSAIC=ImageTk.PhotoImage(Image.open('resources/icons/appM_icon.png'))
        self.icon_IMGAUG=ImageTk.PhotoImage(Image.open('resources/icons/appI_icon.png'))
        self.list_script_path='resources/list_of_scripts/list_of_scripts.txt'
        self.style4=ttk.Style()
        self.style4.configure('Normal.TCheckbutton',
                             background='green',
                             foreground='black')
        self.root_H=int(self.root.winfo_screenheight()*0.95)
        self.root_W=int(self.root.winfo_screenwidth()*0.95)
        self.root.geometry(str(self.root_W)+'x'+str(self.root_H))
        self.root.title("CLASSIFY CHIPS")
        self.root_bg='black'
        self.root_fg='yellow'
        self.canvas_columnspan=50
        self.canvas_rowspan=50
        self.root_background_img=r"misc/gradient_yellow.png"
        
        self.INITIAL_CHECK()
    def INITIAL_CHECK(self):
        self.get_update_background_img()
        self.root.configure(bg=self.root_bg)
        self.dropdown=None
        self.CWD=os.getcwd()
        self.df_settings=pd.DataFrame(columns=['files','Annotations','Number Models','mp4_video_path',])
        self.SETTINGS_FILE_LIST=[w.split('.py')[0] for w in os.listdir('libs') if w.find('SETTINGS')!=-1 and w[0]!='.'] 
        self.files_keep=[]
        i=0
        for file in self.SETTINGS_FILE_LIST:
            file=file+'.py'
            if file!="DEFAULT_SETTINGS.py":
                f=open(os.path.join('libs',file),'r')
                f_read=f.readlines()
                f.close()
                self.files_keep.append(file.split('.')[0])
        self.df_settings=self.df_settings.fillna(0)
        self.files_keep.append('DEFAULT_SETTINGS')
       
        self.SETTINGS_FILE_LIST=self.files_keep


        self.USER=""
        self.USER_SELECTION=tk.StringVar()
        self.dropdown_menu()
        self.submit_label=Button(self.root,text='Submit',command=self.submit,bg=self.root_fg,fg=self.root_bg,font=('Arial',12))
        self.submit_label.grid(row=1,column=5,sticky='se')
        self.delete_label=Button(self.root,text='Delete',command=self.popupWindow_delete,bg=self.root_bg,fg=self.root_fg,font=('Arial',12))
        self.delete_label.grid(row=2,column=5,sticky='se')

            

    def select_file_script(self):
        filetypes=(('sh','*.sh'),('All files','*.*'))
        initialdir_i=os.getcwd()
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=initialdir_i,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print(self.filename)
            f=open(self.list_script_path,'r')
            f_old=f.readlines()
            f.close()
            if len(f_old)>0:
                print('length >0')
                f_old.append(self.filename+'\n')
                f_new=f_old
            else:
                f_old=['None']
                f_old.append(self.filename+'\n')
                f_new=f_old
            print(f_new)
            print(len(f_new))
            print('------')
            f_new_dic={path_i:i for i,path_i in enumerate(f_new) if path_i.find('.sh')!=-1}
            f_new_list=list(f_new_dic.keys())
            f=open(self.list_script_path,'w')
            tmp=[f.writelines(w) for w in f_new_list]
            f.close()
            self.dropdown_menu()
        showinfo(title='Selected File',
                 message=self.filename)
    def dropdown_menu(self):
        if self.dropdown!=None:
            self.dropdown_label.destroy()
            self.dropdown.destroy()

   
        self.USER_SELECTION=tk.StringVar()
        if self.USER in self.SETTINGS_FILE_LIST:
            self.USER_SELECTION.set(self.USER)
        else:
            self.USER_SELECTION.set(self.SETTINGS_FILE_LIST[0])
        self.dropdown=tk.OptionMenu(self.root,self.USER_SELECTION,*self.SETTINGS_FILE_LIST)
        self.dropdown.grid(row=1,column=9,sticky='sw')
        
        self.dropdown_label=Button(self.root,image=self.icon_single_file,command=self.run_cmd_libs,bg=self.root_bg,fg=self.root_fg,font=('Arial',12))
        self.dropdown_label.grid(row=1,column=8,sticky='sw')
    
    def run_cmd_libs(self):
        cmd_i=open_cmd+" {}.py".format(os.path.join('libs',self.USER_SELECTION.get()))
        os.system(cmd_i)

    def run_cmd_editpaths(self):
        cmd_i=open_cmd+" {}".format(self.list_script_path)
        os.system(cmd_i)
        
    def popupWindow_delete(self):

        self.USER=self.USER_SELECTION.get()
        self.USER=self.USER.strip()
        get_default_settings(self.USER)
        SAVED_SETTINGS_PATH=os.path.join('libs',self.USER)
        if os.path.exists(SAVED_SETTINGS_PATH+".py"):
            f=open(SAVED_SETTINGS_PATH+".py")
            f_read=f.readlines()
            f.close()
            path_of_interest='None'
            for line in f_read:
                if line.find('YOLO_MODEL_PATH')!=-1:                   
                    path_of_interest=line.split('=')[1].replace('\n','').strip().replace('r"','"').replace("r'","'").replace("'","").replace('"',"")


        if path_of_interest!='None':
            delete_prompt="{}.py \n and \n {}".format(SAVED_SETTINGS_PATH,path_of_interest)
        else:
            delete_prompt="{}.py".format(SAVED_SETTINGS_PATH)
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//2.0),int(self.root.winfo_screenheight()*0.95//6.0)) )
        self.top.configure(background = 'black')
        self.top.title('Are you sure you want to delete all associated files with')
        self.delete_note2=tk.Label(self.top,text='{}'.format(delete_prompt),bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.delete_note2.grid(row=0,column=0,columnspan=5,sticky='ne')
        self.b=Button(self.top,text='Yes',command=self.delete,bg=self.root_fg, fg=self.root_bg)
        self.b.grid(row=4,column=1,sticky='se')
        self.c=Button(self.top,text='No',command=self.cleanup,bg=self.root_bg, fg=self.root_fg)
        self.c.grid(row=4,column=2,sticky='se')

    def delete(self):
        self.USER=self.USER_SELECTION.get()
        self.USER=self.USER.strip()
        get_default_settings(self.USER)
        SAVED_SETTINGS_PATH=os.path.join('libs',self.USER)
        if os.path.exists(SAVED_SETTINGS_PATH+".py"):
            f=open(SAVED_SETTINGS_PATH+".py")
            f_read=f.readlines()
            f.close()
            for line in f_read:
                if line.find('YOLO_MODEL_PATH')!=-1:
                    path_of_interest=line.split('=')[1].replace('\n','').strip().replace('r"','"').replace("r'","'").replace("'","").replace('"',"")
                    print('Found YOLO_MODEL_PATH: \n{} \n'.format(path_of_interest))
                    if os.path.exists(path_of_interest):
                        os.system('rm -rf {}'.format(path_of_interest))
                        print('Deleted all files located at {}'.format(path_of_interest))
        os.remove(SAVED_SETTINGS_PATH+".py")
        print('Deleted SAVED_SETTINGS_PATH: \n',SAVED_SETTINGS_PATH)
        self.cleanup()
        self.INITIAL_CHECK()

    def submit(self):
        global SAVED_SETTINGS_PATH
        global PROCEED 
        PROCEED=True
        self.USER=self.USER_SELECTION.get()
        self.USER=self.USER.strip()
        get_default_settings(self.USER)
        SAVED_SETTINGS_PATH=os.path.join('libs',self.USER)
        self.close()

    def get_update_background_img(self):
        self.image=Image.open(self.root_background_img)
        self.image=self.image.resize((self.root_W,self.root_H),Image.ANTIALIAS)
        self.bg=ImageTk.PhotoImage(self.image)
        self.canvas=tk.Canvas(self.root,width=self.root_W,height=self.root_H)
        self.canvas.grid(row=0,column=0,columnspan=self.canvas_columnspan,rowspan=self.canvas_rowspan,sticky='nw')
        self.canvas.create_image(0,0,image=self.bg,anchor='nw')
    def close(self):
        self.root.destroy()
    def cleanup(self):
        self.top.destroy()

class classify_chips:
    def __init__(self,root_tk,SAVED_SETTINGS_PATH):
        self.PREFIX=DEFAULT_SETTINGS.PREFIX
        self.MODELS_PATH=os.path.abspath(DEFAULT_SETTINGS.MODELS_PATH)
        self.chipW=DEFAULT_SETTINGS.chipW
        self.chipH=DEFAULT_SETTINGS.chipH
        self.epochs=DEFAULT_SETTINGS.epochs
        self.batch_size=DEFAULT_SETTINGS.batch_size
        self.PORT=DEFAULT_SETTINGS.PORT
        self.HOST=DEFAULT_SETTINGS.HOST
        self.target=DEFAULT_SETTINGS.target
        self.TRAIN_TEST_SPLIT=DEFAULT_SETTINGS.TRAIN_TEST_SPLIT
        self.custom_model_path=os.path.abspath(DEFAULT_SETTINGS.custom_model_path)
        self.classes_path=os.path.abspath(DEFAULT_SETTINGS.classes_path)
        self.data_dir=os.path.abspath(DEFAULT_SETTINGS.data_dir)
        self.data_dir_test=os.path.abspath(DEFAULT_SETTINGS.data_dir_test)
        self.data_dir_train=os.path.abspath(DEFAULT_SETTINGS.data_dir_train)
        self.TRAIN=DEFAULT_SETTINGS.TRAIN
        self.LOAD_PREVIOUS=DEFAULT_SETTINGS.LOAD_PREVIOUS
        self.root_bg=DEFAULT_SETTINGS.root_bg#'black'
        self.root_fg=DEFAULT_SETTINGS.root_fg#'lime'
        self.canvas_columnspan=DEFAULT_SETTINGS.canvas_columnspan
        self.canvas_rowspan=DEFAULT_SETTINGS.canvas_rowspan
        self.root_background_img=DEFAULT_SETTINGS.root_background_img #r"misc/gradient_blue.jpg"
        self.ignore_list_str=DEFAULT_SETTINGS.ignore_list_str#'IGNORE;blank'#help='ignore these classes from predictions')
        self.print_every=DEFAULT_SETTINGS.print_every
        self.LEARNING_RATE=DEFAULT_SETTINGS.LEARNING_RATE
        self.SAVED_SETTINGS_PATH=SAVED_SETTINGS_PATH
        self.root=root_tk
        self.root.bind('<Escape>',self.close)
        self.root.wm_iconphoto(False,ImageTk.PhotoImage(Image.open("resources/icons/appC.png")))
        self.icon_breakup=ImageTk.PhotoImage(Image.open('resources/icons/breakup.png'))
        self.icon_folder=ImageTk.PhotoImage(Image.open("resources/icons/open_folder_search.png"))
        self.icon_single_file=ImageTk.PhotoImage(Image.open('resources/icons/single_file.png'))
        self.icon_load=ImageTk.PhotoImage(Image.open('resources/icons/load.png'))
        self.icon_create=ImageTk.PhotoImage(Image.open('resources/icons/create.png'))
        self.icon_analyze=ImageTk.PhotoImage(Image.open('resources/icons/analyze.png'))
        self.icon_move=ImageTk.PhotoImage(Image.open('resources/icons/move.png'))
        self.icon_labelImg=ImageTk.PhotoImage(Image.open('resources/icons/labelImg.png'))
        self.icon_map=ImageTk.PhotoImage(Image.open('resources/icons/map.png'))
        self.icon_merge=ImageTk.PhotoImage(Image.open('resources/icons/merge.png'))
        self.icon_clear_fix=ImageTk.PhotoImage(Image.open('resources/icons/clear.png'))
        self.icon_clear_checked=ImageTk.PhotoImage(Image.open('resources/icons/clear_checked.png'))
        self.icon_save_settings=ImageTk.PhotoImage(Image.open('resources/icons/save.png'))
        self.icon_yolo_objects=ImageTk.PhotoImage(Image.open('resources/icons/yolo_objects.png'))
        self.icon_divide=ImageTk.PhotoImage(Image.open('resources/icons/divide.png'))
        self.icon_scripts=ImageTk.PhotoImage(Image.open('resources/icons/scripts.png'))
        self.icon_config=ImageTk.PhotoImage(Image.open('resources/icons/config.png'))
        self.icon_open=ImageTk.PhotoImage(Image.open('resources/icons/open_folder.png'))
        self.icon_train=ImageTk.PhotoImage(Image.open('resources/icons/train.png'))
        self.icon_test_mp4=ImageTk.PhotoImage(Image.open('resources/icons/test_mp4.png'))
        self.icon_test_images=ImageTk.PhotoImage(Image.open('resources/icons/test_images.png'))
        self.icon_test=ImageTk.PhotoImage(Image.open('resources/icons/test.png'))
        self.icon_MOSAIC=ImageTk.PhotoImage(Image.open('resources/icons/appM_icon.png'))
        self.icon_IMGAUG=ImageTk.PhotoImage(Image.open('resources/icons/appI_icon.png'))    

        self.root_H=int(self.root.winfo_screenheight()*0.95)
        self.root_W=int(self.root.winfo_screenwidth()*0.95)
        self.root.geometry(str(self.root_W)+'x'+str(self.root_H))
        self.root.title(os.path.basename(self.SAVED_SETTINGS_PATH))

        self.get_update_background_img()
        self.root.configure(bg=self.root_bg)
        self.drop_targets=None
        self.CWD=os.getcwd()
        self.dropdown_menu_TENSORFLOW=None
        self.dropdown_menu_PYTORCH=None
    #Buttons
        self.save_settings_button=Button(self.root,image=self.icon_save_settings,command=self.save_settings,bg=self.root_bg,fg=self.root_fg)
        self.save_settings_button.grid(row=1,column=4,sticky='se')
        self.save_settings_note=tk.Label(self.root,text='Save Settings',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.save_settings_note.grid(row=2,column=4,sticky='ne')
        self.return_to_main_button=Button(self.root,text='Return to Main Menu',command=self.return_to_main,bg=self.root_fg,fg=self.root_bg)
        self.return_to_main_button.grid(row=0,column=0,sticky='se')

        self.OPEN_save_settings_button=Button(self.root,text="OPEN Saved Settings",command=self.OPEN_SAVED_SETTINGS,bg=self.root_fg,fg=self.root_bg)
        self.OPEN_save_settings_button.grid(row=1,column=0,sticky='se')
        self.LOAD_save_settings_button=Button(self.root,text="LOAD Saved Settings",command=self.LOAD_SAVED_SETTINGS,bg=self.root_fg,fg=self.root_bg)
        self.LOAD_save_settings_button.grid(row=2,column=0,sticky='se')


        self.PREFIX_VAR=tk.StringVar()
        self.PREFIX_VAR.set(self.PREFIX)
        self.PREFIX_entry=tk.Entry(self.root,textvariable=self.PREFIX_VAR)
        self.PREFIX_entry.grid(row=3,column=0,columnspan=5,sticky='sew')
        self.PREFIX_label=tk.Label(self.root,text='PREFIX',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.PREFIX_label.grid(row=4,column=0,sticky='ne')

        self.WIDTH_NUM_VAR=tk.StringVar()
        self.WIDTH_NUM_VAR.set(self.chipW)
        self.WIDTH_NUM_entry=tk.Entry(self.root,textvariable=self.WIDTH_NUM_VAR)
        self.WIDTH_NUM_entry.grid(row=5,column=0,sticky='se')
        self.WIDTH_NUM_label=tk.Label(self.root,text='chipW',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.WIDTH_NUM_label.grid(row=6,column=0,sticky='ne')

        self.HEIGHT_NUM_VAR=tk.StringVar()
        self.HEIGHT_NUM_VAR.set(self.chipH)
        self.HEIGHT_NUM_entry=tk.Entry(self.root,textvariable=self.HEIGHT_NUM_VAR)
        self.HEIGHT_NUM_entry.grid(row=7,column=0,sticky='se')
        self.HEIGHT_NUM_label=tk.Label(self.root,text='chipH',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.HEIGHT_NUM_label.grid(row=8,column=0,sticky='ne')

        self.EPOCHS_NUM_VAR=tk.StringVar()
        self.EPOCHS_NUM_VAR.set(self.epochs)
        self.EPOCHS_entry=tk.Entry(self.root,textvariable=self.EPOCHS_NUM_VAR)
        self.EPOCHS_entry.grid(row=9,column=0,sticky='se')
        self.EPOCHS_label=tk.Label(self.root,text='epochs',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.EPOCHS_label.grid(row=10,column=0,sticky='ne')

        self.BATCH_SIZE_VAR=tk.StringVar()
        self.BATCH_SIZE_VAR.set(self.batch_size)
        self.BATCH_SIZE_entry=tk.Entry(self.root,textvariable=self.BATCH_SIZE_VAR)
        self.BATCH_SIZE_entry.grid(row=11,column=0,sticky='se')
        self.BATCH_SIZE_label=tk.Label(self.root,text='batch_size',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.BATCH_SIZE_label.grid(row=12,column=0,sticky='ne')

        self.TRAIN_TEST_SPLIT_VAR=tk.StringVar()
        self.TRAIN_TEST_SPLIT_VAR.set(self.TRAIN_TEST_SPLIT)
        self.TRAIN_TEST_SPLIT_entry=tk.Entry(self.root,textvariable=self.TRAIN_TEST_SPLIT_VAR)
        self.TRAIN_TEST_SPLIT_entry.grid(row=13,column=0,sticky='se')
        self.TRAIN_TEST_SPLIT_label=tk.Label(self.root,text='TRAIN_TEST_SPLIT',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.TRAIN_TEST_SPLIT_label.grid(row=14,column=0,sticky='ne')

        self.PORT_VAR=tk.StringVar()
        self.PORT_VAR.set(self.PORT)
        self.PORT_entry=tk.Entry(self.root,textvariable=self.PORT_VAR)
        self.PORT_entry.grid(row=15,column=0,sticky='se')
        self.PORT_label=tk.Label(self.root,text='PORT',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.PORT_label.grid(row=16,column=0,sticky='ne')

        self.HOST_VAR=tk.StringVar()
        self.HOST_VAR.set(self.HOST)
        self.HOST_entry=tk.Entry(self.root,textvariable=self.HOST_VAR)
        self.HOST_entry.grid(row=17,column=0,sticky='se')
        self.HOST_label=tk.Label(self.root,text='HOST',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.HOST_label.grid(row=18,column=0,sticky='ne')

        self.dropdown_menu_TENSORFLOW_FUNC()

        self.dropdown_menu_PYTORCH_FUNC()


        try:
            s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            s.connect(("8.8.8.8",80))
            self.IP_ADDRESS=s.getsockname()[0]
        except:
            self.IP_ADDRESS="127.0.0.1"
        if self.IP_ADDRESS!=self.HOST:
            self.HOST_note=tk.Label(self.root,text=f'WARNING! LOCAL IP ADDRESS is {self.IP_ADDRESS}',bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
            self.HOST_note.grid(row=17,column=1,sticky='e')
        else:
            self.HOST_note=tk.Label(self.root,text=f'Same as LOCAL IP ADDRESS of {self.IP_ADDRESS}',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.HOST_note.grid(row=17,column=1,sticky='e')           

        self.open_MODELS_PATH_label_var=tk.StringVar()
        self.open_MODELS_PATH_label_var.set(self.MODELS_PATH)

        self.open_DATA_DIR_PATH_label_var=tk.StringVar()
        self.open_DATA_DIR_PATH_label_var.set(self.data_dir)

        self.open_DATA_DIR_TRAIN_PATH_label_var=tk.StringVar()
        self.open_DATA_DIR_TRAIN_PATH_label_var.set(self.data_dir_train)

        self.open_DATA_DIR_TEST_PATH_label_var=tk.StringVar()
        self.open_DATA_DIR_TEST_PATH_label_var.set(self.data_dir_test)

        self.style3=ttk.Style()
        self.style3.configure('Normal.TRadiobutton',
                             background=self.root_bg,
                             foreground=self.root_fg)
        try:
            self.TRAIN_TRUE_button.destroy()
        except:
            pass
        try:
            self.TRAIN_FALSE_button.destroy()
        except:
            pass
        self.TRAIN_VAR=tk.StringVar()
        self.TRAIN_VAR.set(self.TRAIN)
        self.TRAIN_TRUE_button=ttk.Radiobutton(text='TRAIN',style='Normal.TRadiobutton',variable=self.TRAIN_VAR,value=1,command=partial(self.selection,self.TRAIN_VAR))
        self.TRAIN_TRUE_button.grid(row=1,column=1,stick='se')

        self.TRAIN_FALSE_button=ttk.Radiobutton(text='TEST',style='Normal.TRadiobutton',variable=self.TRAIN_VAR,value=0,command=partial(self.selection,self.TRAIN_VAR))
        self.TRAIN_FALSE_button.grid(row=2,column=1,stick='ne')

        self.CREATE_BUTTONS()


    def dropdown_menu_TENSORFLOW_FUNC(self):
        if self.dropdown_menu_TENSORFLOW!=None:
            self.dropdown_label_TENSORFLOW.destroy()
            self.dropdown_TENSORFLOW.destroy()

   
        self.USER_SELECTION_TENSORFLOW=tk.StringVar()
        self.TENSORFLOW_OPTIONS=['imagenet_resnet_v1_101_feature_vector_5','tf2-preview_mobilenet_v2_feature_vector_4']

        self.USER_SELECTION_TENSORFLOW.set(self.TENSORFLOW_OPTIONS[0])
        self.dropdown_TENSORFLOW=tk.OptionMenu(self.root,self.USER_SELECTION_TENSORFLOW,*self.TENSORFLOW_OPTIONS)
        self.dropdown_TENSORFLOW.grid(row=10,column=6,sticky='nw')
        
        self.dropdown_label_TENSORFLOW=tk.Label(self.root,text="TENSORFLOW MODEL OPTIONS",bg=self.root_bg,fg=self.root_fg,font=('Arial',8))
        self.dropdown_label_TENSORFLOW.grid(row=9,column=6,sticky='sw')

    def dropdown_menu_PYTORCH_FUNC(self):
        if self.dropdown_menu_PYTORCH!=None:
            self.dropdown_label_PYTORCH.destroy()
            self.dropdown_PYTORCH.destroy()

   
        self.USER_SELECTION_PYTORCH=tk.StringVar()
        self.PYTORCH_OPTIONS=['resnet_50']

        self.USER_SELECTION_PYTORCH.set(self.PYTORCH_OPTIONS[0])
        self.dropdown_PYTORCH=tk.OptionMenu(self.root,self.USER_SELECTION_PYTORCH,*self.PYTORCH_OPTIONS)
        self.dropdown_PYTORCH.grid(row=10,column=5,sticky='nw')
        
        self.dropdown_label_PYTORCH=tk.Label(self.root,text="PYTORCH MODEL OPTIONS",bg=self.root_bg,fg=self.root_fg,font=('Arial',8))
        self.dropdown_label_PYTORCH.grid(row=9,column=5,sticky='sw')

    def selection(self,var_i):
        print(f'You selected {var_i.get()}')
        print('self.TRAIN_VAR.get()',self.TRAIN_VAR.get())
        if str(self.TRAIN_VAR.get())=="False" or str(self.TRAIN_VAR.get())=='0':
            self.TRAIN=False
            self.TRAIN_VAR.set(False)

        else:
            self.TRAIN=True
            self.TRAIN_VAR.set(True)


    def get_targets(self):
        try:
            for k,v in self.targets_label_dic.items():
                v.destroy()
        except:
            pass
        try:
            for k,v in self.targets_label_count_dic.items():
                v.destroy()
        except:
            pass
        self.targets_label_dic={}
        self.targets_label_count_dic={}
        self.targets_count_dic={}
        if os.path.exists(self.data_dir_train):
            targets=os.listdir(self.data_dir_train)
            self.targets_list=[w for w in targets if os.path.isdir(os.path.join(self.data_dir_train,w))]
            self.target=";".join([w for w in self.targets_list])
        else:
            self.targets_list=[w for w in self.target.split(';') if len(w)>0]
        
        target_items=[w for w in self.target.split(";") if len(w)>0]
        print('Found a total of {} classes:'.format(len(target_items)))
        j=10
        k=10
        self.targets_label_dic[j]=tk.Label(self.root,text='CLASS',bg=self.root_fg,fg=self.root_bg,font=("Arial", 10))
        self.targets_label_dic[j].grid(row=j,column=5+k,sticky='se')
        self.targets_label_count_dic[j]=tk.Label(self.root,text='COUNT',bg=self.root_fg,fg=self.root_bg,font=("Arial", 10))
        self.targets_label_count_dic[j].grid(row=j,column=6+k,sticky='se')  
        j+=1
        for  i,item_i in enumerate(target_items):
                i=i+j
                self.targets_label_dic[i]=tk.Label(self.root,text=item_i,bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
                self.targets_label_dic[i].grid(row=i+1,column=5+k,sticky='ne')
                count_i=len(os.listdir(os.path.join(self.data_dir_train,item_i)))
                self.targets_count_dic[i]=count_i
                self.targets_label_count_dic[i]=tk.Label(self.root,text=self.targets_count_dic[i],bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
                self.targets_label_count_dic[i].grid(row=i+1,column=6+k,sticky='ne')               
                i+=1
                print(f' {i} = {item_i}')
        self.TARGET_VAR.set(self.target)
    def dropdown_menu_targets(self):
        self.get_targets()
        try:
            self.dropdown_targets.destroy()
        except:
            pass
        self.USER_SELECTION_TARGETS=tk.StringVar()
        self.USER_SELECTION_TARGETS.set(self.targets_list[0])
        #self.dropdown_targets=tk.OptionMenu(self.root,self.USER_SELECTION_TARGETS,*self.targets_list)
        #self.dropdown_targets.grid(row=19,column=1,sticky='sw')
        

    def CREATE_BUTTONS(self):
        try:
            self.open_MODELS_PATH_label.destroy()
        except:
            pass
        try:
            self.open_MODELS_PATH_button.destroy()
        except:
            pass

        self.open_MODELS_PATH_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.MODELS_PATH,'save path',self.open_MODELS_PATH_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_MODELS_PATH_button.grid(row=1,column=5,sticky='se')
        self.open_MODELS_PATH_note=tk.Label(self.root,text="MODEL PATH dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_MODELS_PATH_note.grid(row=2,column=5,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_MODELS_PATH_label_var.get())
        self.open_MODELS_PATH_label=Button(self.root,textvariable=self.open_MODELS_PATH_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_MODELS_PATH_label.grid(row=1,column=6,columnspan=50,sticky='sw')

        try:
            self.open_DATA_DIR_PATH_label.destroy()
        except:
            pass
        try:
            self.open_DATA_DIR_PATH_button.destroy()
        except:
            pass

        self.open_DATA_DIR_PATH_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.data_dir,'save path',self.open_DATA_DIR_PATH_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_DATA_DIR_PATH_button.grid(row=3,column=5,sticky='se')
        self.open_DATA_DIR_PATH_note=tk.Label(self.root,text="data dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_DATA_DIR_PATH_note.grid(row=4,column=5,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_DATA_DIR_PATH_label_var.get())
        self.open_DATA_DIR_PATH_label=Button(self.root,textvariable=self.open_DATA_DIR_PATH_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_DATA_DIR_PATH_label.grid(row=3,column=6,columnspan=50,sticky='sw')

        try:
            self.open_DATA_DIR_TRAIN_PATH_label.destroy()
        except:
            pass
        try:
            self.open_DATA_DIR_TRAIN_PATH_button.destroy()
        except:
            pass

        self.open_DATA_DIR_TRAIN_PATH_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.data_dir_train,'save path',self.open_DATA_DIR_TRAIN_PATH_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_DATA_DIR_TRAIN_PATH_button.grid(row=5,column=5,sticky='se')
        self.open_DATA_DIR_TRAIN_PATH_note=tk.Label(self.root,text="data dir train",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_DATA_DIR_TRAIN_PATH_note.grid(row=6,column=5,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_DATA_DIR_TRAIN_PATH_label_var.get())
        self.open_DATA_DIR_TRAIN_PATH_label=Button(self.root,textvariable=self.open_DATA_DIR_TRAIN_PATH_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_DATA_DIR_TRAIN_PATH_label.grid(row=5,column=6,columnspan=50,sticky='sw')

        try:
            self.open_DATA_DIR_TEST_PATH_label.destroy()
        except:
            pass
        try:
            self.open_DATA_DIR_TEST_PATH_button.destroy()
        except:
            pass

        self.open_DATA_DIR_TEST_PATH_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.data_dir_test,'save path',self.open_DATA_DIR_TEST_PATH_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_DATA_DIR_TEST_PATH_button.grid(row=7,column=5,sticky='se')
        self.open_DATA_DIR_TEST_PATH_note=tk.Label(self.root,text="data dir test",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_DATA_DIR_TEST_PATH_note.grid(row=8,column=5,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_DATA_DIR_TEST_PATH_label_var.get())
        self.open_DATA_DIR_TEST_PATH_label=Button(self.root,textvariable=self.open_DATA_DIR_TEST_PATH_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_DATA_DIR_TEST_PATH_label.grid(row=7,column=6,columnspan=50,sticky='sw')

        try:
            self.CREATE_SCRIPTS_button.destroy()
        except:
            pass

        self.CREATE_SCRIPTS_button=Button(self.root,text='CREATE SCRIPTS',command=self.CREATE_SCRIPTS,bg=self.root_bg,fg=self.root_fg)
        self.CREATE_SCRIPTS_button.grid(row=10,column=3,sticky='se')


        self.TARGET_VAR=tk.StringVar()
        self.TARGET_VAR.set(self.target)
        self.TARGET_entry=tk.Entry(self.root,textvariable=self.TARGET_VAR)
        self.TARGET_entry.grid(row=19,column=0,columnspan=10,sticky='ew')
        self.TARGET_label=tk.Label(self.root,text='targets',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.TARGET_label.grid(row=20,column=0,sticky='ne')
        self.dropdown_menu_targets()

        self.IGNORE_VAR=tk.StringVar()
        self.IGNORE_VAR.set(self.ignore_list_str)
        self.IGNORE_entry=tk.Entry(self.root,textvariable=self.IGNORE_VAR)
        self.IGNORE_entry.grid(row=21,column=0,columnspan=10,sticky='ew')
        self.IGNORE_label=tk.Label(self.root,text='ignore_list_str',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.IGNORE_label.grid(row=22,column=0,sticky='ne')
        self.check_ignore()

    def check_ignore(self):
        #print('Make sure your IGNORE list is separated by ";" semicolons between items.')
        tmp_ignore=self.IGNORE_VAR.get()
        if tmp_ignore.find(';')==-1:
            print("WARNING!  Did not see any semicolons")
        else:
            ignore_items=[w for w in tmp_ignore.split(";") if len(w)>0]
            print('Found a total of {} items to ignore:'.format(len(ignore_items)))
            for  i,item_i in enumerate(ignore_items):
                i+=1
                print(f' {i} = {item_i}')
        self.ignore_list_str=tmp_ignore
        self.IGNORE_VAR.set(self.ignore_list_str)
    
    def OPEN_SAVED_SETTINGS(self):
        if self.SAVED_SETTINGS_PATH.find('.py')==-1:
            open_path=self.SAVED_SETTINGS_PATH+'.py'
        else:
            open_path=self.SAVED_SETTINGS_PATH
        cmd_i=open_cmd+" {}".format(open_path)
        Thread(target=self.run_cmd,args=(cmd_i,)).start()
    
    def LOAD_SAVED_SETTINGS(self):
        global RELOAD,return_to_main
        RELOAD=True
        return_to_main=True
        self.root.destroy()

    def TRAIN_BUTTONS(self):
        cmd_i=f"bash {self.TRAIN_SCRIPT_PATH}"
        self.popup_TRAIN_button=Button(self.root,text='TRAIN PYTORCH',command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg)
        self.popup_TRAIN_button.grid(row=11,column=5,sticky='sw')

    def TEST_BUTTONS(self):
        cmd_i=f"bash {self.TEST_SCRIPT_PATH}"
        self.popup_TEST_button=Button(self.root,text='TEST PYTORCH',command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg)
        self.popup_TEST_button.grid(row=12,column=5,sticky='sw')

    def TRAIN_BUTTONS_TENSORFLOW(self):
        cmd_i=f"bash {self.TRAIN_SCRIPT_PATH_TENSORFLOW}"
        self.popup_TRAIN_button_TENSORFLOW=Button(self.root,text='TRAIN TENSORFLOW',command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg)
        self.popup_TRAIN_button_TENSORFLOW.grid(row=11,column=6,sticky='sw')

    def TEST_BUTTONS_TENSORFLOW(self):
        cmd_i=f"bash {self.TEST_SCRIPT_PATH_TENSORFLOW}"
        self.popup_TEST_button_TENSORFLOW=Button(self.root,text='TEST TENSORFLOW',command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg)
        self.popup_TEST_button_TENSORFLOW.grid(row=12,column=6,sticky='sw')

    def INFERENCE_BUTTONS(self):
        cmd_i=f"bash {self.INFERENCE_SCRIPT_PATH}"
        self.popup_INFERENCE_button=Button(self.root,text='INFERENCE PYTORCH',command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg)
        self.popup_INFERENCE_button.grid(row=13,column=5,sticky='sw')

    def INFERENCE_BUTTONS_TENSORFLOW(self):
        cmd_i=f"bash {self.INFERENCE_SCRIPT_PATH_TENSORFLOW}"
        self.popup_INFERENCE_button_TENSORFLOW=Button(self.root,text='INFERENCE TENSORFLOW',command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg)
        self.popup_INFERENCE_button_TENSORFLOW.grid(row=13,column=6,sticky='sw')



    def return_to_main(self):
        global return_to_main
        return_to_main=True
        self.root.destroy()


                 
    def select_folder(self,folder_i,title_i,var_i=None):
            filetypes=(('All files','*.*'))
            if var_i:
                folder_i=var_i.get()
            if os.path.exists(folder_i):
                self.foldername=fd.askdirectory(title=title_i,
                                            initialdir=folder_i)
            else:
                self.foldername=fd.askdirectory(title=title_i)
            if self.foldername=='' or len(self.foldername)==0:
                showinfo(title='NOT FOUND! Using previous path',
                        message=self.foldername)
            elif self.foldername!='' and len(self.foldername)!=0:
                showinfo(title='Selected Folder',
                    message=self.foldername)
                folder_i=self.foldername
                folder_i=os.path.abspath(folder_i)
                var_i.set(folder_i)
                self.CREATE_BUTTONS()


    def CREATE_SCRIPTS(self):
        
        self.save_settings()
        shutil.copy(self.SAVED_SETTINGS_PATH,self.MODELS_PATH)
        SAVED_SETTINGS_PATH=os.path.join(self.MODELS_PATH,os.path.basename(self.SAVED_SETTINGS_PATH))
        for PYTORCH_MODEL in self.PYTORCH_OPTIONS:
            #TRAIN
            self.TRAIN_SCRIPT_PATH=os.path.join(self.MODELS_PATH,'TRAIN_PYTORCH_{}.sh'.format(PYTORCH_MODEL))
            self.TRAIN_PATH=os.path.abspath('resources')
            f=open(self.TRAIN_SCRIPT_PATH,'w')
            f.writelines(f'cd {self.TRAIN_PATH}\n')
            f.writelines(f'python3 PYTORCH_MULTICLASS.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{PYTORCH_MODEL}" --TRAIN\n')
            f.close()

            #TEST
            self.TEST_SCRIPT_PATH=os.path.join(self.MODELS_PATH,'TEST_PYTORCH_{}.sh'.format(PYTORCH_MODEL))
            self.TEST_PATH=os.path.abspath('resources')
            f=open(self.TEST_SCRIPT_PATH,'w')
            f.writelines(f'cd {self.TEST_PATH}\n')
            f.writelines(f'python3 PYTORCH_MULTICLASS.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{PYTORCH_MODEL}"\n')
            f.close()

            #INFERENCE
            self.INFERENCE_SCRIPT_PATH=os.path.join(self.MODELS_PATH,'INFERENCE_PYTORCH_{}.sh'.format(PYTORCH_MODEL))
            self.INFERENCE_PATH=os.path.abspath('resources')
            f=open(self.INFERENCE_SCRIPT_PATH,'w')
            f.writelines(f'cd {self.INFERENCE_PATH}\n')
            f.writelines(f'python3 INFERENCE_PYTORCH.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{PYTORCH_MODEL}"\n')
            f.close()

        for TENSORFLOW_MODEL in self.TENSORFLOW_OPTIONS:
            #TRAIN TFLITE
            self.TRAIN_SCRIPT_PATH_TENSORFLOW=os.path.join(self.MODELS_PATH,'TRAIN_TENSORFLOW_{}.sh'.format(TENSORFLOW_MODEL))
            self.TRAIN_PATH_TENSORFLOW=os.path.abspath('resources')
            f=open(self.TRAIN_SCRIPT_PATH_TENSORFLOW,'w')
            f.writelines(f'cd {self.TRAIN_PATH_TENSORFLOW}\n')
            f.writelines(f'python3 TENSORFLOW_MULTICLASS.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{TENSORFLOW_MODEL}" --TRAIN\n')
            f.close()

            #TEST TFLITE
            self.TEST_SCRIPT_PATH_TENSORFLOW=os.path.join(self.MODELS_PATH,'TEST_TENSORFLOW_{}.sh'.format(TENSORFLOW_MODEL))
            self.TEST_PATH_TENSORFLOW=os.path.abspath('resources')
            f=open(self.TEST_SCRIPT_PATH_TENSORFLOW,'w')
            f.writelines(f'cd {self.TEST_PATH_TENSORFLOW}\n')
            f.writelines(f'python3 TENSORFLOW_MULTICLASS.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{TENSORFLOW_MODEL}"\n')
            f.close()

            #INFERENCE TENSORFLOW
            self.INFERENCE_SCRIPT_PATH_TENSORFLOW=os.path.join(self.MODELS_PATH,'INFERENCE_TENSORFLOW_{}.sh'.format(TENSORFLOW_MODEL))
            self.INFERENCE_PATH_TENSORFLOW=os.path.abspath('resources')
            f=open(self.INFERENCE_SCRIPT_PATH_TENSORFLOW,'w')
            f.writelines(f'cd {self.INFERENCE_PATH_TENSORFLOW}\n')
            f.writelines(f'python3 INFERENCE_TENSORFLOW.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{TENSORFLOW_MODEL}"\n')
            f.close()

        #TRAIN
        self.TRAIN_SCRIPT_PATH=os.path.join(self.MODELS_PATH,'TRAIN_PYTORCH.sh')
        self.TRAIN_PATH=os.path.abspath('resources')
        f=open(self.TRAIN_SCRIPT_PATH,'w')
        f.writelines(f'cd {self.TRAIN_PATH}\n')
        f.writelines(f'python3 PYTORCH_MULTICLASS.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{self.USER_SELECTION_PYTORCH.get()}" --TRAIN\n')
        f.close()

        #TRAIN TFLITE
        self.TRAIN_SCRIPT_PATH_TENSORFLOW=os.path.join(self.MODELS_PATH,'TRAIN_TENSORFLOW.sh')
        self.TRAIN_PATH_TENSORFLOW=os.path.abspath('resources')
        f=open(self.TRAIN_SCRIPT_PATH_TENSORFLOW,'w')
        f.writelines(f'cd {self.TRAIN_PATH_TENSORFLOW}\n')
        f.writelines(f'python3 TENSORFLOW_MULTICLASS.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{self.USER_SELECTION_TENSORFLOW.get()}" --TRAIN\n')
        f.close()

        #TEST
        self.TEST_SCRIPT_PATH=os.path.join(self.MODELS_PATH,'TEST_PYTORCH.sh')
        self.TEST_PATH=os.path.abspath('resources')
        f=open(self.TEST_SCRIPT_PATH,'w')
        f.writelines(f'cd {self.TEST_PATH}\n')
        f.writelines(f'python3 PYTORCH_MULTICLASS.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{self.USER_SELECTION_PYTORCH.get()}"\n')
        f.close()

        #TEST TFLITE
        self.TEST_SCRIPT_PATH_TENSORFLOW=os.path.join(self.MODELS_PATH,'TEST_TENSORFLOW.sh')
        self.TEST_PATH_TENSORFLOW=os.path.abspath('resources')
        f=open(self.TEST_SCRIPT_PATH_TENSORFLOW,'w')
        f.writelines(f'cd {self.TEST_PATH_TENSORFLOW}\n')
        f.writelines(f'python3 TENSORFLOW_MULTICLASS.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{self.USER_SELECTION_TENSORFLOW.get()}"\n')
        f.close()

        #INFERENCE
        self.INFERENCE_SCRIPT_PATH=os.path.join(self.MODELS_PATH,'INFERENCE_PYTORCH.sh')
        self.INFERENCE_PATH=os.path.abspath('resources')
        f=open(self.INFERENCE_SCRIPT_PATH,'w')
        f.writelines(f'cd {self.INFERENCE_PATH}\n')
        f.writelines(f'python3 INFERENCE_PYTORCH.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{self.USER_SELECTION_PYTORCH.get()}"\n')
        f.close()

        #INFERENCE TENSORFLOW
        self.INFERENCE_SCRIPT_PATH_TENSORFLOW=os.path.join(self.MODELS_PATH,'INFERENCE_TENSORFLOW.sh')
        self.INFERENCE_PATH_TENSORFLOW=os.path.abspath('resources')
        f=open(self.INFERENCE_SCRIPT_PATH_TENSORFLOW,'w')
        f.writelines(f'cd {self.INFERENCE_PATH_TENSORFLOW}\n')
        f.writelines(f'python3 INFERENCE_TENSORFLOW.py --SETTINGS_PATH="{SAVED_SETTINGS_PATH}" --MODEL_TYPE="{self.USER_SELECTION_TENSORFLOW.get()}"\n')
        f.close()


        print('self.TRAIN_VAR.get()',self.TRAIN_VAR.get())
        if str(self.TRAIN_VAR.get())=="False" or str(self.TRAIN_VAR.get())=='0':
            self.TRAIN=False
            self.TRAIN_VAR.set(False)
            print("TEST SCRIPTS")
            self.TEST_BUTTONS()
            self.TEST_BUTTONS_TENSORFLOW()
            self.INFERENCE_BUTTONS()
            self.INFERENCE_BUTTONS_TENSORFLOW()
        else:
            print("TRAIN SCRIPTS")
            self.TRAIN=True
            self.TRAIN_VAR.set(True)
            self.TRAIN_BUTTONS()
            self.TRAIN_BUTTONS_TENSORFLOW()

    def run_thread_cmd(self,cmd_i):
        Thread(target=self.run_cmd,args=(cmd_i,)).start()

    def show_table(self):
        self.app = TestApp(self.top, self.df_filename_csv)
        self.app.pack(fill=tk.BOTH, expand=1)
               

    def popupWindow_ERROR(self,message):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.5),int(self.root.winfo_screenheight()*0.95//1.5)) )
        self.top.title('ERROR')
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=0,column=0,sticky='se')
        self.label_Error=tk.Label(self.top,text=message,bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.label_Error.grid(row=1,column=1,sticky='s')


    def close(self,event):
        self.root.destroy()

    def save_settings(self,save_root='libs'):
    
        self.data_dir=self.open_DATA_DIR_PATH_label_var.get()
        self.data_dir_train=self.open_DATA_DIR_TRAIN_PATH_label_var.get()
        self.data_dir_test=self.open_DATA_DIR_TEST_PATH_label_var.get()
        self.PREFIX=self.PREFIX_VAR.get()
        self.chipW=int(self.WIDTH_NUM_VAR.get())
        self.chipH=int(self.HEIGHT_NUM_VAR.get())
        self.epochs=int(self.EPOCHS_NUM_VAR.get())
        self.batch_size=int(self.BATCH_SIZE_VAR.get())
        self.TRAIN_TEST_SPLIT=float(self.TRAIN_TEST_SPLIT_VAR.get())
        self.LEARNING_RATE=float(self.LEARNING_RATE)
        self.get_targets()
        self.target=self.TARGET_VAR.get()
        self.check_ignore()
        self.ignore_list_str=self.IGNORE_VAR.get()
        self.prefix_foldername=self.PREFIX_VAR.get()+"_chipW"+str(self.chipW)+"_chipH"+str(self.chipH)+"_classes"+str(1+self.target.count(";"))

        self.MODELS_PATH=self.open_MODELS_PATH_label_var.get()
        if self.MODELS_PATH.find(self.prefix_foldername)==-1:
            self.MODELS_PATH=os.path.join(self.MODELS_PATH,self.prefix_foldername)
            if os.path.exists(self.MODELS_PATH)==False:
                os.makedirs(self.MODELS_PATH)
            self.open_MODELS_PATH_label_var.set(self.MODELS_PATH)
        self.classes_path=os.path.abspath(os.path.join(self.MODELS_PATH,'PYTORCH_classes.txt'))
        PYTORCH_MODEL_PATH=self.USER_SELECTION_PYTORCH.get()
        self.custom_model_path=os.path.join(self.MODELS_PATH,'PYTORCH_'+PYTORCH_MODEL_PATH+'_CUSTOM.pth')
        
        if os.path.exists(os.path.join('libs','DEFAULT_SETTINGS.py')):
            f=open(os.path.join('libs','DEFAULT_SETTINGS.py'),'r')
            f_read=f.readlines()
            f.close()
            from libs import DEFAULT_SETTINGS as DS
            all_variables = dir(DS)
            all_real_variables=[]
            # Iterate over the whole list where dir( )
            # is stored.
            for name in all_variables:
                # Print the item if it doesn't start with '__'
                if not name.startswith('__'):
                    if name.find('path_prefix_volumes_one')==-1 and name.find('path_prefix_elements')==-1 and name.find('path_prefix_mount_mac')==-1 and name!='os':
                        all_real_variables.append(name)
            f_new=[]
            for prefix_i in all_real_variables:
                prefix_i_value=''
                try:
                    prefix_i_comb="self."+prefix_i
                    prefix_i_comb=prefix_i_comb.strip()
                   
                    prefix_i_value=eval(prefix_i_comb)
                    print(prefix_i_comb,prefix_i_value)
                except:
                    pass
                if prefix_i=='path_prefix':
                    pass
                elif prefix_i=='TRAIN':
                    prefix_i_value=self.TRAIN
                elif (prefix_i.lower().find('path')!=-1 or prefix_i.lower().find('background')!=-1):
                    prefix_i_value="r'"+prefix_i_value+"'"
                elif type(prefix_i_value).__name__.find('int')!=-1:
                    pass
                elif type(prefix_i_value).__name__.find('str')!=-1:
                    prefix_i_value="'"+prefix_i_value+"'"
                
                f_new.append(prefix_i+"="+str(prefix_i_value)+'\n')               
            prefix_save=_platform+'_'+self.prefix_foldername+'_SAVED_SETTINGS'
            f_new.append('MODEL_PATH=r"{}"\n'.format(os.path.join(self.MODELS_PATH,self.prefix_foldername)))
            f=open('{}.py'.format(os.path.join(save_root,prefix_save.replace('-','_'))),'w')
            wrote=[f.writelines(w) for w in f_new]
            f.close()
            self.SAVED_SETTINGS_PATH='{}.py'.format(os.path.join(save_root,prefix_save.replace('-','_')))
            self.SAVED_SETTINGS_PATH=os.path.abspath(self.SAVED_SETTINGS_PATH)
            self.root.title(os.path.basename(self.SAVED_SETTINGS_PATH))
        self.CREATE_BUTTONS()

    def get_update_background_img(self):
        self.image=Image.open(self.root_background_img)
        self.image=self.image.resize((self.root_W,self.root_H),Image.ANTIALIAS)
        self.bg=ImageTk.PhotoImage(self.image)
        self.canvas=tk.Canvas(self.root,width=self.root_W,height=self.root_H)
        self.canvas.grid(row=0,column=0,columnspan=self.canvas_columnspan,rowspan=self.canvas_rowspan,sticky='nw')
        self.canvas.create_image(0,0,image=self.bg,anchor='nw')

    def run_cmd(self,cmd_i):
        os.system(cmd_i)         

if __name__=='__main__':
    while return_to_main==True:
        return_to_main=False
        root_tk=tk.Tk()
        main_resnet=main_entry(root_tk)
        main_resnet.root.mainloop()
        del main_resnet
        if PROCEED==True:
            root_tk=tk.Tk()
            main_resnet=classify_chips(root_tk,SAVED_SETTINGS_PATH)
            main_resnet.root.mainloop()
            del main_resnet
            del root_tk
            if RELOAD==False:
                PROCEED=False
            else:
                RELOAD=False
            os.system('rm -rf __pycache__')
            os.system('rm -rf libs/__pycache__')



    






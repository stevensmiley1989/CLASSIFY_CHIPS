'''
TENSORFLOW_MULTICLASS
========
Created by Steven Smiley 8/27/2022

TENSORFLOW_MULTICLASS.py is a script that trains/tests a Tensorflow Image Classification model with on Chips.

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
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os
import argparse
import shutil
import os


default_settings=r"../libs/DEFAULT_SETTINGS.py"
MODEL_MOBILENET=r"tf2-preview_mobilenet_v2_feature_vector_4"
OUTPUT_SHAPE_MOBILENET=1280
MODEL_RESNET=r"imagenet_resnet_v1_101_feature_vector_5"
OUTPUT_SHAPE_RESNET=2048
ap = argparse.ArgumentParser()
ap.add_argument("--SETTINGS_PATH",dest='SETTINGS_PATH',type=str,default=default_settings,help='path to SETTINGS.py')
ap.add_argument("--MODEL_TYPE",type=str,default=MODEL_RESNET,help='type of classifier to use')

global args
args = vars(ap.parse_args())
SETTINGS_PATH=args['SETTINGS_PATH']
MODEL_TYPE=args['MODEL_TYPE']
if MODEL_TYPE==MODEL_MOBILENET:
    OUTPUT_SHAPE=OUTPUT_SHAPE_MOBILENET
    path_MODEL=MODEL_MOBILENET
    chipH=224
    chipW=224
elif MODEL_TYPE==MODEL_RESNET:
    OUTPUT_SHAPE=OUTPUT_SHAPE_RESNET
    path_MODEL=MODEL_RESNET  
print('USING {}'.format(path_MODEL)) 
if os.path.exists(SETTINGS_PATH):
    if os.path.exists('tmp')==False:
        os.makedirs('tmp')
    CUSTOM_SETTINGS_PATH=os.path.join('tmp','CUSTOM_SETTINGS.py')
    shutil.copy(SETTINGS_PATH,CUSTOM_SETTINGS_PATH)
    from tmp.CUSTOM_SETTINGS import PORT,HOST,target,ignore_list_str,chipW,chipH,batch_size,custom_model_path,custom_model_path,classes_path,data_dir,TRAIN,LOAD_PREVIOUS,data_dir_test,data_dir_train
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
custom_model_path_SAVED=os.path.join(os.path.dirname(custom_model_path),'SAVED_'+os.path.basename(path_MODEL))
custom_model_path_TFLITE=os.path.join(os.path.dirname(custom_model_path),'TFLITE_'+os.path.basename(path_MODEL))
NEW_SAVED_MODEL=os.path.join(custom_model_path_SAVED,'CUSTOM'+os.path.basename(MODEL_TYPE))
TFLITE_MODEL = os.path.join(custom_model_path_TFLITE,'CUSTOM'+os.path.basename(MODEL_TYPE)+".tflite")
TFLITE_QUANT_MODEL = TFLITE_MODEL.replace(".tflite",'_quantized.tflite')
if os.path.exists(custom_model_path_SAVED)==False:
    os.makedirs(custom_model_path_SAVED)
if os.path.exists(NEW_SAVED_MODEL)==False:
     os.makedirs(NEW_SAVED_MODEL)
if os.path.exists(custom_model_path_TFLITE)==False:
     os.makedirs(custom_model_path_TFLITE)

if TRAIN:
    pass
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

print("Version: ", tf.__version__)
print("Hub version: ", hub.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
IMAGE_SHAPE = (chipW, chipH)
print('chipW',chipW)
print('chipH',chipH)
TRAINING_DATA_DIR = str(data_dir)
print(TRAINING_DATA_DIR);
datagen_kwargs = dict(rescale=1./255, validation_split=TRAIN_TEST_SPLIT)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    TRAINING_DATA_DIR, 
    subset="validation", 
    shuffle=True,
    target_size=IMAGE_SHAPE
)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR, 
    subset="training", 
    shuffle=True,
    target_size=IMAGE_SHAPE)

# Learn more about data batches
image_batch_train, label_batch_train = next(iter(train_generator))
print("Image batch shape: ", image_batch_train.shape)
print("Label batch shape: ", label_batch_train.shape)

dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)


model = tf.keras.Sequential([
  hub.KerasLayer(path_MODEL, 
                 output_shape=[OUTPUT_SHAPE],
                 trainable=False),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])
model.build([None, chipW, chipH, 3])
model.summary()
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])

# Run model training
#train_generator.batch_size=batch_size
#valid_generator.batch_size = batch_size
steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
val_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)
if TRAIN:
    classes_path=os.path.join(os.path.dirname(NEW_SAVED_MODEL),"classes.txt")
    f=open(classes_path,'w')
    tmp=[f.writelines(w+"\n") for w in dataset_labels]
    f.close()
    hist = model.fit(
        train_generator, 
        epochs=epochs,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=val_steps_per_epoch).history
    # Measure accuracy and loss after training

    final_loss, final_accuracy = model.evaluate(valid_generator, steps = val_steps_per_epoch)
    print("Final loss: {:.2f}".format(final_loss))
    print("Final accuracy: {:.2f}%".format(final_accuracy * 100))

    # Visualize training process

    plt.figure()
    plt.ylabel("Loss (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0,2])
    plt.plot(hist["loss"])
    plt.plot(hist["val_loss"])

    plt.figure()
    plt.ylabel("Accuracy (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0,1])
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])

    #NEW_SAVED_MODEL = "saved_models/my_custom_model"
    tf.keras.models.save_model(model, NEW_SAVED_MODEL)
    # Get the concrete function from the Keras model.
    run_model = tf.function(lambda x : custom_model(x))
    custom_model = tf.keras.models.load_model(NEW_SAVED_MODEL,
    custom_objects={'KerasLayer':hub.KerasLayer})
    # Save the concrete function.
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converted_tflite_model = converter.convert()

    open(TFLITE_MODEL, "wb").write(converted_tflite_model)

    # Convert the model to quantized version with post-training quantization
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    open(TFLITE_QUANT_MODEL, "wb").write(tflite_quant_model)




    # 
if TRAIN==False:
    custom_model = tf.keras.models.load_model(NEW_SAVED_MODEL,
    custom_objects={'KerasLayer':hub.KerasLayer})
    # Get images and labels batch from validation dataset generator

    val_image_batch, val_label_batch = next(iter(valid_generator))
    true_label_ids = np.argmax(val_label_batch, axis=-1)

    print("Validation batch shape:", val_image_batch.shape)
    tf_model_predictions = custom_model.predict(val_image_batch)
    print("Prediction results shape:", tf_model_predictions.shape)
    predicted_ids = np.argmax(tf_model_predictions, axis=-1)
    predicted_labels = dataset_labels[predicted_ids]
    plt.figure(figsize=(10,9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(val_label_batch.shape[0]):
      plt.subplot(1+int(np.sqrt(val_label_batch.shape[0])),1+int(np.sqrt(val_label_batch.shape[0])),n+1)
      plt.imshow(val_image_batch[n])
      color = "green" if predicted_ids[n] == true_label_ids[n] else "red"
      plt.title(predicted_labels[n].title(), color=color)
      plt.axis('off')
      plt.suptitle("Model predictions (green: correct, red: incorrect)")
    plt.show()

    tf_pred_dataframe = pd.DataFrame(tf_model_predictions)
    tf_pred_dataframe.columns = dataset_labels

    print("Prediction results for the first elements")
    print(tf_pred_dataframe.head())


    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
    print("Load TFLite model and see some details about input/output")

    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    print("== Input details ==")
    print("name:", input_details[0]['name'])
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])

    print("\n== Output details ==")
    print("name:", output_details[0]['name'])
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])
    print("valid_generator.batch_size",valid_generator.batch_size)
    tflite_interpreter.resize_tensor_input(input_details[0]['index'], (val_image_batch.shape[0], chipW, chipH, 3))
    tflite_interpreter.resize_tensor_input(output_details[0]['index'], (val_image_batch.shape[0], tf_model_predictions.shape[1])) #5 og
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    print("== Input details ==")
    print("name:", input_details[0]['name'])
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])

    print("\n== Output details ==")
    print("name:", output_details[0]['name'])
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])

    tflite_interpreter.set_tensor(input_details[0]['index'], val_image_batch)
    print(input_details[0]['index'])
    #tflite_interpreter.set_tensor(val_label_batch.shape[0],val_image_batch)

    tflite_interpreter.invoke()

    tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
    print("Prediction results shape:", tflite_model_predictions.shape)

    print("Convert prediction results to Pandas dataframe, for better visualization")

    tflite_pred_dataframe = pd.DataFrame(tflite_model_predictions)
    tflite_pred_dataframe.columns = dataset_labels

    print("TFLite prediction results for the first elements")
    print(tflite_pred_dataframe.head())

    # Load quantized TFLite model
    tflite_interpreter_quant = tf.lite.Interpreter(model_path=TFLITE_QUANT_MODEL)

    # Learn about its input and output details
    input_details = tflite_interpreter_quant.get_input_details()
    output_details = tflite_interpreter_quant.get_output_details()

    # Resize input and output tensors to handle batch of 32 images
    tflite_interpreter_quant.resize_tensor_input(input_details[0]['index'], (val_image_batch.shape[0], chipW, chipH, 3))
    tflite_interpreter_quant.resize_tensor_input(output_details[0]['index'], (val_image_batch.shape[0], tf_model_predictions.shape[1]))
    tflite_interpreter_quant.allocate_tensors()

    input_details = tflite_interpreter_quant.get_input_details()
    output_details = tflite_interpreter_quant.get_output_details()

    print("== Input details ==")
    print("name:", input_details[0]['name'])
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])

    print("\n== Output details ==")
    print("name:", output_details[0]['name'])
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])

    # Run inference
    tflite_interpreter_quant.set_tensor(input_details[0]['index'], val_image_batch)

    tflite_interpreter_quant.invoke()

    tflite_q_model_predictions = tflite_interpreter_quant.get_tensor(output_details[0]['index'])
    print("\nPrediction results shape:", tflite_q_model_predictions.shape)

    # Convert prediction results to Pandas dataframe, for better visualization

    tflite_q_pred_dataframe = pd.DataFrame(tflite_q_model_predictions)
    tflite_q_pred_dataframe.columns = dataset_labels

    print("Quantized TFLite model prediction results for the first elements")
    print(tflite_q_pred_dataframe.head())

    # Concatenate results from all models

    all_models_dataframe = pd.concat([tf_pred_dataframe, 
                                    tflite_pred_dataframe, 
                                    tflite_q_pred_dataframe], 
                                    keys=['TF Model', 'TFLite', 'TFLite quantized'],
                                    axis='columns')
    print(all_models_dataframe.head())

    # Swap columns to hava side by side comparison

    all_models_dataframe = all_models_dataframe.swaplevel(axis='columns')[tflite_pred_dataframe.columns]
    all_models_dataframe.head()
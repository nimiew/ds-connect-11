import os
import shutil
import math
import numpy as np
import tensorflow as tf
seed = 11
tf.set_random_seed(seed)
np.random.seed(seed)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator, save_img, load_img, array_to_img, img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.mobilenet_v2 import MobileNetV2
import matplotlib.pyplot as plt
from IPython.display import Image, display
import json

def preprocess(test_percentage, augment):
    base_dir = os.getcwd()
    train_dir = os.path.join(base_dir, "train")
    if os.path.exists(train_dir) == True:
        shutil.rmtree(train_dir, ignore_errors=False, onerror=None)
    if os.path.exists(train_dir) == False:
        os.makedirs(train_dir)

    test_dir = os.path.join(base_dir, "test")
    if os.path.exists(test_dir) == True:
        shutil.rmtree(test_dir, ignore_errors=False, onerror=None)    
    if os.path.exists(test_dir) == False:
        os.makedirs(test_dir)
    
    data_dir = os.path.join(base_dir, "data")
    category_names = os.listdir(data_dir)
    category_names.sort()

    # Create directories for each class in train_dir
    train_category_dirs = []
    for i in category_names:
        train_category_dirs.append(os.path.join(train_dir, i))
    for i in train_category_dirs:
        if os.path.exists(i) == False:
            os.makedirs(i)

    # Create directories for each class in test_dir
    test_category_dirs = []
    for i in category_names:
        test_category_dirs.append(os.path.join(test_dir, i))    
    for i in test_category_dirs:
        if os.path.exists(i) == False:
            os.makedirs(i)

    # Move images in each category to their respective train/test directories
    for category_name in category_names:
        category_path = os.path.join(data_dir, category_name)
        num_images = len(os.listdir(category_path))
        idx = 0
        num_training_samples = math.ceil(num_images * (1-test_percentage))
        for image_name in os.listdir(category_path):
            src_image_path = os.path.join(category_path, image_name)
            if idx < num_training_samples:
                dst_dir = os.path.join(train_dir, category_name)
            else:
                dst_dir = os.path.join(test_dir, category_name)
            shutil.copy(src_image_path, dst_dir)
            idx += 1
    
    # If augment, need to create augmented images and store them in aug_train directory
    if(augment):
        aug_train_dir = os.path.join(base_dir, "aug_train")
        if os.path.exists(aug_train_dir) == True:
            shutil.rmtree(aug_train_dir)
        if os.path.exists(aug_train_dir) == False:
            os.makedirs(aug_train_dir)
        aug_train_category_dirs = []
        for i in category_names:
            aug_train_category_dirs.append(os.path.join(aug_train_dir, i))
        for i in aug_train_category_dirs:
            if os.path.exists(i) == False:
                os.makedirs(i)
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        num_aug_samples = num_samples('train') * 10
        i = 0
        for batch in datagen.flow_from_directory(train_dir, target_size=(224,224), class_mode='categorical', batch_size=1):
            img_array = batch[0][0]
            dst_dir = os.path.join(aug_train_dir, category_names[np.argmax(batch[1][0])])
            dst_fname = os.path.join(dst_dir, str(i))+".jpg"
            save_img(dst_fname, img_array)
            i += 1
            if(i==num_aug_samples):
                break

                
def num_classes():
    return len(os.listdir("data"))


def create_basic_CNN():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(96, 96, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes(), activation='softmax'))
    return model


def compile(model):
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=0.001),
                metrics=['acc'])


def train(model, train_dir, validation_dir, num_epochs, save_name):
    batch_size = 16
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(96, 96),
            batch_size=batch_size,
            class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(96, 96),
            batch_size=batch_size,
            class_mode='categorical')

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(save_name, monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    history = model.fit_generator(
            generator=train_generator,
            epochs=num_epochs,
            steps_per_epoch=num_samples('train')/batch_size,
            validation_steps=num_samples('test')/batch_size,
            validation_data=validation_generator,
            callbacks=callbacks)
    
    return history


def evaluate(model, train_dir, validation_dir):
    batch_size = 16
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(96, 96),
            batch_size=batch_size,
            class_mode='categorical')
    print("Accuracy: {}".format(model.evaluate_generator(validation_generator,
                                steps=num_samples('test')/batch_size)[-1]))


def create_conv_base():
    conv_base = MobileNetV2(weights='imagenet',
                    include_top=False,
                    input_shape=(96, 96, 3))
    return conv_base


def num_samples(dir):
    num_sample = 0
    for subdir in os.listdir(dir):
        num_sample+=len(os.listdir(os.path.join(dir, subdir)))
    return num_sample


def extract_features(conv_base, directory, sample_count, augment=False):
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)

    if(augment):
        datagen = train_datagen
        sample_count *= 10
    else:
        datagen = test_datagen

    features = np.zeros(shape=(sample_count, 3, 3, 1280))
    labels = np.zeros(shape=(sample_count, num_classes()))

    batch_size = 1
    generator = datagen.flow_from_directory(
        directory,
        target_size=(96, 96),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break     
    return features, labels


def reshape_features(features):
    vector_size = 1
    dimensions = features.shape[1:] # First index is num_samples, the rest is dimension of sample. 
    for dimension in dimensions:
        vector_size *= dimension # "Flattening"
    return np.reshape(features, (features.shape[0],vector_size))


def create_MLP():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=3 * 3 * 1280))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes(), activation='softmax'))
    return model


def train_MLP(mlp, train_features, train_labels, validation_features, validation_labels, num_epochs, save_name):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(save_name, monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]
    mlp.fit(train_features, train_labels,
                    epochs=num_epochs,
                    batch_size=32,
                    validation_data=(validation_features, validation_labels),
                    callbacks=callbacks)


def evaluate_MLP(mlp, validation_features, validation_labels):
    print("Accuracy: {}".format(mlp.evaluate(validation_features, validation_labels)[1]))


def show_augment_image():
    category_names = os.listdir(os.path.join(os.getcwd(), "data"))
    fnames = []
    for category_name in category_names:
        fnames.extend([os.path.join('train', category_name, fname) for fname in os.listdir(os.path.join('train', category_name))])

    # We pick one image to "augment"
    img_path = fnames[3]

    # Read the image and resize it
    img = load_img(img_path, target_size=(96, 96))

    # Convert it to a numpy array with shape (96, 96, 3)
    x = img_to_array(img)

    # Reshape it to (1, 150, 150, 3)
    x = x.reshape((1,) + x.shape)

    # The .flow() command below generates batches of randomly transformed images.
    # It will loop indefinitely, so we need to 'break' at some point!

    datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        plt.imshow(array_to_img(batch[0]))
        i += 1
        if i % 4 == 0: # Only plot 4 randomly augmented images
            break

def show_data():
    category_names = os.listdir(os.path.join(os.getcwd(), "data"))
    fnames = []
    for category_name in category_names:
        fname = os.listdir(os.path.join('train', category_name))[0]
        fnames.append(os.path.join('train', category_name, fname))
    for fname in fnames:
        display(Image(filename=fname))

def predict_and_show(model):
    input_dir = os.path.join(os.getcwd(), "predict", "images")
    # output_dir = os.path.join(os.getcwd(), "output", "images")
    # Get Features
    conv_base = MobileNetV2(weights='imagenet',
                    include_top=False,
                    input_shape=(96, 96, 3))
    conv_base_output_shape = list(conv_base.layers[-1].output_shape[1:])
    shape = []
    shape.append(num_samples("predict"))
    shape.extend(conv_base_output_shape)
    features = np.zeros(shape=tuple(shape))
    batch_size = 1
    generator = ImageDataGenerator(rescale=1./255).flow_from_directory("predict",
                                                    target_size=(96, 96),
                                                    batch_size=batch_size,
                                                    class_mode=None,
                                                    shuffle=False #Important
                                                    )
    i = 0
    for inputs_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        i += 1
        if i * batch_size >= len(os.listdir(input_dir)):
            break
    features = reshape_features(features)
    input_filenames = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir)])
    preds = model.predict(features) # List of Lists (type = np.array)
    classes = sorted(os.listdir("data"))
    pred_labels = [classes[np.argmax(pred)] for pred in preds] # Name of predicted class
    preds = preds.tolist()
    if os.path.exists("output"):
        shutil.rmtree("output", ignore_errors=True)
    os.makedirs("output")
    for idx, filename in enumerate(input_filenames):
        result = dict()
        result['input_file'] = filename
        result['probabilities'] = []
        for idx2, prob in enumerate(preds[idx]):
            result['probabilities'].append({classes[idx2] : prob})
        result['label'] = pred_labels[idx]
        with open(os.path.join("output", filename.split(".")[0].split(os.sep)[-1]) + ".json", 'w') as outfile:
            json.dump(result, outfile, indent=4)
        display(Image(filename=filename))
        print(json.dumps(result, indent=4))
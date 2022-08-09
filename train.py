import tensorflow as tf
import tensorflow_datasets as tfds
from keras.callbacks import ModelCheckpoint


def normalize(image, label):
    normalized_image = tf.cast(image, tf.float32) / 255.
    return normalized_image, label


def train():
    # 1. Import Data
    (ds_train, ds_test), ds_info = tfds.load('horses_or_humans', split=['train','test'], as_supervised=True, with_info=True)
    ds_train = ds_train.map(normalize)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.shuffle(100)

    ds_test = ds_test.map(normalize)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.shuffle(100)


    # 2. Build Model

    # 3. Train Model

    # 4. Lưu Model
    return ds_train, ds_test, ds_info


def run():
    ds_train, ds_test, ds_info = train()
    class_no = ds_info.features['label'].num_classes
    shape = (300, 300, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=shape,
                                                   include_top=False,
                                                   weights='imagenet')

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    preds = tf.keras.layers.Dense(class_no, activation='softmax')(x)

    # Freeze layers
    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Model(inputs=base_model.inputs, outputs=preds)

    epohcs = 10
    model.compile(optimizer='sgd', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('model/best.hdf5', monitor='val_loss', save_best_only=True)
    callback = [checkpoint]

    model.fit(ds_train, epochs=epohcs, validation_data=ds_test, callbacks=callback)

    model.save('models/model.h5')
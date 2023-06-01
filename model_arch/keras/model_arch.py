from keras.applications import ResNet152V2, EfficientNetB0, EfficientNetV2B0
from keras.layers import (
    Input, add, Dense, Dropout, Flatten, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization,
    MultiHeadAttention, LayerNormalization, Resizing, Reshape, Add, concatenate, ConvLSTM2D, LSTM,
)
from keras.models import Sequential, Model
import keras.backend as K
from key_map import base_key


# weighted loss function in one-hot encoding
def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        weighted_loss = -(K.sum(weights * y_true * K.log(y_pred), axis=-1))
        return weighted_loss
    return loss


def get_CNN_model(width, height, all_key_weights_dino):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(height, width, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(len(base_key.all_key_and_type_comb), activation='softmax'))

    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.compile(loss=weighted_categorical_crossentropy(all_key_weights_dino), optimizer="adam", metrics=["accuracy"])

    return model


def get_ResNet152V2_model(width, height, all_key_weights_dino=None):
    input_shape = (height, width, 3)
    num_classes = len(base_key.all_key_and_type_comb)
    inputs = Input(shape=input_shape)

    base_model = ResNet152V2(include_top=False, weights=None, input_tensor=inputs, input_shape=input_shape, pooling=None, classes=num_classes)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def get_EfficientNetV2B0_model(width, height, all_key_weights_dino=None):
    input_shape = (height, width, 3)
    num_classes = len(base_key.all_key_and_type_comb)
    inputs = Input(shape=input_shape)

    base_model = EfficientNetV2B0(include_top=False, weights=None, input_tensor=inputs, input_shape=input_shape, pooling=None, classes=num_classes)
    x = base_model.output
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=weighted_categorical_crossentropy(all_key_weights_dino), optimizer="adam", metrics=["accuracy"])

    return model


def get_convLSTM2d_model(FPS, seconds, width, height, all_key_weights_dino=None):
    num_classes = len(base_key.all_key_and_type_comb)
    frame_size = FPS * seconds

    input1 = Input(shape=(seconds, height, width, 3))
    prev_frames = ConvLSTM2D(filters=4, kernel_size=(3, 3), padding='same', return_sequences=True)(input1)
    prev_frames = BatchNormalization()(prev_frames)
    prev_frames = Flatten()(prev_frames)
    prev_frames = Dense(16, activation='relu')(prev_frames)
    prev_frames = Dense(64, activation='relu')(prev_frames)

    input2 = Input(shape=(frame_size, num_classes))
    prev_keys = LSTM(units=128, return_sequences=True)(input2)
    prev_keys = LSTM(units=256, return_sequences=True)(prev_keys)
    prev_keys = LSTM(units=512)(prev_keys)
    prev_keys = Dense(units=64, activation='relu')(prev_keys)

    input3_shape = (height, width, 3)
    input3 = Input(shape=input3_shape)
    base_model = EfficientNetV2B0(include_top=False, weights=None, input_tensor=input3, input_shape=input3_shape, pooling=None, classes=num_classes)
    current_frame = base_model.output
    current_frame = Flatten()(current_frame)
    current_frame = Dense(units=64, activation='relu')(current_frame)

    concatenated = concatenate([prev_frames, prev_keys, current_frame])
    concatenated = Dropout(0.1)(concatenated)

    current_key = Dense(units=num_classes, activation='softmax')(concatenated)

    model = Model(inputs=[input1, input2, input3], outputs=current_key)

    model.compile(loss=weighted_categorical_crossentropy(all_key_weights_dino), optimizer='adam', metrics=['accuracy'])
    
    return model


def get_conv_lstm_model(frame_size, width, height, all_key_weights_dino=None):
    num_classes = len(base_key.all_key_and_type_comb)

    input1 = Input(shape=(frame_size, num_classes))
    prev_keys = LSTM(units=128, return_sequences=True)(input1)
    prev_keys = LSTM(units=256, return_sequences=True)(prev_keys)
    prev_keys = LSTM(units=512)(prev_keys)
    prev_keys = Dense(units=64, activation='relu')(prev_keys)

    input2_shape = (height, width, 3)
    input2 = Input(shape=input2_shape)
    base_model = EfficientNetV2B0(include_top=False, weights=None, input_tensor=input2, input_shape=input2_shape, pooling=None, classes=num_classes)
    current_frame = base_model.output
    current_frame = Flatten()(current_frame)
    current_frame = Dense(units=64, activation='relu')(current_frame)

    concatenated = concatenate([prev_keys, current_frame])
    concatenated = Dropout(0.1)(concatenated)

    current_key = Dense(units=num_classes, activation='softmax')(concatenated)

    model = Model(inputs=[input1, input2], outputs=current_key)

    model.compile(loss=weighted_categorical_crossentropy(all_key_weights_dino), optimizer='adam', metrics=['accuracy'])
    
    return model
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import glob
import os
import json
from key_map import base_key
import model_arch.keras.model_arch as model_arch
import warnings
warnings.filterwarnings("ignore")

width = 210
height = 60

key_weights_dino = {
    base_key.code2key["0"]: 0.1,  # none
    base_key.code2key["38"]: 3,  # up_arrow
    base_key.code2key["40"]: 1,  # down_arrow
}
all_key_weights_dino = [key_weights_dino.get(key_and_type.rsplit("_", 1)[0], base_key.default_key_weights) for key_and_type in base_key.all_key_and_type_comb]


def get_images_and_labels(data_path):
    with open(f'{data_path}/record.jsonl', 'r') as f:
        records: list[dict] = [json.loads(line) for line in f.readlines()]

    X = []
    Y = []

    for record in records:
        tick = record.get('tick')
        image_path = f'{data_path}/images/{tick}.png'
        image = Image.open(image_path)
        image = np.asarray(image.resize((width, height)), dtype=np.float16) / 255

        keyboard = record.get('keyboard')[0]  # TODO: support multiple keyboards (multi-hot encoding)
        label = keyboard['key'] + '_' + keyboard['type']

        X.append(image)
        Y.append(label)

    X = np.asarray(X)

    return X, Y


def sliding_window_generator(X, Y, FPS, seconds, batch_size=32, epochs=10):
    window_size = FPS * seconds

    for epoch in range(epochs):
        for i in range(len(X) - window_size - batch_size):
            prev_frames = []
            prev_keys = []
            current_frame = []
            current_key = []
            for j in range(batch_size):
                si = i + j
                prev_frames.append(X[si:si + seconds])
                prev_keys.append(Y[si:si + window_size])
                current_frame.append(X[si + window_size])
                current_key.append(Y[si + window_size])
                
            yield [np.asarray(prev_frames), np.asarray(prev_keys), np.asarray(current_frame)], np.asarray(current_key)


def onehot_labels(labels):
    labels = [[key] for key in labels]
    onehot_encoder = OneHotEncoder(sparse=False, categories=[base_key.all_key_and_type_comb])
    onehot_labels = onehot_encoder.fit_transform(labels)
    return onehot_labels


def plot_confusion_matrix():
    y_pred = model.predict(test_X)
    y_pred = np.argmax(y_pred, axis=1)
    test_y_new = np.argmax(test_y, axis=1)
    display_labels = [base_key.all_key_and_type_comb[num] for num in np.unique(np.concatenate([test_y_new, y_pred]))]

    cm = confusion_matrix(test_y_new, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    disp = disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')
    plt.title('Confusion Matrix')
    plt.show()


def plot_accuracy_and_loss():
    plt.plot(history.history['accuracy'], linewidth=2)
    plt.title('Train Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(history.history['loss'], color='orange', linewidth=2)
    plt.title('Train Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == "__main__":
    data_paths = glob.glob("./data/train/*") + glob.glob("./data/test/*")
    
    FPS = 32
    seconds = 4
    epochs = 100
    batch_size = 64

    model = model_arch.get_convLSTM2d_model(FPS, seconds, width, height, all_key_weights_dino)
    print(model.summary())

    X = np.array([])
    Y = np.array([])

    for data_path in data_paths:
        x, y = get_images_and_labels(data_path)
        y = onehot_labels(y)
        X = np.concatenate((X, x), axis=0) if X.size else x
        Y = np.concatenate((Y, y), axis=0) if Y.size else y
    
    # prev_frames, prev_keys, current_frame, current_key = sliding_window(X, Y, frame_size)
    data_gen = sliding_window_generator(X, Y, FPS, seconds, batch_size=batch_size, epochs=epochs)
    # train_X, test_X, train_y, test_y = train_test_split(X[:10], Y[:10], test_size=0.1, random_state=10)
    # test_size = 0.1
    # train_X = [prev_frames[:int(len(prev_frames)*(1-test_size))], prev_keys[:int(len(prev_keys)*(1-test_size))], current_frame[:int(len(current_frame)*(1-test_size))]]
    # test_X = [prev_frames[int(len(prev_frames)*(1-test_size)):], prev_keys[int(len(prev_keys)*(1-test_size)):], current_frame[int(len(current_frame)*(1-test_size)):]]
    # train_y = current_key[:int(len(current_key)*(1-test_size))]
    # test_y = current_key[int(len(current_key)*(1-test_size)):]

    history = model.fit(data_gen, epochs=epochs, steps_per_epoch=len(X)//batch_size)

    # train_accuracy = model.evaluate([np.array(train_X[0]), np.array(train_X[1]), np.array(train_X[2])], train_y)
    # print("Train accuracy: %", train_accuracy[1]*100)
        
    # test_accuracy = model.evaluate([np.array(test_X[0]), np.array(test_X[1]), np.array(test_X[2])], test_y)
    # print("Test accuracy: %", test_accuracy[1]*100)   

    # plot_accuracy_and_loss()
    # plot_confusion_matrix()
        
    open("model.json","w").write(model.to_json())
    model.save_weights("weights.h5")
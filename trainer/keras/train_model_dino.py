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
import model_arch
import warnings
warnings.filterwarnings("ignore")

width = 208
height = 64

key_weights_dino = {
    base_key.code2key["0"]: 0.5,  # none
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
        image = np.asarray(image.resize((width, height)) / 255, dtype=np.float16)

        keyboard = record.get('keyboard')[0]
        label = keyboard['key'] + '_' + keyboard['type']

        X.append(image)
        Y.append(label)

    X = np.asarray(X)

    return X, Y


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
    data_paths = glob.glob("./data/*")
    
    # model = model_arch.get_CNN_model(width, height, all_key_weights_dino)
    model = model_arch.get_ViT_b16_model(width, height, all_key_weights_dino)
    print(model.summary())

    epochs = 100

    X = np.array([])
    Y = np.array([])

    for data_path in data_paths:
        x, y = get_images_and_labels(data_path)
        y = onehot_labels(y)
        X = np.concatenate((X, x), axis=0) if X.size else x
        Y = np.concatenate((Y, y), axis=0) if Y.size else y

    train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.1, random_state=10)

    history = model.fit(train_X, train_y, epochs=epochs, batch_size=128)

    train_accuracy = model.evaluate(train_X, train_y)
    print("Train accuracy: %", train_accuracy[1]*100)
        
    test_accuracy = model.evaluate(test_X, test_y)
    print("Test accuracy: %", test_accuracy[1]*100)   

    plot_accuracy_and_loss()
    plot_confusion_matrix()
        
    open("model.json","w").write(model.to_json())
    model.save_weights("weights.h5")

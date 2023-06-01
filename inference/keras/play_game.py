import time
from PIL import Image
from mss import mss
import keyboard
import numpy as np
from keras.models import model_from_json
import random
from key_map import base_key, keyboard_lib_map
from collections import deque


def get_key_set(
        max_release_time: tuple = (0.2, 0.02), 
        cool_time: float = 0.2
    ):
    """
    Args:
        max_release_time (mean_seconds, std_seconds):
          Set the maximum amount of time a key can be held when pressed
          Use in random.gauss(mean, std)

        cool_time (float): 
          cool time (seconds)
          How long you can press that key again
    """
    return {
        "max_release_time": max_release_time, 
        "cool_time": cool_time,
        "last_press_time": 0,
    }


def press(set_keys: dict, keystroke_que: deque, keyboard_lib_key_code: int, base_key_name: str):
    print(base_key_name, "press")
    keyboard.press(keyboard_lib_key_code)
    set_keys[base_key_name]["last_press_time"] = time.time()
    keystroke_que.append(keyboard_lib_key_code)


# release works only when the key is pressed
def release(keystroke_que: deque, keyboard_lib_key_code: int):
    if keyboard_lib_key_code in keystroke_que:
        print(keyboard_lib_map.code2base_key_name[keyboard_lib_key_code], "release")
        keyboard.release(keyboard_lib_key_code)
        time.sleep(max(0, random.gauss(0.01, 0.002)))
        for key_code in keystroke_que:
            if key_code == keyboard_lib_key_code:
                keystroke_que.remove(key_code)
                break


def exit():
    global is_exit
    is_exit = True


if __name__ == '__main__':
    frame = {"top":280, "left":0, "width":700, "height":200}
    width = 210
    height = 60

    model_dir = "./models/dino/efficientNetV2B0"
    model = model_from_json(open(f"{model_dir}/model.json","r").read())
    model.load_weights(f"{model_dir}/weights.h5")

    keyboard.add_hotkey("esc", exit)
    ss_manager = mss()
    is_exit = False

    # Use only the set key
    set_keys = {
        base_key.code2key["38"]: get_key_set(),  # up_arrow
        base_key.code2key["40"]: get_key_set((0.4, 0.03)),  # down_arrow
    }

    keystroke_que = deque()

    while True:
        if is_exit == True:
            break

        screenshot = ss_manager.grab(frame)
        # cv.imshow("screenshot", np.array(screenshot))
        # cv.waitKey(1)
        image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        # grey_image = image.convert("L")
        a_img = np.asarray(image.resize((width, height)))
        img = np.asarray(a_img / 255, dtype=np.float16)
        
        X = np.asarray([img])
        prediction = model.predict(X)
        
        result = np.argmax(prediction)
            
        print("prediction", base_key.all_key_and_type_comb[result])
        base_key_name_and_key_type: str = base_key.all_key_and_type_comb[result]
        base_key_name, base_key_type = base_key_name_and_key_type.rsplit("_", 1)
        keyboard_lib_key_code = keyboard_lib_map.base_key_name2code.get(base_key_name)

        set_key = set_keys.get(base_key_name)
        if base_key_name != "None" and set_key:
            if base_key_type == "press" and set_key["last_press_time"] + set_key["cool_time"] < time.time():
                release(keystroke_que, keyboard_lib_key_code)
                press(set_keys, keystroke_que, keyboard_lib_key_code, base_key_name)
            if base_key_type == "release":
                release(keystroke_que, keyboard_lib_key_code)

        # Checking key release time
        if 0 < len(keystroke_que):
            for keyboard_lib_key_code in deque(keystroke_que):
                base_key_name = keyboard_lib_map.code2base_key_name[keyboard_lib_key_code]
                set_key = set_keys[base_key_name]
                release_time = random.gauss(*set_key["max_release_time"])
                if set_key["last_press_time"] + release_time < time.time():
                    release(keystroke_que, keyboard_lib_key_code)

        keyboard.press_and_release(28)  # Enter
        

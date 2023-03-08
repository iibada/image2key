import time
from PIL import Image
from mss import mss
import keyboard
from pathlib import Path
from threading import Thread 
from key_map import base_key, keyboard_lib_map
from datetime import datetime
import json


def get_status(tick: int, events):
    keyboards = []
    if len(events) == 0:
        keyboard = {
            "key": base_key.code2key["0"],
            "type": "none"
        }
        keyboards.append(keyboard)
    else:
        for event in events:
            keyboard = {
                "key": keyboard_lib_map.code2base_key_name[event.scan_code],
                "type": keyboard_lib_map.key_type2base_key_name[event.event_type]
            }
            keyboards.append(keyboard)

    status: dict = {
        "keyboard": keyboards,
        "tick": tick,
        "milli": int(time.time() * 1000)
    }
    return status


def save_data(path, tick, events):
    if len(events) == 0:
        scan_code = 0
    else:
        scan_codes = [event.scan_code for event in events]
        scan_code = scan_codes[0]
    key = keyboard_lib_map.code2base_key_name[scan_code]
    if key == "esc":
        return
    print("{}: {}".format(key, tick))
    img = ss_manager.grab(frame)
    image = Image.frombytes("RGB", img.size, img.rgb)
    image.save(f"{path}/images/{tick}.png")
    status = get_status(tick, events)
    with open(f"{path}/record.jsonl", "a") as f:
        f.write(json.dumps(status) + "\n")


def exit():
    global is_play
    is_play = False


if __name__ == '__main__':
    path = f'./data/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    images_path = f'{path}/images'
    Path(images_path).mkdir(parents=True, exist_ok=True)
    keyboard.add_hotkey("esc", exit)
    ss_manager = mss()
    # frame = {"top":280, "left":0, "width":700, "height":200}  # dino
    # frame = {"top":468, "left":668, "width":586, "height":178}  # pacman
    frame = {"top":300, "left":620, "width":680, "height":600}  # snake
    
    FPS = 20
    tick = 0
    is_play = True

    while is_play:
        tick += 1
        keyboard.start_recording()
        time.sleep(1 / FPS)
        events = keyboard.stop_recording()
        Thread(target=save_data, args=(path, tick, events)).start()



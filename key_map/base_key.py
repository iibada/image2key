code2key = {
    "0": "none",
    
    "8": "backspace",
    "9": "tab",
    "13": "enter",
    "16": "shift",
    "17": "ctrl",
    "18": "alt",
    "19": "pause",
    "20": "caps_lock",
    "21": "right_alt",
    "27": "esc",
    "32": "space",
    "33": "page_up",
    "34": "page_down",
    "35": "end",
    "36": "home",
    "37": "left_arrow",
    "38": "up_arrow",
    "39": "right_arrow",
    "40": "down_arrow",
    "44": "print_screen",
    "45": "insert",
    "46": "delete",

    "48": "0",
    "49": "1",
    "50": "2",
    "51": "3",
    "52": "4",
    "53": "5",
    "54": "6",
    "55": "7",
    "56": "8",
    "57": "9",

    "65": "a",
    "66": "b",
    "67": "c",
    "68": "d",
    "69": "e",
    "70": "f",
    "71": "g",
    "72": "h",
    "73": "i",
    "74": "j",
    "75": "k",
    "76": "l",
    "77": "m",
    "78": "n",
    "79": "o",
    "80": "p",
    "81": "q",
    "82": "r",
    "83": "s",
    "84": "t",
    "85": "u",
    "86": "v",
    "87": "w",
    "88": "x",
    "89": "y",
    "90": "z",

    "91": "left_window_key",
    "92": "right_window_key",
    "93": "select_key",

    "96": "numpad_0",
    "97": "numpad_1",
    "98": "numpad_2",
    "99": "numpad_3",
    "100": "numpad_4",
    "101": "numpad_5",
    "102": "numpad_6",
    "103": "numpad_7",
    "104": "numpad_8",
    "105": "numpad_9",
    "106": "multiply",
    "107": "add",
    "109": "subtract",
    "110": "decimal_point",
    "111": "divide",

    "112": "f1",
    "113": "f2",
    "114": "f3",
    "115": "f4",
    "116": "f5",
    "117": "f6",
    "118": "f7",
    "119": "f8",
    "120": "f9",
    "121": "f10",
    "122": "f11",
    "123": "f12",

    "144": "num_lock",
    "145": "scroll_lock",

    "186": "semi_colon",
    "187": "equal_sign",
    "188": "comma",
    "189": "dash",
    "190": "period",
    "191": "forward_slash",
    "192": "grave_accent",
    "219": "open_bracket",
    "220": "back_slash",
    "221": "close_braket",
    "222": "single_quote"
}
key2code = {v: k for k, v in code2key.items()}
all_keys = list(code2key.values())

key_type = {
    "1": "press",
    "2": "release",
}
all_key_type = list(key_type.values())

# Used for multi-hot encoding
key_and_type_comb = {
    "0_0": "none_none"
}
for key_code, key_name in list(code2key.items())[1:]:
    for type_code, type_name in key_type.items():
        key_and_type_comb[key_code + "_" + type_code] = key_name + "_" + type_name
all_key_and_type_comb = list(key_and_type_comb.values())

# Set key weights
default_key_weights = 0.0
key_weights_dino = {
    code2key["0"]: 0.01,  # none
    code2key["38"]: 3,  # up_arrow
    code2key["40"]: 1,  # down_arrow
}
all_key_weights_dino = [key_weights_dino.get(key_and_type.rsplit("_", 1)[0], default_key_weights) for key_and_type in all_key_and_type_comb]
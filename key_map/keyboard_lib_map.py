from key_map import base_key


code2base_key_name: dict = {
    0: base_key.code2key["0"],      # None
    1: base_key.code2key["27"],      # esc
    2: base_key.code2key["49"],      # 1
    3: base_key.code2key["50"],      # 2
    4: base_key.code2key["51"],      # 3
    5: base_key.code2key["52"],      # 4
    6: base_key.code2key["53"],      # 5
    7: base_key.code2key["54"],      # 6
    8: base_key.code2key["55"],      # 7
    9: base_key.code2key["56"],      # 8
    10: base_key.code2key["57"],     # 9
    11: base_key.code2key["48"],     # 0
    12: base_key.code2key["189"],    # dash
    13: base_key.code2key["187"],    # equal_sign
    14: base_key.code2key["8"],      # backspace
    15: base_key.code2key["9"],      # tab
    16: base_key.code2key["81"],     # q
    17: base_key.code2key["87"],     # w
    18: base_key.code2key["69"],     # e
    19: base_key.code2key["82"],     # r
    20: base_key.code2key["84"],     # t
    21: base_key.code2key["89"],     # y
    22: base_key.code2key["85"],     # u
    23: base_key.code2key["73"],     # i
    24: base_key.code2key["79"],     # o
    25: base_key.code2key["80"],     # p
    26: base_key.code2key["219"],    # open_bracket
    27: base_key.code2key["221"],    # close_braket
    28: base_key.code2key["13"],     # enter
    29: base_key.code2key["17"],     # ctrl
    30: base_key.code2key["65"],     # a
    31: base_key.code2key["83"],     # s
    32: base_key.code2key["68"],     # d
    33: base_key.code2key["70"],     # f
    34: base_key.code2key["71"],     # g
    35: base_key.code2key["72"],     # h
    36: base_key.code2key["74"],     # j
    37: base_key.code2key["75"],     # k
    38: base_key.code2key["76"],     # l
    39: base_key.code2key["186"],    # semi_colon
    40: base_key.code2key["222"],    # single_quote
    41: base_key.code2key["192"],    # grave_accent
    42: base_key.code2key["16"],     # shift
    43: base_key.code2key["220"],    # back_slash
    44: base_key.code2key["90"],     # z
    45: base_key.code2key["89"],     # x
    46: base_key.code2key["67"],     # c
    47: base_key.code2key["86"],     # v
    48: base_key.code2key["66"],     # b
    49: base_key.code2key["78"],     # n
    50: base_key.code2key["77"],     # m
    51: base_key.code2key["188"],    # comma
    52: base_key.code2key["190"],    # period
    53: base_key.code2key["191"],    # forward_slash
    # 53: base_key.code2key["111"],  # divide
    55: base_key.code2key["106"],    # multiply
    # 55: base_key.code2key["44"],   # print_screen
    56: base_key.code2key["18"],     # alt
    # 56: base_key.code2key["21"],   # right_alt
    57: base_key.code2key["32"],     # space
    58: base_key.code2key["20"],     # caps_lock
    59: base_key.code2key["112"],    # f1
    60: base_key.code2key["113"],    # f2
    61: base_key.code2key["114"],    # f3
    62: base_key.code2key["115"],    # f4
    63: base_key.code2key["116"],    # f5
    64: base_key.code2key["117"],    # f6
    65: base_key.code2key["118"],    # f7
    66: base_key.code2key["119"],    # f8
    67: base_key.code2key["120"],    # f9
    68: base_key.code2key["121"],    # f10
    69: base_key.code2key["19"],     # pause
    # 69: base_key.code2key["144"],  # num_lock
    70: base_key.code2key["145"],    # scroll_lock
    71: base_key.code2key["36"],     # home
    # 71: base_key.code2key["103"],  # numpad_7
    72: base_key.code2key["38"],     # up_arrow
    # 72: base_key.code2key["104"],  # numpad_8
    73: base_key.code2key["33"],     # page_up
    # 73: base_key.code2key["105"],  # numpad_9
    74: base_key.code2key["109"],    # subtract
    75: base_key.code2key["37"],     # left_arrow
    # 75: base_key.code2key["100"],  # numpad_4
    76: base_key.code2key["101"],    # numpad_5
    # 77: base_key.code2key["102"],  # numpad_6
    77: base_key.code2key["39"],     # right_arrow
    78: base_key.code2key["107"],    # add
    79: base_key.code2key["35"],     # end
    # 79: base_key.code2key["97"],   # numpad_1
    80: base_key.code2key["40"],     # down_arrow
    # 80: base_key.code2key["98"],   # numpad_2
    81: base_key.code2key["34"],     # page_down
    # 81: base_key.code2key["99"],   # numpad_3
    82: base_key.code2key["45"],     # insert
    # 82: base_key.code2key["96"],   # numpad_0
    83: base_key.code2key["46"],     # delete
    # 83: base_key.code2key["110"],  # decimal_point
    87: base_key.code2key["122"],    # f11
    88: base_key.code2key["123"],    # f12
    91: base_key.code2key["91"],     # left_window_key
    92: base_key.code2key["92"],     # right_window_key
    93: base_key.code2key["93"],     # select_key
}

base_key_name2code = {v: k for k, v in code2base_key_name.items()}

key_type2base_key_name = {
    "down": base_key.key_type["1"],
    "up": base_key.key_type["2"]
}
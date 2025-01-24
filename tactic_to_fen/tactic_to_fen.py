from PIL import Image, ImageChops, ImageEnhance
import os
import hashlib
from util import generate_image_hash, image_to_squares, fix_fen
from square_recognizer_ai import image_prediction_to_fen_representation, get_model, clear_model_from_ram
#dir = "/home/atoste/Downloads/chess_board_recognition/tactic_to_fen"
## Path to the input image
#image = "page32index0.jpg"
#input_image_path = dir + "/woodpecker/" + image
#i = 0
#string_initial = ""
#model = get_model()
#for input_image_path in [str(dir+"/woodpecker/page32index1.jpg"), str(dir+"/woodpecker/page32index2.jpg"),str(dir+"/woodpecker/page32index3.jpg"),str(dir+"/woodpecker/page32index4.jpg")]:
#    i = 0
#    for image in image_to_squares(input_image_path):
#            i += 1
#            output_dir = dir + "/tmp"
#            string_initial += image_prediction_to_fen_representation(image, model)
#            if i == 8:
#                i = 0
#                string_initial += "/"
#    print(fix_fen(string_initial))
#    exit()

def get_fen_model():
    return get_model()

def clear_fen_model_from_ram(model):
    clear_model_from_ram(model)


def tactic_to_fen(image, model):
    i = 0
    string_initial = ""
    for image in image_to_squares(image,path=False):
        i += 1
        string_initial += image_prediction_to_fen_representation(image, model)
        if i == 8:
            i = 0
            string_initial += "/"
    return fix_fen(string_initial)





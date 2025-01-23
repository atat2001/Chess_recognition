from PIL import Image, ImageChops, ImageEnhance
import os
import hashlib

# Function to generate a hash for an image
def generate_image_hash(image):
    image_bytes = image.tobytes()
    return hashlib.md5(image_bytes).hexdigest()

def image_to_squares(image, path=True):
    output_images = []

    if path:
        # Open the image
        image = Image.open(image)

    width, height = image.size

    # Calculate the size of each sub-image
    sub_image_width = width // 8
    sub_image_height = height // 8

    # Split the image into 64 (8x8) smaller images
    for row in range(8):
        for col in range(8):
            left = col * sub_image_width
            upper = row * sub_image_height
            right = (col + 1) * sub_image_width
            lower = (row + 1) * sub_image_height
            sub_image = image.crop((left, upper, right, lower))
            output_images.append(sub_image)
            #sub_image_hash = generate_image_hash(sub_image)
            #sub_image_path = os.path.join(output_dir, f"{sub_image_hash}.jpg")
            #sub_image.save(sub_image_path)
    return output_images

def fix_fen(string_input):
    n = 0
    output = ""
    for x in string_input:
        if x == "_":
            n +=1
        else:
            if n != 0:
                output += str(n)
                n = 0
            output += x
    if n != 0:
        output += str(n)
    return output[:-1]

def get_long_fen(fen):
    fen = fen.split("/")
    long_fen = ""
    for row in fen:
        for x in row:
            if x.isdigit():
                long_fen += "_"*int(x)
            else:
                long_fen += x
    return long_fen

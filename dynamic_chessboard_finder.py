import mss
import cv2
import numpy as np
import time
from PIL import Image,ImageChops, ImageEnhance
import os
import hashlib
from populate_app import get_last_index,add_puzzle,get_next_index  ## this is optional
import fitz  # PyMuPDF
from pynput import keyboard
import sys



"""
Main app: detect_chess_diagram and get_chessboards do edge detection to find any square shapes, identify if they are chessboards or not(using detect_board/verify_board).
            detect_chess_diagram will return the frame(which can highlight square shapes and chessboards) and will save the chessboards/none chessboards found.
            get_chessboards will output a List: [[board_image, frame],...]
        
        detect_chess_on_screen_by_key will wait for you to input START_CHAR, once you do it will take a screenshot, using get_chessboards will identify the boards
        on screen and will run the trainer/training_gui to get the fen. Its important that you correct the fen on the trainer or discard of the information after using
        the trainer using trainin_data_manager's delete_predictions function to avoid finetuning with bad info!
        The trainer stores every fen you click on Done or Train so make sure to correct them.

        After the trainer it will open the gui, which is pretty self explanatory. If the position is inverted(pawns moving to the wrong side) you can revert it
        on edit board -> invert position.

        This was trained using the woodpecker method book, if you want to use this on another source you can train the CNNs.
        I recomend not mixing sources, so training it from scratch for another book/source.
"""
START_CHAR = "ç"     ## chose this key as nothing uses this key but might be useful to change as not every keyboard has it

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the directory containing util.py to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir,'detect_board')))
from verify_board import verify, get_model, clear_model_from_ram
# Add the directory containing util.py to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir,'tactic_to_fen')))

from tactic_to_fen.tactic_to_fen import tactic_to_fen, get_fen_model, clear_fen_model_from_ram
sys.path.append(os.path.abspath(os.path.join(current_dir,'gui')))
import gui.gui

# Add the directory containing util.py to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir,'trainer')))
import training_gui

sys.path.append(os.path.abspath(os.path.join(current_dir,'chessboard_trainer')))
import chessboard_trainer


# Directory paths
recognized_chessboards_dir = 'data/train/chessboard_try'
not_chessboards_dir = 'data/train/not_chessboard_try'

## not provided due to copywrights, you need to get your own pdf
woodpecker = "/home/atoste/Desktop/chess/The_Woodpecker_Method_by_Axel_Smith_and.pdf"

# gets the last index of the puzzles added to Encroissant(only used for that, ignore if you dont want to save it there)
current_index = get_next_index(get_last_index())  

relative_index = 0

# Create directories if they don't exist
os.makedirs(recognized_chessboards_dir, exist_ok=True)
os.makedirs(not_chessboards_dir, exist_ok=True)


# Function to trim white and gray padding
def trim_white_and_gray_padding(img, threshold=125):
    # Convert image to grayscale
    gray_img = img.convert("L")
    # Enhance contrast to better differentiate between padding and content
    enhancer = ImageEnhance.Contrast(gray_img)
    gray_img = enhancer.enhance(2.0)

    # Create a binary image where light pixels are white and others are black
    binary_img = gray_img.point(lambda p: p > threshold and 255)
    # Invert the binary image
    inverted_img = ImageChops.invert(binary_img)
    #inverted_img.save("/home/atoste/Downloads/chess_board_recognition/tactic_to_fen/tmp/binary_white.jpg")
    #print(f"Saved grayscale image:")
    # Get the bounding box of the non-black areas
    bbox = inverted_img.getbbox()
    if bbox:
        return img.crop(bbox)
    return img


# Function to trim dark padding from the edges
def trim_dark_padding(img, threshold=125, adjust=2):
    # Convert image to grayscale
    gray_img = img.convert("L")
    # Create a binary image where dark pixels are white and others are black
    binary_img = gray_img.point(lambda p: p > threshold and 255)

    
    # Invert the binary image
    inverted_img = ImageChops.invert(binary_img)
    #inverted_img.save("/home/atoste/Downloads/chess_board_recognition/tactic_to_fen/tmp/binary_black.jpg")
    #print(f"Saved binary image: {inverted_img}")

    # Get the bounding box of the non-black areas
    bbox = inverted_img.getbbox()
    if bbox:
        return img.crop(bbox)
    return img

## saves tactic from woodpecker to a dir
def save_to_dir(roi_pil, page, index, path = "woodpecker"):
    recognized_chessboard_path = os.path.join(path, f'page{page}index{index}.jpg')
    # Ensure the directory exists
    os.makedirs(os.path.dirname(recognized_chessboard_path), exist_ok=True)
    
    roi_pil.save(recognized_chessboard_path)

def save_image_in_try_chessboards_or_not(model,roi_pil,classification=None):
    # Convert the image to a hash to check for uniqueness
    img_hash = hashlib.md5(roi_pil.tobytes()).hexdigest()
    not_chessboard_path = os.path.join(not_chessboards_dir, f'{img_hash}.jpg')
    recognized_chessboard_path = os.path.join(recognized_chessboards_dir, f'{img_hash}.jpg')

    if not os.path.exists(not_chessboard_path) and not os.path.exists(recognized_chessboard_path):
        if classification is not None:
            if classification:
                roi_pil.save(recognized_chessboard_path)
            else:
                roi_pil.save(not_chessboard_path)

        elif(verify(model,roi_pil)):
            # Save the image to the "chessboards" directory
            roi_pil.save(recognized_chessboard_path)
            #print(f"Saved recognized chessboard to {recognized_chessboard_path}")
        else:
            # Save the image to the "not chessboards" directory
            roi_pil.save(not_chessboard_path)
            #print(f"Saved recognized chessboard to {not_chessboard_path}")


# Function to capture the screen (entire screen or specific region)
def capture_screen(region=None):
    with mss.mss() as sct:
        # Capture the whole screen or a specific region
        if region:
            screenshot = sct.grab(region)
        else:
            screenshot = sct.grab(sct.monitors[1])  # Capture the first monitor

        # Convert the screenshot to a numpy array for OpenCV
        img = np.array(screenshot)

        # Convert RGB to BGR (because OpenCV uses BGR format)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img_bgr

def get_countour_coordinates_and_frame(frame, show_contours=False):
    # Load image, grayscale, median blur, sharpen image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # Threshold and morph close
    thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and filter using threshold area
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    outputs = []
    min_area = 100
    if show_contours:
        frame_aux = frame.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            x,y,w,h = cv2.boundingRect(c)
            if show_contours:
                cv2.rectangle(frame_aux, (x, y), (x + w, y + h), (0,255,0), 2)
            outputs += [[x,y,w,h],]
    if show_contours:
        return [outputs,] + [frame_aux,]
    return [outputs,] + [frame,]

# Function to detect chess diagrams using adaptive thresholding and edge detection
def detect_chess_diagram(frame, current_index,page = 0, save_as_image = True, show_contours=False,model_input=None):
    if model_input is None:
        model = get_model()
    else:
        model = model_input
    contour_coordinates,frame_aux = get_countour_coordinates_and_frame(frame, show_contours)
    relative_index = 0
    for x,y,w,h in contour_coordinates:
        # Extract the region of interest (ROI)
        roi = frame[y:y + h, x:x + w]  # This slices the image to get the desired rectangle
        # Convert ROI to PIL image
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))      
        roi_pil = trim_white_and_gray_padding(roi_pil)  
        roi_pil = trim_dark_padding(roi_pil)     
        roi_pil = trim_white_and_gray_padding(roi_pil)
        # Save the recognized chessboard image to the "not chessboards" directory
        save_image_in_try_chessboards_or_not(model,roi_pil)

        # Verify if the ROI is a chessboard
        if verify(model,roi_pil):
            save_to_dir(roi_pil,page, current_index + relative_index)
            relative_index += 1
            if show_contours:
                #print("Chessboard detected!")
                cv2.rectangle(frame_aux, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Yellow rectangle for detected chessboard  
    if not model_input is None:
        clear_model_from_ram(model)   ## if model was not passed as input, clear it from ram
    if show_contours:
        return frame_aux

    return frame

# Function that returns a list of chessboards from a frame
def get_chessboards(frame, current_index,page = 0, save_as_image = True, show_contours=False, model_input=None):
    if model_input is None:
        model = get_model()
    else:
        model = model_input
    ## trying new contours
    contour_coordinates,frame_aux = get_countour_coordinates_and_frame(frame, show_contours)
    relative_index = 0
    outputs = []
    for x,y,w,h in contour_coordinates:
        # Extract the region of interest (ROI)
        roi = frame[y:y + h, x:x + w]  # This slices the image to get the desired rectangle
        # Convert ROI to PIL image
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))      
        roi_pil = trim_white_and_gray_padding(roi_pil)  
        roi_pil = trim_dark_padding(roi_pil)     
        roi_pil = trim_white_and_gray_padding(roi_pil)
        # Save the recognized chessboard image to the "not chessboards" directory
        save_image_in_try_chessboards_or_not(model,roi_pil)

        # Verify if the ROI is a chessboard
        if verify(model,roi_pil):
            outputs += [roi_pil,]
            #save_to_dir(roi_pil,page, current_index + relative_index)
            relative_index += 1
            if show_contours:
                #print("Chessboard detected!")
                cv2.rectangle(frame_aux, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Yellow rectangle for detected chessboard  
    if not model_input is None:
        clear_model_from_ram(model)   ## if model was not passed as input, clear it from ram
    return outputs

def get_contour_images_and_classification(frame, model_input=None):
    show_contours = False
    if model_input is None:
        model = get_model()
    else:
        model = model_input
    contour_coordinates,frame_aux = get_countour_coordinates_and_frame(frame, show_contours)
    relative_index = 0
    roi_pils = []
    classifications = []
    for x,y,w,h in contour_coordinates:
        # Extract the region of interest (ROI)
        roi = frame[y:y + h, x:x + w]  # This slices the image to get the desired rectangle
        # Convert ROI to PIL image
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))      
        roi_pil = trim_white_and_gray_padding(roi_pil)  
        roi_pil = trim_dark_padding(roi_pil)     
        roi_pil = trim_white_and_gray_padding(roi_pil)

        roi_pils += [roi_pil,]
        # Verify if the ROI is a chessboard
        if verify(model,roi_pil):
            classifications += [True,]
        else:
            classifications += [False,]
    if not model_input is None:
        clear_model_from_ram(model)   ## if model was not passed as input, clear it from ram
    return [roi_pils,] + [classifications,]

def detect_chess_on_screen_by_key(region=None, delay=0.5, key='p',train_board = True):
    global current_index
    model = None
    fen_model = None
    def on_key_event(key,model, fen_model):
        try:
            if key.char == START_CHAR:
                chessboards = []
                # Capture the screen
                frame = capture_screen(region)
                if train_board:
                    true_classifications = []
                    chessboards = []
                    countours, classification = get_contour_images_and_classification(frame, model_input=model)
                    clear_model_from_ram(model)
                    i = 0
                    while len(classification)-i > 6:
                        true_classifications += chessboard_trainer.training_gui(countours[i:i+6],classification[i:i+6])
                        i += 6
                    true_classifications += chessboard_trainer.training_gui(countours[i:-1],classification[i:-1])
                    for index in range(len(true_classifications)):
                        if true_classifications[index]:
                            save_image_in_try_chessboards_or_not(None,countours[index],True)
                            chessboards += [countours[index],]
                        else:
                            save_image_in_try_chessboards_or_not(None,countours[index],False)
                else:
                    # Detect chessboard in the frame and show contours
                    chessboards = get_chessboards(frame, current_index,model_input=model)
                    clear_model_from_ram(model)
                model = None
                fens_and_frames = []
                for chessboard in chessboards:
                    fens_and_frames += [[tactic_to_fen(chessboard,fen_model), chessboard]]

                ## app was slow bc of memory, this will avoid the memory usage but might make it a bit slower
                clear_fen_model_from_ram(fen_model)
                fen_model = None
                print(len(fens_and_frames))
                for fen in fens_and_frames:
                    print(fen[0])
                    training_gui.main(fen[0], fen[1])
                    print("gui closed")
        except Exception as e:
            print(e)
    while True:
        if model == None:
            model = get_model()
        if fen_model == None:
            fen_model = get_fen_model()
        try:
            # Start listening to keyboard events
            with keyboard.Listener(on_press=lambda event: on_key_event(event, model=model,fen_model=fen_model)) as listener:
                listener.join()
        except:
            pass
        ## or we load the model or we wait, not both
        if model == None or fen_model == None:
            if model == None:
                model = get_model()
            if fen_model == None:
                fen_model = get_fen_model()
        else:
            time.sleep(delay)

def detect_chess_on_screen(region=None, delay=0.5):
    global current_index
    model = get_model()  ## loads chess recognizer model
    while True:
        # Capture the screen
        frame = capture_screen(region)
        # Detect chessboard in the frame and show contours
        processed_frame = detect_chess_diagram(frame, current_index,model)
        # Display the frame with contours and any detected diagrams
        cv2.imshow('Chess Diagram Detection with Improved Method', processed_frame)
        # Wait for a short duration between frames
        time.sleep(delay)
        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    model = clear_model_from_ram(model)
    cv2.destroyAllWindows()

# Function to convert PDF page to image
def pdf_page_to_image(pdf_path, page_number):
    document = fitz.open(pdf_path)
    page = document.load_page(page_number)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


# Function to process PDF pages
def detect_chess_in_pdf(pdf_path, show_contours=False):
    current_index
    document = fitz.open(pdf_path)
    num_pages = document.page_count
    for page_number in range(num_pages):
        # Convert PDF page to image
        img = pdf_page_to_image(pdf_path, page_number)

        # Convert PIL image to OpenCV format
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Detect chessboard in the frame and show contours
        processed_frame = detect_chess_diagram(frame, 0, page=page_number, show_contours=show_contours)
        if show_contours:
            # Display the frame with contours and any detected diagrams
            cv2.imshow('Chess Diagram Detection from PDF', processed_frame)

            # Break the loop when 'q' key is pressed
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    if show_contours:
        cv2.destroyAllWindows()

# Run the detection on the entire book
#detect_chess_in_pdf(woodpecker)
detect_chess_on_screen_by_key()

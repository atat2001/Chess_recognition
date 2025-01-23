

"""
Scraper that gets fens from the website https://helpman.komtera.lt/chessocr/.
It will get the fen, its useful when you want to get fens automaticly to speed up the start of the learning process (you have to 
do it if you want the app to recognize different types of pieces)
Currently I would advise to do 1 file for each piece font.
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from PIL import Image
import time
import os


def convert_img_to_fen(img):
    dirname = os.path.dirname(__file__)
    image_path = os.path.join(dirname, "tmp")
    image = img.save(f"{image_path}/image.png")
    
    options = webdriver.ChromeOptions()

    options.page_load_strategy = 'normal'
    driver = webdriver.Chrome(options=options)

    driver.get("https://helpman.komtera.lt/chessocr/")


    drop_zone = driver.find_element(By.CLASS_NAME,"drop-zone__input")

    path = f"{image_path}/image.png"
    full_path = os.path.abspath(path)
    drop_zone.send_keys(full_path)

    wait = WebDriverWait(driver, timeout=8)
    wait.until(lambda d : driver.find_element(By.ID, "helpman-link").get_attribute('href')[32:] != "8/8/8/8/8/8/8/8")

    fen = driver.find_element(By.ID, "helpman-link").get_attribute('href')[32:]

    print(fen)


    driver.quit()   
    # remove image.png from tmp
    os.remove(f"{image_path}/image.png")
    return fen


#convert_img_to_fen(Image.open("data/train/chessboard/0a3628e8e180ebbe99ca86d6d46b5d04.jpg"))
#convert_img_to_fen(Image.open("data/train/chessboard/0c33143678ce37fec7f7563ed9c1c901.jpg"))

import pygame
from PIL import Image
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '../detect_board')))
import verify_board
import train_ml

WIDTH, HEIGHT = 1200, 600
IMAGE_SIZE = 300

GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

def draw_images(screen, images):
    y = 0
    while(y < 2):
        x = 0
        while(x < 3):
            if len(images) <= x+y*3:
                break
            image = images[x + y * 3]
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
            screen.blit(image, (x * IMAGE_SIZE, y * IMAGE_SIZE))
            x = x + 1
        y = y + 1

def draw_boxes(screen, results):
    y = 0
    while(y < 2):
        x = 0
        while(x < 3):
            if len(results) <= x+y*3:
                break
            color = GREEN if results[x+y*3] else RED
            pygame.draw.rect(screen, color, pygame.Rect(x*IMAGE_SIZE, y*IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), 5)
            x = x + 1
        y = y + 1


def draw_buttons(screen):
    font = pygame.font.Font(None, 36)
    text = font.render('Submit', True, WHITE)
    text_rect = text.get_rect(center=(WIDTH - 150, 125))
    pygame.draw.rect(screen, GREEN, pygame.Rect(WIDTH - 200, 100, 100, 50))
    screen.blit(text, text_rect)

    text = font.render('Skip', True, WHITE)
    text_rect = text.get_rect(center=(WIDTH - 150, 225))
    pygame.draw.rect(screen, RED, pygame.Rect(WIDTH - 200, 200, 100, 50))
    screen.blit(text, text_rect)

    text = font.render('Train', True, WHITE)
    text_rect = text.get_rect(center=(WIDTH - 150, 325))
    pygame.draw.rect(screen, RED, pygame.Rect(WIDTH - 200, 300, 100, 50))
    screen.blit(text, text_rect)

def get_image_by_pos(pos):
    x, y = pos
    row = y // IMAGE_SIZE
    col = x // IMAGE_SIZE
    return col, row

def handle_image_click(pos, images, results):
    x,y = get_image_by_pos(pos)
    print("changing", x, y)
    if len(images) < x + y * 3:
        return
    results[x + y * 3] = not results[x + y * 3]
    return

def handle_button_click(pos, images, results):
    x, y = pos
    if x > WIDTH - 200 and x < WIDTH - 100:
        if y > 100 and y < 150:
            return 1
        elif y > 200 and y < 250:
            return 0
        elif y > 300 and y < 350:
            return 2
    return -1



def training_gui(images, results_input=None):
    if images is None:
        return None
    results = []
    if images is not None and results_input is None:
        for image in images:
            if(verify_board.verify(image)):
                results = results + [True]
            else:
                results = results + [False]
    elif results_input is not None:
        results = results_input
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chessboard Trainer')
    clock = pygame.time.Clock()
    running = True
    train = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if pos[0] < 3*IMAGE_SIZE:
                    handle_image_click(pos, images, results)
                else:
                    code = handle_button_click(pos, images, results)
                    if code == 1:
                        running = False
                    elif code == 0:
                        results = None
                        running = False
                    elif code == 2:
                        running = False
                        print("Training")
                        train = True
        screen.fill(WHITE)
        if images is not None:
            draw_images(screen, images)
        draw_buttons(screen)
        draw_boxes(screen, results)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    if train:
        train_ml.fine_tune_model()
    return results

training_gui(None)
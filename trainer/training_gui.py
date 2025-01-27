
import chess
import chess.svg
import pygame
import cairosvg
from io import BytesIO
from PIL import Image
import sys
import os

import gui.gui
from training_data_manager import ensure_every_class_has_images, destroy_tmp_files
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '../gui')))
import gui
sys.path.append(os.path.abspath(os.path.join(current_dir, '../tactic_to_fen')))
import util
import square_recognizer_ai as square_recognizer


square_data_folder = os.path.join(current_dir,"square_data")

WIDTH, HEIGHT = 1500, 600
BOARD_SIZE = 8
SQUARE_SIZE = HEIGHT // BOARD_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 150)
HIGHLIGHT = (128, 128, 128)

def get_piece_image(piece):
    if piece is None:
        return None
    svg_data = chess.svg.piece(piece)
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(BytesIO(png_data))
    image = image.resize((int(SQUARE_SIZE), int(SQUARE_SIZE)), Image.LANCZOS)
    return pygame.image.fromstring(image.tobytes(), image.size, image.mode)

def get_square_under_mouse(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

def draw_board(screen, game):
    colors = [WHITE, BLACK]
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            row_aux = row
            col_aux = col
            color = colors[(row_aux + col_aux) % 2]

            pygame.draw.rect(screen, color, pygame.Rect(col_aux * SQUARE_SIZE, row_aux * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            piece = game.piece_at(chess.square(col, 7 - row))
            piece_image = get_piece_image(piece)
            if piece_image:
                piece_rect = piece_image.get_rect(center=(col_aux * SQUARE_SIZE + SQUARE_SIZE // 2, row_aux * SQUARE_SIZE + SQUARE_SIZE // 2))
                screen.blit(piece_image, piece_rect.topleft)

def draw_image(screen, image):
    image = image.resize((HEIGHT, HEIGHT), Image.LANCZOS)
    image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    screen.blit(image, (HEIGHT, 0))

def draw_editing_tab(screen, editing_position):
    editing_rect = pygame.Rect(2*HEIGHT, 0, WIDTH-2*HEIGHT, HEIGHT)
    pygame.draw.rect(screen, BLACK, editing_rect)
    font = pygame.font.Font(None, 36)

    set_position = pygame.Rect(2*HEIGHT + 10, 10, 90, 50)
    pygame.draw.rect(screen, WHITE, set_position)
    analyze_button_text = font.render("Done", True, BLACK)
    screen.blit(analyze_button_text, (set_position.x + 10, set_position.y + 10))
    
    train_button = pygame.Rect(2*HEIGHT + 10 + 120, 10, 90, 50)
    pygame.draw.rect(screen, WHITE, train_button)
    analyze_button_text = font.render("Train", True, BLACK)
    screen.blit(analyze_button_text, (train_button.x + 10, train_button.y + 10))
    ## should launch gui and save data

    pieces = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']
    for i, piece in enumerate(pieces):
        piece_rect = pygame.Rect(2*HEIGHT + 10, 10 + i * 60, 50, 50)
        #pygame.draw.rect(screen, WHITE, piece_rect)
        piece_image = get_piece_image(chess.Piece.from_symbol(piece))
        if piece_image:
            if i > 5:
                piece_rect = piece_image.get_rect(center=(2*HEIGHT + 70 + 25, (10 + (i-6) * 60) + 85))
            else:
                piece_rect = piece_image.get_rect(center=(2*HEIGHT + 10 + 25, (10 + i * 60) + 85))
            screen.blit(piece_image, piece_rect.topleft)
    
    invert_button = pygame.Rect(2*HEIGHT + 10, HEIGHT-70, 180, 50)
    pygame.draw.rect(screen, WHITE, invert_button)
    undo_button_text = font.render("Delete piece", True, BLACK)
    screen.blit(undo_button_text, (invert_button.x + 10, invert_button.y + 10))

def save_data(game,image):
    squares = util.image_to_squares(image, path=False)
    long_fen = util.get_long_fen(game.fen())
    classifications = ['b_b', 'b_k', 'b_n', 'b_p', 'b_q', 'b_r', 'empty', 'w_b', 'w_k', 'w_n', 'w_p', 'w_q', 'w_r']
    folder = ""
    for i, x in enumerate(squares):
        if long_fen[i] == "_":
            folder = "empty"
        if long_fen[i] == "R":
            folder = "w_r"
        if long_fen[i] == "N":
            folder = "w_n"
        if long_fen[i] == "B":
            folder = "w_b"
        if long_fen[i] == "Q":
            folder = "w_q"
        if long_fen[i] == "K":
            folder = "w_k"
        if long_fen[i] == "P":
            folder = "w_p"
        if long_fen[i] == "r":
            folder = "b_r"
        if long_fen[i] == "n":
            folder = "b_n"
        if long_fen[i] == "b":
            folder = "b_b"
        if long_fen[i] == "q":
            folder = "b_q"
        if long_fen[i] == "k":
            folder = "b_k"
        if long_fen[i] == "p":
            folder = "b_p"
        x.save(f"{square_data_folder}/{folder}/{util.generate_image_hash(x)}.jpg")

def train(game):    
    pygame.quit()
    #sys.exit()
    ensure_every_class_has_images()

    square_recognizer.fine_tune(square_data_folder)
    destroy_tmp_files()
    gui.gui.main(game.fen())
    pass

def handle_editing_click(pos, editing_position,game,image):
    x, y = pos
    if y > 10 and y < 10 + 50:
        if x < 2*HEIGHT + 110:
            print("Done, saving game")
            save_data(game,image)
            pygame.quit()
            print("opening gui")
            gui.gui.main(game.fen())
            print("gui closed")
        else:
            print("Done, training")
            save_data(game,image)
            train(game)
            #gui.main()

    if y > HEIGHT-70 and y < HEIGHT-20:
        editing_position[1] = None
    if x < 2*HEIGHT + 70 + 25 + SQUARE_SIZE//2 and y > 95 - SQUARE_SIZE//2 and y < (10 + 6 * 60) + 85 + SQUARE_SIZE//2:
        pieces = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']
        index = 0
        for i, piece in enumerate(pieces):
            if y > 10 + i * 60 and y < 10 + i * 60 + 85 + SQUARE_SIZE//2:
                editing_position[1] = chess.Piece.from_symbol(piece)
                index = i
                break
        if x > 2*HEIGHT + 40 + 25:
            editing_position[1] = chess.Piece.from_symbol(pieces[index+6])
    return

def main(fen=None, image=None):
    pygame.init()
    game = chess.Board()

    if fen != None:
        game.set_fen(fen)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Train GUI')
    editing_position = [None, None]
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x,y = pygame.mouse.get_pos()
                if x < HEIGHT:
                    selected_square = get_square_under_mouse((x,y))
                    selected_square = chess.SQUARE_NAMES[selected_square[1] + 8 * (7 - selected_square[0])]
                    game.set_piece_at(chess.parse_square(selected_square), editing_position[1])
                elif x > 2*HEIGHT:
                    handle_editing_click((x,y), editing_position,game,image)
                pass
        if not pygame.get_init():
            running = False
            continue
        screen.fill(WHITE)
        draw_board(screen,game)
        if not image is None:
            draw_image(screen, image)
        draw_editing_tab(screen, editing_position)
        pygame.display.flip()
    if pygame.get_init():
        pygame.quit()

        
if __name__ == '__main__':
    main()
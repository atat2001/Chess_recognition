import pygame
import sys
import threading
import queue
import chess
import chess.svg
import cairosvg
from io import BytesIO
from PIL import Image
from analyse import get_engine, analyse_position
# Constants
WIDTH, HEIGHT = 900, 600
BOARD_SIZE = 8
SQUARE_SIZE = HEIGHT // BOARD_SIZE
ANALYSIS_HEIGHT = HEIGHT
ANALYSIS_WIDTH = 300
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 150)
HIGHLIGHT = (128, 128, 128)

def get_square_under_mouse(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

def in_analysis_tab(pos):
    x, y = pos
    return x >= HEIGHT

def get_piece_image(piece):
    if piece is None:
        return None
    svg_data = chess.svg.piece(piece)
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(BytesIO(png_data))
    image = image.resize((int(SQUARE_SIZE), int(SQUARE_SIZE)), Image.LANCZOS)
    return pygame.image.fromstring(image.tobytes(), image.size, image.mode)

def get_small_piece_image(piece):
    if piece is None:
        return None
    svg_data = chess.svg.piece(piece)
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(BytesIO(png_data))
    image = image.resize((int(SQUARE_SIZE//2), int(SQUARE_SIZE//2)), Image.LANCZOS)
    return pygame.image.fromstring(image.tobytes(), image.size, image.mode)

def draw_board(screen, selected_square, game, flipped_board):
    colors = [WHITE, BLACK]
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            row_aux = row
            col_aux = col
            if flipped_board[0]:
                row_aux = 7 - row_aux
                col_aux = 7 - col_aux
            color = colors[(row_aux + col_aux) % 2]
            if selected_square == (row_aux, col_aux):
                color = HIGHLIGHT
            pygame.draw.rect(screen, color, pygame.Rect(col_aux * SQUARE_SIZE, row_aux * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            piece = game.piece_at(chess.square(col, 7 - row))
            piece_image = get_piece_image(piece)
            if piece_image:
                piece_rect = piece_image.get_rect(center=(col_aux * SQUARE_SIZE + SQUARE_SIZE // 2, row_aux * SQUARE_SIZE + SQUARE_SIZE // 2))
                screen.blit(piece_image, piece_rect.topleft)

def switch_turns(game):
    game.push(chess.Move.null())
    
def draw_analysis_tab(screen, show_analysis, analysis):
    analysis_rect = pygame.Rect(HEIGHT, 0, WIDTH-HEIGHT, HEIGHT)
    pygame.draw.rect(screen, BLACK, analysis_rect)
    
    font = pygame.font.Font(None, 36)
    
    analyze_button_rect = pygame.Rect(HEIGHT + 10, 10, 180, 50)
    pygame.draw.rect(screen, WHITE, analyze_button_rect)
    analyze_button_text = font.render("Analyze", True, BLACK)
    screen.blit(analyze_button_text, (analyze_button_rect.x + 10, analyze_button_rect.y + 10))
    
    undo_button_rect = pygame.Rect(HEIGHT + 10, HEIGHT-70, 180, 50)
    pygame.draw.rect(screen, WHITE, undo_button_rect)
    undo_button_text = font.render("Undo", True, BLACK)
    screen.blit(undo_button_text, (undo_button_rect.x + 10, undo_button_rect.y + 10))

    switch_rect = pygame.Rect(HEIGHT + 10, HEIGHT-130, 180, 50)
    pygame.draw.rect(screen, WHITE, switch_rect)
    undo_button_text = font.render("Switch turn", True, BLACK)
    screen.blit(undo_button_text, (switch_rect.x + 10, switch_rect.y + 10))
    
    edit_board = pygame.Rect(HEIGHT + 10, HEIGHT-190, 180, 50)
    pygame.draw.rect(screen, WHITE, edit_board)
    undo_button_text = font.render("Edit position", True, BLACK)
    screen.blit(undo_button_text, (switch_rect.x + 10, edit_board.y + 10))

    flip_button = pygame.Rect(HEIGHT + 10, HEIGHT-250, 180, 50)
    pygame.draw.rect(screen, WHITE, flip_button)
    undo_button_text = font.render("Flip board", True, BLACK)
    screen.blit(undo_button_text, (flip_button.x + 10, flip_button.y + 10))

    #analysis = None
    if show_analysis:
        # Draw analysis
        text_lines = str(analysis).split("\n")
        font = pygame.font.Font(None, 24)
        for i, line in enumerate(text_lines):
            text_surface = font.render(line, True, WHITE)
            screen.blit(text_surface, (HEIGHT + 10, 130 + i * 30))
    return analysis


def draw_editing_tab(screen, editing_position):
    editing_rect = pygame.Rect(HEIGHT, 0, WIDTH-HEIGHT, HEIGHT)
    pygame.draw.rect(screen, BLACK, editing_rect)
    font = pygame.font.Font(None, 36)

    set_position = pygame.Rect(HEIGHT + 10, 10, 180, 50)
    pygame.draw.rect(screen, WHITE, set_position)
    analyze_button_text = font.render("Done", True, BLACK)
    screen.blit(analyze_button_text, (set_position.x + 10, set_position.y + 10))

    pieces = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']
    for i, piece in enumerate(pieces):
        #piece_rect = pygame.Rect(WIDTH + 10, 10 + i * 60, 50, 50)
        pygame.draw.rect(screen, WHITE, piece_rect)
        piece_image = get_piece_image(chess.Piece.from_symbol(piece))
        if piece_image:
            if i > 5:
                piece_rect = piece_image.get_rect(center=(HEIGHT + 70 + 25, (10 + (i-6) * 60) + 85))
            else:
                piece_rect = piece_image.get_rect(center=(HEIGHT + 10 + 25, (10 + i * 60) + 85))
            screen.blit(piece_image, piece_rect.topleft)
    
    invert_button = pygame.Rect(HEIGHT + 10, HEIGHT-70, 180, 50)
    pygame.draw.rect(screen, WHITE, invert_button)
    undo_button_text = font.render("Invert_position", True, BLACK)
    screen.blit(undo_button_text, (invert_button.x + 10, invert_button.y + 10))

def get_inverse(fen):
    #fen = 'RNBK1B1R/PPPPQPPP/5N2/3pP3/4p1p1/2n2n2/ppp2p1p/r1bkqb1r b'
    fields = fen.split(' ')
    fields[0] = fields[0][::-1]
    flipped_fen = ' '.join(fields)
    return flipped_fen

def handle_editing_tab_click(pos,game, editing_position):
    x,y = pos
    editing_position[1] = None
    if x >= HEIGHT + 10 and y >= 10 and x <= HEIGHT + 190 and y <= 60:
        editing_position[0] = False
    elif x >= HEIGHT + 10 and y >= HEIGHT - 70 and x <= HEIGHT + 190 and y <= HEIGHT - 20:
        game.set_fen(get_inverse(game.fen()))
    half_piece_size = 30
    piece = None
    if x >= HEIGHT + 10 + 25 - half_piece_size and x <= HEIGHT + 10 + 25 + half_piece_size:
        if y >= 10 + 85 - half_piece_size and y <= 10 + (6*60) + half_piece_size:
            piece_index = round((y - 95) / 60)
            piece = ['r', 'n', 'b', 'q', 'k', 'p'][piece_index]
    if x >= HEIGHT + 70 + 25 - half_piece_size and x <= HEIGHT + 70 + 25 + half_piece_size:
        if y >= 10 + 85 - half_piece_size and y <= 10 + (6*60) + half_piece_size:
            piece_index = round((y - 95) / 60)
            piece = ['R', 'N', 'B', 'Q', 'K', 'P'][piece_index]
    if piece == None:
        return
    editing_position[1] = chess.Piece.from_symbol(piece)

def handle_analysis_tab_click(pos,game, editing_position, flipped_board):
    x,y = pos
    if x >= HEIGHT + 10 and y >= 10 and x <= HEIGHT + 190 and y <= 60:
        return True
    elif x >= HEIGHT + 10 and y >= HEIGHT - 70 and x <= HEIGHT + 190 and y <= HEIGHT - 20:
        try:
            game.pop()
        except:
            pass
    elif x >= HEIGHT + 10 and y >= HEIGHT - 130 and x <= HEIGHT + 190 and y <= HEIGHT - 80:
        switch_turns(game)
    elif x >= HEIGHT + 10 and y >= HEIGHT - 190 and x <= HEIGHT + 190 and y <= HEIGHT - 140:
        editing_position[0] = True
    elif x >= HEIGHT + 10 and y >= HEIGHT - 250 and x <= HEIGHT + 190 and y <= HEIGHT - 200:
        flipped_board[0] = not flipped_board[0]
        #print("flipped")    
    return False


def undid(pos):
    x,y = pos
    if x >= HEIGHT + 10 and y >= HEIGHT - 70 and x <= HEIGHT + 190 and y <= HEIGHT - 20:
        return True
    return False

def press(selected_square, previous_square, game, editing_position,flipped_board, promotion):
    promotion[0] = None
    promotion[1] = None
    if selected_square == previous_square and not editing_position[0]:
        return False
    if flipped_board[0]:
        selected_square = (7 - selected_square[0], 7 - selected_square[1])
        previous_square = (7 - previous_square[0], 7 - previous_square[1])
    selected_square = chess.SQUARE_NAMES[selected_square[1] + 8 * (7 - selected_square[0])]
    
    if editing_position[0] and not editing_position[1] is None:
        game.set_piece_at(chess.parse_square(selected_square), editing_position[1])
        editing_position[1] = None
        return True
    
    previous_square = chess.SQUARE_NAMES[previous_square[1] + 8 * (7 - previous_square[0])]
    ##print(chess.parse_square(selected_square))
    if editing_position[0]:
        if editing_position[1] is None:
            piece = game.piece_at(chess.parse_square(previous_square))
            game.set_piece_at(chess.parse_square(selected_square), piece)
            game.remove_piece_at(chess.parse_square(previous_square))
            return True
    else:
        #print("press->")
        #print(selected_square)
        move = chess.Move.from_uci(previous_square+selected_square)
        if (selected_square[-1] == "8" or selected_square[-1] == "1") and (game.piece_at(chess.parse_square(previous_square)).symbol() == "p" or game.piece_at(chess.parse_square(previous_square)).symbol() == "P"):   
            #move = chess.Move.from_uci(previous_square+selected_square+"q")
            #print("setting promotion")
            promotion[0] = previous_square
            promotion[1] = selected_square
        if move in game.legal_moves:
            game.push(move)
            return True
        return False

def is_promotion(pos,promotion,flipped_board):

    #print("is it promotion?")
    selected_square = get_square_under_mouse(pos)
    if flipped_board[0]:
        selected_square = (7 - selected_square[0], 7 - selected_square[1])
    return promotion[1] == chess.SQUARE_NAMES[selected_square[1] + 8 * (7 - selected_square[0])]

def display_promotion(screen,promotion,game,flipping_board):
    #{"tr":"q", "tl":"r","bl":"n","br":"b"}
    ##print("displaying_promotion")
    ##print(promotion[1])
    col = ord(promotion[1][0])-ord("a")
    
    row = 7-(int(promotion[1][-1])-1)
    if flipping_board[0]:
        row = 7-row
        col = 7-col
    ##print(col,row)
    options = ['R', 'Q', 'N', 'B']
    if game.turn == chess.BLACK:
        options = ['r', 'q', 'n', 'b']

    top_center = row*SQUARE_SIZE + (SQUARE_SIZE // 4)
    bot_center =  row*SQUARE_SIZE + (SQUARE_SIZE) // 4 + (SQUARE_SIZE) // 2
    left_center = col*SQUARE_SIZE + (SQUARE_SIZE // 4)
    right_center = col*SQUARE_SIZE + (SQUARE_SIZE) // 4+ (SQUARE_SIZE) // 2
    #rook
    pygame.draw.rect(screen,WHITE,pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE//2, SQUARE_SIZE//2))
    piece_image1 = get_small_piece_image(chess.Piece.from_symbol(options[0]))
    piece_rect = piece_image1.get_rect(center=(left_center,top_center))
    screen.blit(piece_image1, piece_rect.topleft)
    #queen
    pygame.draw.rect(screen,WHITE,pygame.Rect(col * SQUARE_SIZE + SQUARE_SIZE//2, row * SQUARE_SIZE , SQUARE_SIZE//2, SQUARE_SIZE//2))
    piece_image2 = get_small_piece_image(chess.Piece.from_symbol(options[1]))
    piece_rect = piece_image2.get_rect(center=(right_center,top_center))
    screen.blit(piece_image2, piece_rect.topleft)
    #knight
    pygame.draw.rect(screen,WHITE,pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE+SQUARE_SIZE//2, SQUARE_SIZE//2, SQUARE_SIZE//2))
    piece_image3 = get_small_piece_image(chess.Piece.from_symbol(options[2]))
    piece_rect = piece_image3.get_rect(center=(left_center,bot_center))
    screen.blit(piece_image3, piece_rect.topleft)
    #bishop
    pygame.draw.rect(screen,WHITE,pygame.Rect(col * SQUARE_SIZE+SQUARE_SIZE//2, row * SQUARE_SIZE+SQUARE_SIZE//2, SQUARE_SIZE//2, SQUARE_SIZE//2))
    piece_image = get_small_piece_image(chess.Piece.from_symbol(options[3]))
    piece_rect = piece_image.get_rect(center=(right_center,bot_center))
    screen.blit(piece_image, piece_rect.topleft)


def get_promotion_piece(pos,flipped_board):
    selected_square = get_square_under_mouse(pos)
    #if flipped_board[0]:
    #    selected_square = (7 - selected_square[0], 7 - selected_square[1])
    x, y = pos
    x_middle = selected_square[1]*SQUARE_SIZE + SQUARE_SIZE // 2
    y_middle = selected_square[0]*SQUARE_SIZE + SQUARE_SIZE // 2
    aux = {"tr":"q", "tl":"r","bl":"n","br":"b"}
    key = ""
    #print(x)
    #print(x_middle)
    if y > y_middle:
        key += "b"
    else:
        key += "t"
    if x > x_middle:
        key = key +"r"
    else:
        key = key +"l"
    return aux[key]

def thread_analyse_position(engine, board):
    def x(engine,board,q):
        q.put(analyse_position(board, engine))
    q = queue.Queue()
    analysis_thread = threading.Thread(target=x, args=(engine,board,q,))
    analysis_thread.start()
    analysis_thread.join()
    return q.get()

def main(fen=None):
    print("Starting GUI")
    pygame.init()
    game = chess.Board()
    if fen != None:
        game.set_fen(fen)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess GUI')

    engine = None
    editing_position = [False, None]
    selected_square = None
    previous_square = None
    last_analysis_results = None
    show_analysis = False
    updated = False
    promotion = [None,None,None]
    flipped_board = [False]
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if in_analysis_tab(pos):
                    if editing_position[0]:
                            handle_editing_tab_click(pos,game, editing_position)

                    else:
                        if handle_analysis_tab_click(pos,game, editing_position,flipped_board):
                            show_analysis = not show_analysis
                        if undid(pos):
                            updated = False
                else:
                    previous_square = selected_square
                    selected_square = get_square_under_mouse(pos)
                    if is_promotion(pos, promotion,flipped_board):
                        #print("true")
                        move = chess.Move.from_uci(promotion[0]+promotion[1] + get_promotion_piece(pos,flipped_board))
                        #print(move)
                        game.push(move)
                        selected_square = None
                        updated = False
                    promotion[0] = None
                    promotion[1] = None

                    if editing_position[0] and not editing_position[1] is None:
                        if press(selected_square,previous_square, game, editing_position,flipped_board,promotion):
                            updated = False
                    elif previous_square != None:
                        if press(selected_square,previous_square, game, editing_position,flipped_board,promotion):
                            updated = False
                        selected_square = None
                ##print(f"Selected square: {selected_square}")

        if not updated and show_analysis:
            if engine is None:
                engine=get_engine()
            last_analysis_results = thread_analyse_position(engine, game)
            ##print(last_analysis_results)
            updated = True
        screen.fill(WHITE)
        draw_board(screen, selected_square,game,flipped_board)
        # if clicked on analysis button show analysis
        if editing_position[0]:
            draw_editing_tab(screen, editing_position)
        else:
            draw_analysis_tab(screen, show_analysis, last_analysis_results)
        if not (promotion[0] is None):
            display_promotion(screen,promotion,game,flipped_board)
        pygame.display.flip()
if __name__ == "__main__":
    main()
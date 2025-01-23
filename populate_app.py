import os
"""
This is used to add the puzzles to app EnCroissant, ill keep it as it is still usefull, but usage is optional.
"""

FILE_DOT_INFO = """{"type":"puzzle","tags":[]}"""

## path to the path where encroissant stores things
PATH_TO_PUZZLES = "/home/atoste/Documents/EnCroissant"

def fen_to_index_innit():
    files = os.listdir(PATH_TO_PUZZLES)
    fen_to_index = {}
    for file in files:
        if "woodpecker_" in file and ".pgn" in file:
            with open(f"{PATH_TO_PUZZLES}/{file}", "r") as f:
                for fen in f:
                    if("[FEN \"" in fen):
                        fen = fen.replace("[FEN \"", "")
                        fen = fen.replace("\"]", "")
                        file = file.replace("woodpecker_", "")
                        file = file.replace(".pgn", "")
                        fen_to_index[fen] = int(file)
    return fen_to_index

fen_to_index = fen_to_index_innit()

def get_last_index():
    # open path to puzzles and get the file with the biggest number in the name~
    files = os.listdir(PATH_TO_PUZZLES)
    x = 1
    for file in files:
        if "woodpecker_" in file and ".pgn" in file:
            file = file.replace("woodpecker_", "")
            file = file.replace(".pgn", "")
            if(int(file) > x):
                x = int(file)
    return x

def get_next_index(index):
    if((index) % 3 == 0):
        return index  + 4
    return index + 1




def add_puzzle(fen, index):
    if(fen == None):
        return False
    global fen_to_index
    if(fen in fen_to_index):
        print("fen exists:")
        print(fen_to_index[fen])
        return False
    fen_to_index[fen] = index
    # add to file
    index_string = str(index).zfill(4)
    with open(f"{PATH_TO_PUZZLES}/woodpecker_{index_string}.pgn", "a") as f:
        pgn = "[Event \"?\"]\n[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"?\"]\n[White \"?\"]\n[Black \"?\"]\n[Result \"*\"]\n[SetUp \"1\"]\n[FEN \"" + fen + "\"\" w - - 0 1\"]\n*"
        
        print("################")
        print("################")
        print(pgn)
        print("################")
        print("################")
        f.write(pgn)
    with open(f"{PATH_TO_PUZZLES}/woodpecker_{index_string}.info", "a") as f:
        f.write(FILE_DOT_INFO)
    return True
    
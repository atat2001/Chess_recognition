import chess.engine
import chess

NR_MOVES = 3

def get_engine():
    path = "/usr/local/bin/stockfish/stockfish-ubuntu-x86-64-avx2"
    # Connect to the Stockfish engine
    engine =  chess.engine.SimpleEngine.popen_uci(path)
    return engine

# Stockfish engine automatically closes when exiting the 'with' block
def analyse_position(board,engine):
    if board.is_checkmate():
        return "Checkmate!"
    # Analyze the position
    result = engine.analyse(board, chess.engine.Limit(time=0.2), multipv=3)
    output = ""
    # Print the top 3 lines
    for _, info in enumerate(result):
        copy_board = chess.Board(fen=board.fen())
        score = info['score'].white().score(mate_score=1000)
        if score < 950:
            score = str(score/100)
        else:
            score = "M" + str(1000 - score) 
        output += score + "  "
        print(info['pv'])
        for x in range(2*NR_MOVES):
            if x >= NR_MOVES or 2*x+1 >= len(info['pv']):
                if 2*x < len(info['pv']):
                    output += str(x+1) + " " + str(copy_board.san(info['pv'][2*x]))
                break
            #print(info['pv'][2*x])
            #print(info['pv'][2*x+1])
            output += str(x+1) + " " + str(copy_board.san(info['pv'][2*x])) + " "
            copy_board.push(info['pv'][2*x])
            output += str(copy_board.san(info['pv'][2*x+1])) + " " 
            copy_board.push(info['pv'][2*x+1])
        output += "\n"
    print(output)
    return output

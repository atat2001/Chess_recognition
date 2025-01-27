# Chess Recognition App

This application identifies chessboards on your screen and opens a simple, interactive analysis board.

## How It Works

1. **Contour Analysis**: The app scans your screen for shapes resembling chessboards.
2. **Chessboard Classification**: A Convolutional Neural Network (CNN) classifies these shapes as either `chessboard` or `not_chessboard`.
3. **Square Analysis**: For each identified chessboard, the app divides it into 64 squares. Another CNN determines what type of chess piece, if any, is in each square.
4. **FEN Generation**: Based on the detected pieces, the app generates a FEN (Forsyth-Edwards Notation) string to represent the position.
5. **Interactive GUI**: The app launches an interface that allows you to analyze the position using a chess engine.

Additionally, the app includes a **Trainer GUI** to improve the piece recognition model and the board recognition model. It displays first the square contours detected, you have to make sure every chessboard has a green square. The second trainer then displays the position predicted by the CNN and lets the user correct it. This simplifies the process of generating high-quality data for fine-tuning. After corrections, the user is directed to the standard analysis GUI.

You should correct the fens for the program to be able to learn (and of course train the program), if you dont correct it you will be feeding wrong information to the program.

## Running the Program

To run the program:

1. Execute the following command:
   ```bash
   python3 dynamic_chessboard_finder.py
   ```
2. Press the "ç" key (configurable in `dynamic_chessboard_finder.py` on line 29) to open the trainer tool and then the analysis tool with the detected chessboard from your screen.

   I couldn't upload the pre-trained fen generation weights so first it is required to train the CNN.

### Dependencies

- Ensure all required Python libraries are installed.
- For analysis, install the **Stockfish chess engine** and specify the path to its executable in `gui/analyse.py` (line 7).

---

Hope you like my project and find it useful.

# Chess Recognition App

This application identifies chessboards on your screen and opens a simple, interactive analysis board.

## How It Works

1. **Contour Analysis**: The app scans your screen for shapes resembling chessboards.
2. **Chessboard Classification**: A Convolutional Neural Network (CNN) classifies these shapes as either `chessboard` or `not_chessboard`.
3. **Square Analysis**: For each identified chessboard, the app divides it into 64 squares. Another CNN determines what type of chess piece, if any, is in each square.
4. **FEN Generation**: Based on the detected pieces, the app generates a FEN (Forsyth-Edwards Notation) string to represent the position.
5. **Interactive GUI**: The app launches an interface that allows you to analyze the position using a chess engine.

Additionally, the app includes a **Trainer GUI** to improve the piece recognition model. It displays the position predicted by the CNN and lets the user correct it. This simplifies the process of generating high-quality data for fine-tuning. After corrections, the user is directed to the standard analysis GUI.

If you want to make the fens better quality on the begining consider using the scrape_fen to get the fen generated by the website, this method is slower but in the beggining it is faster than setting every position up from scratch.

## Features

- **Pre-trained for "The Woodpecker Method" book**: The app is ready to recognize chessboards from this specific resource but can be retrained for other books or sources.

---

## Running the Program

To run the program:

1. Execute the following command:
   ```bash
   python3 dynamic_chessboard_finder.py
   ```
2. Press the "ç" key (configurable in `dynamic_chessboard_finder.py` on line 29) to open the analysis tool with the detected chessboard from your screen.

### Dependencies

- Ensure all required Python libraries are installed.
- For analysis, install the **Stockfish chess engine** and specify the path to its executable in `gui/analyse.py` (line 7).

---

## Training the Models

If you want to use the program with other books or sources, you need to retrain two models:

1. **Chessboard Detection (detect\_board)**:

   - Use `dynamic_chessboard_finder` as it will save its classification attempts to chessboard_try and not_chessboard_try to classify images into `chessboard` or `not_chessboard` categories.
   - Organize images from `chessboard_try` and `not_chessboard_try` into their respective folders.
   - Delete any folders in the `data` directory that aren't `chessboards` or `not_chessboards`.
   - Run `train_ml.py` to train the model.
   - Repeat the process as needed.

2. **Piece Recognition (tactic\_to\_fen)**:

   - Run `dynamic_chessboard_finder` and use the Trainer GUI to match the board to the image.
   - Click "Done" or "Train" (both save the data, but "Train" also fine-tunes the CNN).
   - Use `training_data_manager` to:
     - Delete erroneous data entries from the fine-tuning data(all data generated by the trainer will go to the fine-tuning dataset after pressing done or train).
     - Move corrected data from the fine-tuning dataset to the main training dataset.

Hope you like my project and find it usefull.
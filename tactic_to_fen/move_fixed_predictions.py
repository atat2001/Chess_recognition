import os
import shutil
current_dir = os.path.dirname(os.path.abspath(__file__))

prediction_folder = os.path.join(current_dir,'data_try')
prediction_folders = os.listdir(prediction_folder)
data_folder = os.path.join(current_dir,'data')
for x in prediction_folders:
    for file in os.listdir(prediction_folder + "/" + x):
        src = prediction_folder + "/" + x + "/" + file
        dest = data_folder + "/" + x + "/" + file
        shutil.move(src, dest)
        print(f"Moved {src} to {dest}")
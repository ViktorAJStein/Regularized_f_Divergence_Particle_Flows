from PIL import Image
import torch
import os
import numpy as np

def get_timestamp(file_name):
    return int(file_name.split('-')[-1].split('.')[0])

def create_gif(image_folder, output_gif):
    images = []
    for filename in sorted(os.listdir(image_folder), key=get_timestamp):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(image_folder, filename))
            images.append(img)
            

    if images:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=70,  # You can adjust the duration between frames (in milliseconds) here
            loop=0  # 0 means infinite loop, change it to the number of loops you want
        )
        print(f"GIF saved as {output_gif}")
    else:
        print("No PNG images found in the folder.")



def make_folder(name):
    try:
        os.mkdir(name)
        print(f"Folder '{name}' created successfully.")
    except FileExistsError:
        print(f"Folder '{name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}.")
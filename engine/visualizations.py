import os
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from IPython.display import HTML, display


def create_and_display_animation(image_dir, output_gif, fps=3, figsize=(6, 6), image_size=(800, 600)):
    # Get a list of image file paths in the specified directory
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))

    if not image_files:
        print(f"No image files found in {image_dir}.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(Image.open(image_files[0]), animated=True, aspect='auto', resample=True)
    
    def update(frame):
        im.set_data(Image.open(image_files[frame]))
    
    animation = FuncAnimation(fig, update, frames=len(image_files), repeat=False, blit=False, interval=1000/fps)
    animation.save(f"{output_gif}.gif", writer='pillow', fps=fps)
    # Display animation in Jupyter Notebook using to_jshtml
    display(HTML(animation.to_jshtml(default_mode='loop')))
    
    
    

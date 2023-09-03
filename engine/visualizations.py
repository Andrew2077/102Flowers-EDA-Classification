import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_saved_figs(frames, dir, file_name):
    figs = []

    # Generate and save figures for each iteration
    for frame in frames:
        #* read image
        fig = plt.imread(f'{dir}/{file_name}_{frame}.png')
        figs.append(fig)

    # Create an animation using the saved figures
    def update(frame):
        plt.clf()
        plt.imshow(plt.imread(f'{dir}/iteration_{frame}.png'))
        plt.axis('off')

    anim = FuncAnimation(plt.gcf(), update, frames=frames, interval=1000)
    anim.save(f'{dir}/animation.gif', writer='pillow')
    
    

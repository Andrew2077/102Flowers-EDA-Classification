import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
from PIL import Image


matplotlib.rcParams["font.family"] = "serif"

#* Creates GIF from a directory of images
def create_and_display_animation(
    image_dir,
    output_gif,
    sort_key,
    fps=3,
    figsize=(6, 6),
    image_size=(800, 600),
    save = False
):
    # Get a list of image file paths in the specified directory
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    image_files = sorted(image_files, key=sort_key)
    # print(image_files)

    if not image_files:
        print(f"No image files found in {image_dir}.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    im = ax.imshow(
        Image.open(image_files[0]), animated=True, aspect="auto", resample=True
    )

    def update(frame):
        im.set_data(Image.open(image_files[frame]))

    animation = FuncAnimation(
        fig,
        update,
        frames=len(image_files),
        repeat=False,
        blit=False,
        interval=1000 / fps,
    )
    if save:
        animation.save(f"{output_gif}.gif", writer="pillow", fps=fps)
    # Display animation in Jupyter Notebook using to_jshtml
    display(HTML(animation.to_jshtml(default_mode="loop")))

#* prepare random image for plot
def prepare_img_plot(df, label):
    path = df[df["labels"] == label]["image_path"].sample(1).values[0]
    # path = df[df["labels"] == label]["image_path"].values[0]
    img = Image.open(path)
    img = img.resize((300, 300))
    img = np.array(img)
    # #* convert to RGB
    # img = img[:, :, :3]
    return img

def grad_cam_plot(
    class_dict, true_label, top_4_classes, top_4_probs, origional_img, heatmap, colored_heatmap, overlayed, title, fig_size = (10, 5)
):
    matplotlib.rcParams["font.family"] = "serif"
    plt.style.use("seaborn-v0_8")
    # * create the figure
    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.5, 0.5, 0.7])

    # * image plot
    img_plot = fig.add_subplot(gs[0, 0])
    img_plot.axis("off")
    img_plot.imshow(origional_img)
    img_plot.set_title(f"True label: {class_dict[str(true_label)]}")

    # * cam plot
    cam_plot = fig.add_subplot(gs[0, 1])
    cam_plot.axis("off")
    cam_plot.imshow(heatmap, cmap="gray")
    cam_plot.set_title("GradCAM")

    # * colored cam plot
    colored_cam_plot = fig.add_subplot(gs[1, 0])
    colored_cam_plot.axis("off")
    colored_cam_plot.imshow(colored_heatmap)
    colored_cam_plot.set_title("Colored GradCAM")

    # * overlayed plot
    overlayed_plot = fig.add_subplot(gs[1, 1])
    overlayed_plot.axis("off")
    overlayed_plot.imshow(overlayed)
    overlayed_plot.set_title(f"Pred label: {class_dict[str(top_4_classes[0])]}")

    classes_names = [class_dict[str(i)] for i in top_4_classes]
    bar_plot = fig.add_subplot(gs[:, 2])
    # bar_plot.barh(y = classes_names, width = top_4_probs)
    bar_plot.bar(x=classes_names, height=top_4_probs)
    bar_plot.set_ylabel("Probability")
    bar_plot.set_xlabel("Class")
    bar_plot.set_title("Top 4 classes probabilities")

    # * make the grid look nicer
    for ax in fig.axes:
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color="gray", linestyle="dashed")
        ax.yaxis.grid(color="gray", linestyle="dashed")

    # * rotate the xticks
    plt.setp(
        bar_plot.get_xticklabels(), rotation=-30, ha="left", rotation_mode="anchor"
    )

    # * add space between subplots
    fig.tight_layout(pad=0.5)
    # * white fig background
    fig.patch.set_facecolor("white")
    
    #* fig title
    fig.suptitle(title, fontsize=20, y=1.1)
    
    
def plot_all_feat_cam(gradcam, image_tensor, device, true_label, font_color = 'black', font_size = 10, save =False):

    #* generate all the feature maps
    feat_cams = [
        gradcam.generate_cam(image_tensor.to(device), label) for label in range(0, 102)
    ]
    #* adjust the images
    origional_img, _, overlayed_image = gradcam.adjust_cam_images(image_tensor, feat_cams[0])

    #* build the figure
    fig, (ax_img, ax_cam, ax_overlay) = plt.subplots(1, 3, figsize=(5, 3))

    #* original image axis
    ax_img.axis("off")
    ax_img.imshow(origional_img)
    ax_img.set_title(f"Original Image - {true_label}", fontsize=font_size, color=font_color)

    #* cam heatmap axis
    ax_cam.axis("off")
    cam_map = ax_cam.imshow(feat_cams[0], cmap="jet")

    #* overlayed image axis
    ax_overlay.axis("off")
    overlayed = ax_overlay.imshow(overlayed_image)

    def update(frame):
        #* update the cam heatmap
        cam_map.set_data(feat_cams[frame])
        cam_map.axes.set_title(f"CAM for label: {frame+1}", fontsize=font_size, color=font_color)
        
        #* generate the overlayed image
        _, _, overlayed_image = gradcam.adjust_cam_images(image_tensor, feat_cams[frame])
        overlayed.set_data(overlayed_image)
        overlayed.axes.set_title(f"Overlayed image : {frame+1}", fontsize=font_size, color=font_color)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(feat_cams),
        repeat=False,
        blit=False,
        interval=1000 / 5,
    )

    
    display(HTML(anim.to_jshtml(default_mode="loop")))
    if save:
        anim.save(f"figs/gradcam/all_feat_cam.gif", writer="pillow", fps=5)


def plot_sample(df, start_idx, end_idx, classes, classes_count, font_color="black", save=True):

    classes_count = df["labels"].value_counts()
    images = [prepare_img_plot(df, label) for label in range(start_idx, end_idx)]
    labels = [classes[str(label)] for label in range(start_idx, end_idx)]
    bar_data = [classes_count[label] for label in range(start_idx, end_idx)]

    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.5, 0.5, 1])
    # * add more width to the last column

    for row in range(2):
        for col in range(2):
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")
            ax.imshow(images[row * 2 + col])
            ax.set_title(labels[row * 2 + col])
            # * title font color
            ax.title.set_color(font_color)

    bar_plot = fig.add_subplot(gs[:, 2])
    bar_plot.bar(labels, bar_data)
    bar_plot.set_xlabel("Class")
    # * put the xlabel up the bar
    bar_plot.xaxis.set_label_coords(0.5, 1.1)
    bar_plot.set_ylabel("Count")

    # * rotate the xticks
    plt.setp(
        bar_plot.get_xticklabels(), rotation=-30, ha="left", rotation_mode="anchor"
    )
    # * make bar look nicer
    bar_plot.spines["top"].set_visible(False)
    bar_plot.spines["right"].set_visible(False)
    bar_plot.spines["left"].set_visible(False)
    bar_plot.spines["bottom"].set_visible(False)
    bar_plot.tick_params(axis="x", length=0)
    bar_plot.tick_params(axis="y", length=0)

    # * make the grid look nicer
    for ax in fig.axes:
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color="gray", linestyle="dashed")
        ax.yaxis.grid(color="gray", linestyle="dashed")

    # * fig title
    fig.suptitle(
        f"Sample & distribution of classes from: {start_idx} to {end_idx - 1}",
        fontsize=16,
        y=1,
        color=font_color,
    )
    # * white fig background
    fig.patch.set_facecolor("white")
    # * save the figure
    if save:
        fig.savefig(
            f"figs/samples/sample_{start_idx}_{end_idx}.png", dpi=150, bbox_inches="tight"
        )

        print(
            "saved the figure with name: ",
            f"figs/samples/frames/sample_{start_idx}_{end_idx}.png",
        )
    
    else:
        plt.show()
        



if __name__ == "__main__":
    from engine.data_processing import prepare_df, FlowerDataset
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")

    # import matplotlib
    # plt.style.use('ggplot')

    data_path = r"data/102flowers/jpg"
    label_path = r"data/imagelabels.mat"
    split_path = r"data/setid.mat"
    train_df, _, _ = prepare_df(split_path, label_path, data_path)
    with open("config/flower_to_name.json") as f:
        flower_to_name = json.load(f)
    # plot_sample(train_df, 1, 5, flower_to_name, train_df["labels"].value_counts())
    for i in range(102 // 4):
        plot_sample(
            train_df,
            i * 4 + 1,
            i * 4 + 4 + 1,
            flower_to_name,
            train_df["labels"].value_counts(),
        )

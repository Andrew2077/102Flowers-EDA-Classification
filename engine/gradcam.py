from typing import Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import torch
import torchvision
from PIL import Image
from torchvision import models, transforms


class GradCAM:
    def __init__(self, model, class_dict):
        self.class_dict = class_dict
        # * set model to eval
        self.model = model
        self.model.eval()
        # * set gradient var
        self.gradients = None
        self.device = model.device

    def forward(self, x):
        self.model.zero_grad()
        x.requires_grad_()
        out = self.model(x)
        return out

    def preprocess_image(
        self, img: Optional[str or torch.Tensor] = None
    ) -> torch.Tensor:
        if isinstance(img, str):
            img = Image.open(img)

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_tensor = preprocess(img).unsqueeze(0)
        # * img_tensor_shape = torch.Size([1, 3, 224, 224])
        return img_tensor.to(self.device)

    def tensor_to_img(
        self, img: torch.Tensor, image_type: str
    ) -> Optional[Image.Image or np.ndarray or None]:
        img = img.squeeze(0)
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img = img * np.array([0.229, 0.224, 0.225])
        img = img + np.array([0.485, 0.456, 0.406])
        img = img * 255.0
        img = img.astype(np.uint8)
        if image_type == "pil":
            img = Image.fromarray(img)
        elif image_type == "cv2":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            assert False, "image_type must be pil or cv2"
        return img

    def adjust_cam_images(self, image_tensor, cam):
        # * load original image and overlay gradcam, convert BGR to RGB
        origional_img = self.tensor_to_img(image_tensor, "cv2")
        origional_img = cv2.cvtColor(origional_img, cv2.COLOR_BGR2RGB)

        # * https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

        # * https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html
        overlayed_image = cv2.addWeighted(origional_img, 0.45, cam_heatmap, 0.55, 0)

        return origional_img, cam_heatmap, overlayed_image

    def forward_hook(self, module, input, output):
        self.model.feature_maps = output

    def save_gradient(self, grad):
        self.gradients = grad

    def backward_hook(self, module, grad_input, grad_output):
        # * you might need to change which output grad to use
        # * in our case we want the grad of the output of the last conv layer
        self.save_gradient(grad_output[0])

    def register_hooks(self, target_module=None):
        if target_module == None:
            # * retrieve the last conv layer for gradcam
            # * in resnet50 case layer4 - conv2d[512 to 2048]
            # * target_module = model.model.layer4[2].conv3
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    target_module = module

        # * register forward hook
        # * The hook will be called every time after :func:`forward` has computed an output.
        target_module.register_forward_hook(self.forward_hook)

        # * register backward hook
        # *The hook will be called every time the gradients with respect to module
        # * inputs are computed [backward_propagation].
        # target_module.register_backward_hook(self.backward_hook)
        target_module.register_full_backward_hook(self.backward_hook)

    def generate_cam(self, image_tensor, target_class):
        # * pass image to generate output tensor- shape [1, 102]
        output = self.forward(image_tensor)

        # * generate one hot vector - shape [1, 102]
        one_hot_output = torch.zeros(
            (1, output.size()[-1]), dtype=torch.float, device=self.device
        )

        # * activate only the selected call in the one hot vector
        one_hot_output[0][target_class] = 1

        # * backpropagate the one hot vector to get the gradients
        # * set the gradient of all other classes to be zero
        # * activate only the selected class in the one hot vector
        output.backward(gradient=one_hot_output)

        # * hooked gradients - shape [1, 2048, 7, 7]
        gradients = self.gradients.detach().cpu().numpy()

        # * hooked feature maps - shape [1, 2048, 7, 7]
        feature_maps = self.model.feature_maps.detach().cpu().numpy()

        # * gradient shape [1, 2048, 7, 7], mean over [7, 7] - shape [1, 2048]
        assert (
            np.mean(gradients, axis=(2, 3))[0, :] == np.mean(gradients, axis=(2, 3))[0]
        ).all(), "Adjust back to [0,:]"
        cam_weights = np.mean(gradients, axis=(2, 3))[0]

        # * shape [7, 7]
        cam = np.zeros(feature_maps.shape[2:], dtype=np.float32)

        # * loop over the channels and multiply the weights with the feature maps
        for i, weight in enumerate(cam_weights):
            # * for each channel multiply the weight with the feature map and add it to the cam
            # * CAM will be a weighed sum of the feature maps, where the weights are the gradients
            cam += weight * feature_maps[0, i, :, :]

        # * clip the negative values
        cam = np.maximum(cam, 0)
        # * resize the cam to the original image size
        cam = cv2.resize(cam, (224, 224))
        # * normalize the cam, so that the values are between 0 and 1
        # * subtract the min value and divide by the max value
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

    def apply_grad_cam(self, image_tensor):
        # * register hooks for gradcam
        self.register_hooks()

        # * get the prediction from the model
        with torch.no_grad():
            output = self.model(image_tensor)
            # * apply softmax to get the probabilities of all classes
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # * get the top class and top probability
            topclass_prob, topclass = probabilities.topk(1, dim=1)

        # * generate gradcam
        # * generate cam - shape [224, 224]
        cam = self.generate_cam(image_tensor, topclass.item())
        origional_img, cam_heatmap, overlayed_image = self.adjust_cam_images(
            image_tensor, cam
        )

        return (
            topclass.item(),
            topclass_prob.item(),
            origional_img,
            cam,
            cam_heatmap,
            overlayed_image,
        )

    def save_grad_cam(self, image_tensor, target_class, frame, output_path):
        (
            top_class,
            top_prop,
            original_image,
            cam,
            cam_heatmap,
            overlayed_img,
        ) = self.apply_grad_cam(image_tensor)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Original Image subplot
        axs[0].imshow(original_image)
        axs[0].axis("off")
        axs[0].set_title("Original Image")

        # Grad-Cam subplot
        axs[1].imshow(cam_heatmap, cmap="hot")
        axs[1].axis("off")
        axs[1].set_title("Grad-Cam")

        # Colored Grad-Cam subplot
        axs[2].imshow(overlayed_img)
        axs[2].axis("off")
        axs[2].set_title(
            f"Overlayed: (True = {self.class_dict[str(target_class)]}, Pred = {self.class_dict[str(top_class+1)]})"
        )

        plt.suptitle(f"Grad-CAM Epoch: {frame+1}", fontsize=16)

        # Save the figure
        plt.savefig(f"{output_path}/gradcam_plt_{frame+1}.png", bbox_inches="tight")
        print(f"Saved Grad-CAM Epoch: {frame+1}")
        plt.close()


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    import json

    import matplotlib.pyplot as plt
    from data_processing import FlowerDataset, prepare_df, transformations
    from matplotlib.animation import FuncAnimation
    from models import Resnet50Flower102
    from torch.utils.data import DataLoader
    from utils import set_global_seed

    with open("config/global-configs.json") as f:
        global_configs = json.load(f)

    with open("config/flower_to_name.json") as f:
        flower_to_name = json.load(f)

    split_path = global_configs["dir"]["split_path"]
    labels_Path = global_configs["dir"]["labels_path"]
    data_root = global_configs["dir"]["data_path"]

    _, test_split, _ = prepare_df(split_path, labels_Path, data_root)

    test_dataset = FlowerDataset(test_split, transform=transformations)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    set_global_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grad_cam = GradCAM(Resnet50Flower102(device, True, False), flower_to_name)
    image_tensor, target_class = test_dataset[15]

    image_tensor, target_class = image_tensor.unsqueeze(0).to(device), target_class.to(
        device
    )
    for i in range(3):
        grad_cam.save_grad_cam(
            image_tensor, target_class.item(), i, "figs/gradcam/test"
        )

    # animate_saved_figs(range(3), "figs", "gradcam")

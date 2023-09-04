import json
import numpy as np
import torch
from typing import Optional
from PIL import Image
from torchvision import transforms


def set_global_seed(SEED: int) -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.set_num_threads(1)
    # torch.autograd.set_detect_anomaly(True)


def accuray_fn(ypred, y_true):
    """
    Calculate the accuracy of the prediction
    """
    return (ypred.argmax(dim=1) == y_true).sum() / len(ypred)


def load_configs(path):
    return json.load(open(path, "r"))


def load_class_dict(path=r"config/flower_to_name.json"):
    return json.load(open(path, "r"))


# Custom sorting function to extract the numeric part and sort numerically
def extract_numeric_part(file_path):
    # Split the file path by underscores and take the last part
    file_name = file_path.split("_")[-1]
    # Remove the file extension and any non-numeric characters
    numeric_part = "".join(filter(str.isdigit, file_name))
    return int(numeric_part)  # Convert to integer for numerical sorting


def preprocess_image(img: Optional[str or torch.Tensor] = None) -> torch.Tensor:
    if isinstance(img, str):
        img = Image.open(img)

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = preprocess(img).unsqueeze(0)
    # * img_tensor_shape = torch.Size([1, 3, 224, 224])
    return img_tensor

import torch
import random


def load_cam_model(path, GradCAM, Resnet50Flower102, classes_dict, device):
    # * load the model
    if path is None:
        raise ValueError("path must be a valid path to the model")
    model = Resnet50Flower102(device)

    # * load gradcam, update the model with the best weights
    gradcam = GradCAM(model, classes_dict)
    gradcam.model.load_state_dict(torch.load(path))
    gradcam.model.eval()
    print("model & gradcam loaded")

    return gradcam


def load_model(path, Resnet50Flower102):
    # * load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Resnet50Flower102(device)

    # * load the best weights
    model.load_state_dict(torch.load(path))
    model.eval()
    print("model loaded")

    return model


def sample_idx_to_tensor(df, sample_idx, gradcam):
    # * prepare list of images to choose from
    df["image_name"] = df["image_path"].apply(lambda x: x.split("/")[-1])
    image_list = df["image_name"].tolist()

    # * selecting an image from a list
    if (sample_idx == None) or (sample_idx > len(image_list)) or (sample_idx < 0):
        sample_idx = random.randint(0, len(image_list))
        print(
            f"sample_idx is set randomly to {sample_idx}, allowed range is [0, {len(image_list)}]"
        )
    else:
        selected_image = image_list[sample_idx]

    # * Grap the image path and label
    selected_image_path = df[df["image_name"] == selected_image]["image_path"].values[0]

    selected_image_label = df[df["image_name"] == selected_image]["labels"].values[0]

    # * pass the image to gradcam model
    # * convert image to tensor
    image_tensor = gradcam.preprocess_image(selected_image_path)
    return selected_image_label, selected_image_path, image_tensor


def prepare_sample_for_cam(df, sample_idx, gradcam, device):
    selected_image_label, selected_image_path, image_tensor = sample_idx_to_tensor(
        df, sample_idx, gradcam
    )

    # * inference mode, calculate the prediction probabilities of top 4 classes
    with torch.inference_mode():
        # * feed to the model
        classes_prop = torch.softmax(gradcam.model(image_tensor.to(device)), dim=1)
        top_4_probs, top_4_classes = torch.topk(classes_prop, 4)
        top_4_probs = top_4_probs.cpu().detach().numpy().squeeze()
        # * add 1 to match class_dict
        top_4_classes = top_4_classes.cpu().detach().numpy().squeeze() + 1

    # * apply gradcam to the image tensor
    (_, _, origional_img, heatmap, colored_heatmap, overlayed) = gradcam.apply_grad_cam(
        image_tensor.to(device)
    )
    return (
        selected_image_label,
        selected_image_path,
        top_4_probs,
        top_4_classes,
        origional_img,
        heatmap,
        colored_heatmap,
        overlayed,
    )

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import json


def create_writer(experiment_name: str, model_name: str, extra: str) -> str:
    """creates a tensorboard writer object

    Args:
        experiment_name (str): name of the experiment
        model_name (str): name of the model
        extra (str):  extra information

    Returns:
        str: path to the log directory
    """

    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra != "":
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    return SummaryWriter(log_dir=log_dir)


def set_experiment_params(config_path: str):
    configs = json.load(open(config_path, "r"))
    # * experiment hyperparameters
    expriment_name = configs["writer"]["experiment_name"]
    model_name = configs["writer"]["model_name"]
    extra = configs["writer"]["extra"]

    # * Gloabl hyperparameters
    PRETRAINED_WEIGHTS = configs["config"]["pretrained_weights"]
    FREEZE_RESNET = configs["config"]["freeze_resnet"]
    SEED = configs["config"]["seed"]
    DEVICE = configs["config"]["device"]

    # * Training hyperparameters
    TRAIN_BATCH_SIZE = configs["config"]["train_batch_size"]
    VALIDATION_BATCH_SIZE = configs["config"]["val_batch_size"]
    NUM_EPOCHS = configs["config"]["epochs"]
    LEARNING_RATE = configs["config"]["learning_rate"]
    SCHUFFLE_TRAIN = configs["config"]["schuffle_train"]
    SCHUFFLE_VALIDATION = configs["config"]["schuffle_val"]
    SCHUFFLE_TEST = configs["config"]["schuffle_test"]

    return (
        expriment_name,
        model_name,
        extra,
        PRETRAINED_WEIGHTS,
        FREEZE_RESNET,
        SEED,
        DEVICE,
        TRAIN_BATCH_SIZE,
        VALIDATION_BATCH_SIZE,
        NUM_EPOCHS,
        LEARNING_RATE,
        SCHUFFLE_TRAIN,
        SCHUFFLE_VALIDATION,
        SCHUFFLE_TEST,
    )

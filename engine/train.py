import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter()


def training_step(
    model,
    loss_fn,
    accuray_fn,
    optimizer,
    data_batch,
    target_batch,
    device,
    step_type="train",
):
    data_batch, target_batch = data_batch.to(device), (target_batch - 1).to(device)
    optimizer.zero_grad()

    if step_type == "train":
        model.train()
    elif step_type == "val":
        model.eval()

    else:
        raise ValueError("step_type must be either train or val")

    preds = model(data_batch)
    loss = loss_fn(preds, target_batch)

    if step_type == "train":
        loss.backward()
        optimizer.step()
    return loss.item(), accuray_fn(preds, target_batch)


def training_loop(
    model,
    gradcam,
    test_tensor,
    test_target,
    loss_fn,
    accuray_fn,
    optimizer,
    train_loader,
    val_loader,
    device,
    num_epochs,
    models_direcotry,
    model_name,
    tqdm_cols=None,
    writer=writer,
):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_acc = 0, 0
        epoch_val_loss, epoch_val_acc = 0, 0
        for image_batch, label_batch in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Train Epoch : {epoch+1}",
            leave=True,
            ncols=tqdm_cols,
            colour="green",
        ):
            train_loss, train_acc = training_step(
                model,
                loss_fn,
                accuray_fn,
                optimizer,
                image_batch,
                label_batch,
                device,
                step_type="train",
            )
            epoch_train_loss += train_loss
            epoch_train_acc += train_acc

        for image_batch, label_batch in tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"Val Epoch : {epoch + 1}",
            leave=True,
            ncols=tqdm_cols,
            colour="blue",
        ):
            val_loss, val_acc = training_step(
                model,
                loss_fn,
                accuray_fn,
                optimizer,
                image_batch,
                label_batch,
                device,
                step_type="val",
            )
            epoch_val_loss += val_loss
            epoch_val_acc += val_acc

        if len(history["val_loss"]) == 0:
            print("Saving first model...")
            # * saving model dict
            torch.save(
                model.state_dict(),
                models_direcotry + f"best_{model_name}.pth",
                # models_direcotry + f"model_{epoch+1}.pth",
            )
            # * saving optimizer dict
            # torch.save(
            #     optimizer.state_dict(),
            #     models_direcotry + f"optim_state.pth",
            # )

        elif len(history["val_loss"]) != 0:
            if (epoch_val_loss / len(val_loader)) < min(history["val_loss"]):
                torch.save(
                    model.state_dict(),
                    models_direcotry + f"best_{model_name}.pth",
                    # models_direcotry + f"model_{epoch+1}.pth",
                )
                # torch.save(
                #     optimizer.state_dict(),
                #     models_direcotry + f"optim_state.pth",
                # )
                print(
                    f"Validation loss decreased from {min(history['val_loss'])} to {epoch_val_loss / len(val_loader)}, saving model"
                )

        # ******************** Results Printing ********************#
        history["train_loss"].append(epoch_train_loss / len(train_loader))
        history["train_acc"].append(epoch_train_acc.item() / len(train_loader))
        history["val_loss"].append(epoch_val_loss / len(val_loader))
        history["val_acc"].append(epoch_val_acc.item() / len(val_loader))

        print(
            "---------------------------------------------------------------------------------------------------------------------"
        )
        print(
            f"Epoch {epoch+1} Train loss: {history['train_loss'][-1]:.6f} | Train acc: {(history['train_acc'][-1]*100):.4f}%"
        )
        print(
            f"Epoch {epoch+1} Val loss: {history['val_loss'][-1]:.6f} | Val acc: {(history['val_acc'][-1]*100):.4f}%"
        )
        print(
            "---------------------------------------------------------------------------------------------------------------------"
        )

        # ******************** tensorboard********************#
        # * Tensorboard scalars
        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={
                "train_loss": history["train_loss"][-1],
                "val_loss": history["val_loss"][-1],
            },
            global_step=epoch,
        )
        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={
                "train_acc": history["train_acc"][-1],
                "val_acc": history["val_acc"][-1],
            },
            global_step=epoch,
        )
        # * No graph saved to tensorboard
        # * ValueError: Modules that have backward hooks assigned can't be compiled: Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # writer.add_graph(
        #     model=model,
        #     input_to_model=torch.rand(64, 3, 224, 224).to(device),
        # )

        # * ******************** GradCAM ********************#

        fig = gradcam.plot_grad_cam(test_tensor, test_target.item())
        # #* show figure
        fig.update_layout(title_text=f"Grad-CAM at Epoch {epoch+1}", title_x=0.5)
        fig.update_layout(width=800, height=500)

        # Show the plot
        fig.show()

    writer.close()
    return history

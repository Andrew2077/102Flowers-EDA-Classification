import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

writer = None


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


def model_eval(model, loader, loss_fn, accuray_fn, optimizer, device):
    total_loss, total_acc = 0, 0
    for images, labels in tqdm(
        loader,
        total=len(loader),
        desc=f"Evaluating model",
        leave=True,
        ncols=100,
        colour="magenta",
    ):
        images = images.to(device)
        labels = labels.to(device)
        with torch.inference_mode():
            loss_value, acc_value = training_step(
                model=model,
                loss_fn=loss_fn,
                accuray_fn=accuray_fn,
                optimizer=optimizer,
                data_batch=images,
                target_batch=labels,
                device=device,
                step_type="val",
            )
            total_loss += loss_value
            total_acc += acc_value

    test_loss = total_loss / len(loader)
    test_acc = total_acc / len(loader)

    # print(f"Loss: {test_loss:.6f}, Acc: {total_acc/len(loader):.6f}")
    return test_loss, test_acc.to("cpu").item()


def training_loop(
    model,
    gradcam,
    loss_fn,
    accuray_fn,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    device,
    num_epochs,
    models_direcotry,
    model_name,
    tqdm_cols=None,
    writer=writer,
    SEED=42,
    CAM_tracking=True,
):
    print(
        """
**************************************************************************************************
**************************************************************************************************
---------------------------------   Training Started - -------------------------------------------
Traing model: {model_name}
Number of Epochs: {num_epochs}
Batch size: {train_loader.batch_size}
Device: {device}
**************************************************************************************************
**************************************************************************************************
"""
    )
    #* creating test tensor and target for gradcam
    test_tensor = test_loader.dataset[SEED][0].to(device).unsqueeze(0)
    test_target = test_loader.dataset[SEED][1].to(device)
    #* performing gradcam on initial weights 
    if CAM_tracking:
        print("Ploting the first gradcam on the model's weights before training")
        if not os.path.exists(f"figs/gradcam/frames/{model_name}"):
            os.mkdir(f"figs/gradcam/frames/{model_name}")
        gradcam.save_grad_cam(
            test_tensor, test_target.item(), 0, f"figs/gradcam/frames/{model_name}"
        )
        
    #* History dict
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": []}
    
    #* Main loop
    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_acc = 0, 0
        epoch_val_loss, epoch_val_acc = 0, 0
        
        #* Train Loop
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
            if scheduler is not None:
                scheduler.step()
            
            
        #* Validation Loop
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

        #* Test Loop 
        test_loss, test_acc = model_eval(
            model, test_loader, loss_fn, accuray_fn, optimizer, device
        )

        
        #* Validation Best Model Saving
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
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        # try :
        #     print(f"Learning rate updated to : {scheduler.get_last_lr()}")
        # except:
        #     print("No scheduler")
        

        print(
            "---------------------------------------------------------------------------------------------------------------------"
        )
        #* train epoch results
        print(
            f"Epoch {epoch+1} Train loss: {history['train_loss'][-1]:.6f} | Train acc: {(history['train_acc'][-1]*100):.4f}%"
        )
        #* val epoch results
        print(
            f"Epoch {epoch+1} Val loss: {history['val_loss'][-1]:.6f} | Val acc: {(history['val_acc'][-1]*100):.4f}%"
        )
        #* test epoch result
        print(
            f"Epoch {epoch+1} Test loss: {history['test_loss'][-1]:.6f} | Test acc: {(history['test_acc'][-1]*100):.4f}%"
        )
        
        # ******************** tensorboard********************#
        # * Tensorboard scalars
        if writer is not None:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={
                    "train_loss": history["train_loss"][-1],
                    "val_loss": history["val_loss"][-1],
                    "test_loss": history["test_loss"][-1]
                },
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={
                    "train_acc": history["train_acc"][-1],
                    "val_acc": history["val_acc"][-1],
                    "test_acc": history["test_acc"][-1]
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
        #* saving gradcam images for visualization
        if CAM_tracking:
            gradcam.save_grad_cam(
                test_tensor, test_target.item(), epoch, f"figs/gradcam/frames/{model_name}"
            )
        print(
            "---------------------------------------------------------------------------------------------------------------------"
        )
    if writer is not None:
        writer.close()
    return history

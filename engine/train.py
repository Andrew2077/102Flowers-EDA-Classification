import torch
from tqdm import tqdm

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
    data_batch, target_batch = data_batch.to(device), (target_batch-1).to(device)
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
    loss_fn,
    accuray_fn,
    optimizer,
    train_loader,
    val_loader,
    device,
    num_epochs,
    models_direcotry,
    tqdm_cols = None,
):
    for epoch in range(num_epochs):
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        epoch_train_loss, epoch_train_acc = 0, 0
        epoch_val_loss, epoch_val_acc = 0, 0
        for image_batch, label_batch in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Train Epoch : {epoch+1}",
            leave=True,
            ncols=tqdm_cols,
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

        if len(history["val_loss"]) != 0:
            print("validation loss decreased")
            if (epoch_val_loss / len(val_loader)) < min(history["val_loss"]):
                torch.save(
                    model.state_dict(),
                    models_direcotry + f"model_{epoch+1}.pth",
                )
                print(
                    f"Validation loss decreased from {min(history['val_loss'])} to {epoch_val_loss / len(val_loader)}, saving model"
                )
        

        # * Printing the results
        history["train_loss"].append(epoch_train_loss / len(train_loader))
        history["train_acc"].append(epoch_train_acc / len(train_loader))
        history["val_loss"].append(epoch_val_loss / len(val_loader))
        history["val_acc"].append(epoch_val_acc / len(val_loader))
        print(
            "-----------------------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^----------------------"
        )
        print(
            f"Epoch {epoch+1} Train loss: {history['train_loss'][-1]:.6f} | Train acc: {(history['train_acc'][-1]*100):.4f}%"
        )
        print(
            f"Epoch {epoch+1} Val loss: {history['val_loss'][-1]:.6f} | Val acc: {(history['val_acc'][-1]*100):.4f}%"
        )
        print(
            "-----------------------________________________________----------------------"
        )
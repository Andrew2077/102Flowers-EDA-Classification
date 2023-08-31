def training(
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
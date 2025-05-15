from codes.my_Unet_functions import *

def main():
    random.seed(42)
    transform = get_transforms()
    # train_dataset, val_dataset, test_dataset = prepare_datasets("/app/data/images/default", "/app/data/masks", transform)
    # train_dataset, val_dataset, test_dataset = prepare_datasets("~/snap/snapd-desktop-integration/current/dp_project/images", "~/snap/snapd-desktop-integration/current/dp_project/masks", transform)
    # train_dataset, val_dataset, test_dataset = prepare_datasets("/app/images/default", "/app/masks", transform)
    # train_dataset, val_dataset, test_dataset = prepare_datasets("images/default", "masks", transform)
    train_dataset, val_dataset, test_dataset = prepare_datasets("test", "test_mask", transform)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_dataset, val_dataset, test_dataset)
    model, criterion, optimizer, device, lr_scheduler = initialize_model()
    num_epochs = 200
    patience = 50
    min_delta = 0.001

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        new_model_name='test_scratch_model.pth',
        lr_scheduler=lr_scheduler,
        patience=patience,
        min_delta=min_delta
    )

    cm, precision, recall, f1, iou = compute_metrics(model, test_dataloader, device)
    plot_and_save_confusion_matrix(cm)
    accuracy = calculate_accuracy(model, test_dataloader, device)
    save_training_summary(model, num_epochs, train_losses, val_losses, train_accuracies, val_accuracies,
                          accuracy, precision, recall, f1, iou)
    plot_run_over_epochs(model, train_losses, "Trenovacia strata", val_losses, "Validacna strata", "loss_plot.png",
                         "Strata", "Trenovacia a validacna strata pocas trenovania")
    plot_run_over_epochs(model, train_accuracies, "Trenovacia presnost", val_accuracies, "Validacna presnost",
                         "accuracy_plot.png",
                         "Presnost", "Trenovacia a validacna presnost pocas trenovania")

if __name__ == '__main__':
    main()
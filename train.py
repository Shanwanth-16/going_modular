import os
import torch
import data_setup,engine,model_builder,utils

from torchvision import transforms

#hyprerparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

train_dir = "/Users/shanwanthdhanekula/Desktop/PyTorch_Basics/data/pizza_steak_sushi/train"
test_dir = "/Users/shanwanthdhanekula/Desktop/PyTorch_Basics/data/pizza_steak_sushi/test"

device = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    data_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor()
    ])

    train_dataloader,test_dataloader,class_names = data_setup.create_dataloaders(train_dir,test_dir,data_transform,BATCH_SIZE)

    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

    utils.save_model(model=model,
                    target_dir="models",
                    model_name="05_going_modular_script_mode_tinyvgg_model.pth")


if __name__ == "__main__":
    main()
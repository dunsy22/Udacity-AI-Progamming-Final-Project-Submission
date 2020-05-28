# Imports here
import time
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import argparse
import json


def arg_parser():
    parser = argparse.ArgumentParser(description="Parser for Udacity Neural Network Training Script")
    parser.add_argument('data_directory', help='Provide data directory. example - "flowers" ', type=str)
    parser.add_argument('--save_dir', help='Provide saving directory. Optional argument', type=str,
                        default='./checkpoint.pth')
    parser.add_argument('--arch', help='Vgg16 is default, Alternative: Vgg19 ', type=str, default='Vgg16')
    parser.add_argument('--learning_rate', help='Learning rate, default value 0.0002', type=float, default=0.0002)
    parser.add_argument('--hidden_units', help='Hidden units in Classifier. Default is 1024', type=int,
                        default=1024)
    parser.add_argument('--epochs', help='Number of epochs. Default is 3.', type=int, default=3)
    parser.add_argument('--gpu', help="Defaults to use GPU. Type 'CPU' to switch.", type=str, default='gpu')

    args = parser.parse_args()

    return args


def initialize_model(args):
    if args.arch == 'Vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)

    num_features = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(num_features, 4096),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(4096, args.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(args.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    return model


def train_model(model, args, device, train_loader, valid_loader):
    # criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    print("Device used for training: %s" % device)
    model.to(device)

    # epochs and step counters for training iterations
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move inut and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.4f}.. "
                      f"Valid loss: {valid_loss / len(valid_loader):.4f}.. "
                      f"Valid accuracy: {accuracy / len(valid_loader):.4f}")

                running_loss = 0
                model.train()

    return model


def test_model(model, test_loader, device):
    criterion = nn.NLLLoss()

    print("Device used for testing: %s" % (device))
    model.to(device)

    model.eval()

    test_loss = 0
    accuracy = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model(inputs)
        loss = criterion(logps, labels)
        test_loss += loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss / len(test_loader):.4f} "
          f"Test accuracy: {accuracy / len(test_loader):.4f}")

    running_loss = 0

    return test_loss, accuracy


def save_model(model, args, train_data):
    model.to('cpu')
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': args.arch,
                  'epochs': args.epochs,
                  'gpu': args.gpu,
                  'classifier': model.classifier,
                  'classifier_state_dict': model.classifier.state_dict(),
                  'class_to_idx': model.class_to_idx}

    # saving trained model for future use
    if args.save_dir:
        torch.save(checkpoint, args.save_dir)
    else:
        torch.save(checkpoint, './checkpoint.pth')

    return


def main():
    # Get Keyword Args from Parser
    args = arg_parser()

    # Define device as cuda or cpu
    if args.gpu == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'

    # Load label to name mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Define paths for data
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transform data to prepare for training
    print("Transforming and organizing data to prepare it for training...")
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # Load the transformed datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)

    # Using the transformed datasets, define the dataloaders with batchsize 128
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)
    print("Tranformation complete!")
    time.sleep(2)

    # Configure and initialize model based on arg inputs
    print("Initializing your Neural Network...")
    model = initialize_model(args)
    print("Initialization complete!")
    time.sleep(3)

    # Train the model for X epochs
    print("Training has begun...sit back and relax, this may take a few minutes....")
    train_model(model, args, device, train_loader, valid_loader)
    print("Training complete!")
    time.sleep(3)

    # Test the model accuracy on test_loader data
    print("Testing the model for accuracy using the test dataset...")
    test_model(model, test_loader, device)
    print("Testing complete!")
    time.sleep(3)

    # Save trained and tested model to save_dir
    save_model(model, args, train_data)
    print("Your model has been saved to %s. Time to try it out on some real images!" % (args.save_dir))


if __name__ == '__main__': main()
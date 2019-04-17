# Libraries
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, sys
import copy
import argparse
from sklearn.model_selection import train_test_split

from load_dataset_train import load_dataset, transform_dataset


# Typical command to train a network:
# python3 -m supervised_model_train --batches=8 --epochs=15 --model=squeezenet --v=10

########## Main parameters ########## 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--v', type=str, default="")
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    # Top level data directory. We assume here that the format of the directory
    # conforms to the ImageFolder structure (https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder)
    data_dir = "../comics"

    # Models to choose from [resnet18, alexnet, vgg11bn, squeezenet10, densenet, inception3]
    model_name = str(args.model)
    model_version = str(args.v)

    # Number of classes in the dataset (Cats vs Dogs)
    num_classes = 4
    """ 
    The classes are: 
        - Golden Age (1938 - 1954)
        - Silver Age (1954 - 1970)
        - Bronze Age (1970 - 1986)
        - Modern Age (1986 - now)
    """

    # Batch size for training (change depending on how much memory you have)
    batch_size = args.batches

    # Number of epochs to train for 
    num_epochs = args.epochs

    # Flag for feature extracting. 
    #   - False: we finetune the whole model
    #   - True: we only update the reshaped layer params
    feature_extract = False             # Feature extraction is not enough for what we want to do


    def train_model(model, dataloaders, criterion, optimizer, current_model=None, num_epochs=15, is_inception=False, best_acc=0.0):
        """
        Inputs: PyTorch model, dictionary of dataloaders, loss function, optimizer, specified number of epochs to train and validate for, 
        a boolean flag for when the model is an Inception model.
        This flag is used to accomodate the Inception v3 model, as that architecture uses an auxiliary output and the overall model loss
        respects both the auxiliary output and the final output. 
        The function trains for the specified number of epochs and after each epoch, runs a full validation step. It also keeps track of 
        the best performing model (in terms of validation accuracy). At the end of training, it returns the best performing model.
        After each epoch, the training and validation accuracies are printed.
        """
        since = time.time()

        val_acc_history = []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val", "test"]:
                if phase == "train":
                    model.train()   # Set model to training mode
                elif phase == "val":
                    model.eval()    # Set model to evaluate mode
                else:
                    model.eval()    # Set model to test mode
            
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    #print(inputs, labels)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training, it has an auxiliary output
                        # In train mode, we calculate the loss by summing the final output and the
                        # auxiliary output. But in testing, we only consider the final output.
                        if is_inception and phase == "train":
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    
                        _, preds = torch.max(outputs, 1)

                        # Backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

                # Deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == "val":
                    val_acc_history.append(epoch_acc)
        
            print()
            # Save a checkpoint to resume the training if the program crashes/stops
            path_checkpoint = "./model_saves/temporary/" + model_name + model_version + "_" + str(batch_size) + "_" + str(num_epochs) + ".pth"
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'time_elapsed': time.time()-since
            }, path_checkpoint)


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        
        # Save this data in a file
        model_info_file = open("./model_saves/" + model_name + model_version + "_" + str(batch_size) + "_" + str(num_epochs) + ".txt", "w")
        model_info_file.write('The model computed here is: ')
        model_info_file.write(model_name + model_version + "_" + str(batch_size) + "_" + str(num_epochs) + ".pth")
        model_info_file.write('\n')
        model_info_file.write('Training complete in {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        model_info_file.write('\n')
        model_info_file.write('Best val Acc: {:4f}'.format(best_acc))

        model_info_file.close()

        # Load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def set_parameter_requires_grad(model, feature_extracting):
        """
        Sets the '.requires_grad()' attribute of the parameters in the model to False when we are 
        feature extracting. By default, all of the parameters of a pretrained model have 
        '.requires_grad=True', which is fine if we are training from scratch or finetuning.
        """
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        """
        Initialize these variables, which will be set in this if statement. Each of these
        variables is model specific
        """
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            # Resnet18
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
    
        elif model_name == "alexnet":
            # Alexnet
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224
    
        elif model_name == "vgg":
            # VGG11_bn
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224
    
        elif model_name == "squeezenet":
            # Squeezenet
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            # Densenet
            model_ft = models.densenet121(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """
            Inception v3
            Be careful, expects (299, 299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxiliary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299
    
        else:
            print("Invalid model name, exiting...")
            exit()
    
        return model_ft, input_size

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    """
    Now that we know what the input size must be, we can initialize the data transforms, image datasets,
    and the dataloaders. Notice, the models were pretrained with the hard-coded normalization values.
    """
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("Initializing Datasets and Dataloaders...")

    labels_to_idx = {
        "Golden Age" : 0,
        "Silver Age" : 1,
        "Bronze Age" : 2,
        "Modern Age" : 3
    }

    # Create training, validation and test datasets
    image_datasets = {}
    dataset_full = load_dataset(data_dir, data_transforms["train"], labels_to_idx)

    # Split in train, val and test from the image list
    np.random.seed(42)
    image_datasets["train"], image_datasets["test"] = train_test_split(dataset_full)
    image_datasets["train"], image_datasets["val"] = train_test_split(image_datasets["train"])

    # Transform the datasets
    # image_datasets["train"] = transform_dataset(image_datasets["train"], data_transforms["train"])
    # image_datasets["val"] = transform_dataset(image_datasets["val"], data_transforms["val"])
    # image_datasets["test"] = transform_dataset(image_datasets["test"], data_transforms["test"])

    # Create training, validation and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ["train", "val", "test"]}
    print("HERE 1: ", dataloaders_dict["train"])

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    """
    Gather the parameters to be optimized/updated in this run (Note: all parameters that have 
    '.requires_grad=True' should be optimized). If we are finetuning, we will be updating all
    parameters. However, if we are doing feature extract method, we will only update the parameters
    that we have just initialized, i.e. the parameters with requires_grad is True.
    """
    params_to_update = model_ft.parameters()
    print("Params to learn: ")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    #Setup the loss fxn
    criterion = nn.CrossEntropyLoss()


    if(args.resume):
        # Let's resume computing
        checkpoint = torch.load("./model_saves/temporary/" + model_name + model_version + "_" + str(batch_size) + "_" + str(num_epochs) + ".pth")
        model_ft.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]

        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ["train", "val", "test"]}

        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, current_model=model_ft, num_epochs=num_epochs-epoch-1, is_inception=False, best_acc=checkpoint["best_acc"])


    else:   
        # Train and evaluate
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ["train", "val", "test"]}
        print("HERE 2: ", dataloaders_dict)
        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    # Save the model in the 'models' directory, in the 'model_saves' directory
    saving_model_name = "./model_saves/" + model_name + model_version + "_" + str(batch_size) + "_" + str(num_epochs) + ".pth"
    print("Saving path: ", saving_model_name)

    torch.save(model_ft.state_dict(), saving_model_name)
    print("Model (state_dict) saved!")
    # We remove the temporary file computed
os.remove('./model_saves/temporary/' + model_name + model_version + "_" + str(batch_size) + "_" + str(num_epochs) + ".pth")





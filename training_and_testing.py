import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from vision.torchvision.models.detection import fasterrcnn_resnet50_fpn
from vision.torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def train_models(dataloaders, epochs_list, learning_rates, num_classes, save_path=None, freeze_params=True):
    """
Parameters:
        dataloaders (Dict): Dictionary of training, validation, and testing Dataloaders
        epochs_list (list): List of epochs to train models
        learning_rates (list): List of learning rates to train models
        save_path (str): Path Directory to save models
        freeze_params(bool): Freeze parameters of pre-trained model for transfer learning

This function will train and save multiple models to the stated directory (current directory if None), and will print the best model that performs on the testing dataset.
    """
    print('You will train {} models!'.format(len(epochs_list) * len(learning_rates)))

    filenames = []
    test_losses = []
          
    for epochs in epochs_list:
        # Getting an array to plot the x-axis of plot
        x = np.arange(1, epochs+1)
        all_losses = []
        for lr in learning_rates:
            filename, val_losses, test_loss = train_and_save(epochs, lr, dataloaders, num_classes, save_path, freeze_params)

            #Appending Values
            filenames.append(filename)
            test_losses.append(test_loss)
            all_losses.append(val_losses)
              
        #Plotting val_losses
        plot_loss(all_losses, epochs, learning_rates)
    
    # Finding model with lowest error
    test_losses = np.array(test_losses)
    best_idx = np.argmin(test_losses)
    
    print('\n\nThe Best Model is {} with the lowest test loss of {}!!!'.format(filenames[best_idx], test_losses[best_idx]))

def train_and_save(epochs, lr, dataloaders, num_classes, save_path, freeze_params=True):
    """
    Parameters:
        epochs (int): Number of Epochs
        lr (float): Learning Rate
        dataloaders (Dict): Dictionary of training, validation, and testing Dataloaders
        save_path (str): Path Directory to save models
        freeze_params(bool): Freeze parameters of pre-trained model for transfer learning
    This function trains and saves a single pretrained Faster R-CNN model.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    #Freezing Backbone Parameters if True
    if freeze_params == True:
        for param in model.backbone.parameters():
            param.requires_grad = False

    #Tuning the model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)

    # Moving model to CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Train on more than 1 GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.to(device)

    # Setting Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    
    # Training Model
    print('Training model with {} epochs and {} learning rate'.format(epochs, lr))
    model, val_losses = train_model(model, optimizer, dataloaders, epochs, device)
    
    # Testing Model
    test_loss = test_model(model, dataloaders, device)
          
    # Saving Model
    filename = 'model_e({})_lr({}).pth'.format(epochs, lr)
    
    if save_path is None:
        path = os.getcwd()
          
    else:
        path = save_path
    
    torch.save(model.state_dict(), os.path.join(path, filename))
          
    print('\nSaved model as {} in {}\n\n\n'.format(filename, path))
          
    return filename, val_losses, test_loss
          
def train_model(model, optimizer, dataloaders, epochs, device):
          
    #Getting Dataloaders
    train_dataloader = dataloaders['training']
    val_dataloader = dataloaders['validation']
          
    val_losses = np.zeros(epochs)
    
    min_val_loss = np.inf
    
    best_model_state = model.state_dict()
    
    for e in range(epochs):

        model.train()
        
        # Training the model
        train_loss = 0.0
        for images, targets in train_dataloader:
            images = [image.to(device) for image in images]
            targets = [{'boxes':target['boxes'].to(device), 'labels':target['labels'].to(device)} for target in targets]
            

            #Resetting Gradients
            optimizer.zero_grad()

            #Getting all losses
            loss_dict = model(images, targets)

            #Summing all loss
            loss = sum(loss for loss in loss_dict.values())

            #Calculating Gradients
            loss.backward()

            #Take an Optimizer Step
            optimizer.step()

            train_loss += loss.item()

        #Taking mean of the loss
        train_loss /= len(train_dataloader)
          
        # Validating the model
        model.eval()
        val_loss = 0.0
        for images, targets in val_dataloader:
            images = [image.to(device) for image in images]
            targets = [{'boxes':target['boxes'].to(device), 'labels':target['labels'].to(device)} for target in targets]
          
            #Getting outputs
            outputs = model(images, targets)

            #Getting the loss
            loss_dict = outputs['losses']

            #Summing all loss
            loss = sum(loss for loss in loss_dict.values())

            # Adding Loss
            val_loss += loss.item()

        val_loss /= len(val_dataloader)
          

        val_losses[e] = val_loss
          
        print('Epoch: {}\tTraining Loss: {}\tValidation Loss: {}'.format(e+1, train_loss, val_loss))
        
        if val_loss < min_val_loss:
            print('Saving new model parameters for better validation performance')
            best_model_state = model.state_dict()
            min_val_loss = val_loss
    
    # Loading best model paraments
    model.load_state_dict(best_model_state)
    
    print('Training DONE')
    return model, val_losses
          
          
def test_model(model, dataloaders, device):
    
    print('\nTesting Model')
    
    model.eval()
    test_dataloader = dataloaders['testing']

    test_loss = 0.0
    for images, targets in test_dataloader:
        images = [image.to(device) for image in images]
        targets = [{'boxes':target['boxes'].to(device), 'labels':target['labels'].to(device)} for target in targets]
        
        
        #Getting outputs
        outputs = model(images, targets)
    
        #Getting the loss
        loss_dict = outputs['losses']
        
        #Summing all loss
        loss = sum(loss for loss in loss_dict.values())
        
        # Adding Loss
        test_loss += loss.item()
    
    test_loss /= len(test_dataloader)

    print('Testing Loss: {}'.format(test_loss))
    return test_loss

def plot_loss(all_losses, epochs, learning_rates):
    x = np.arange(1, (epochs+1))
    
    for i in range(len(all_losses)):
        val_losses = all_losses[i]
        plt.plot(x, val_losses, label='Learning Rate {}'.format(learning_rates[i]))
        
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Training with {} Epochs!'.format(epochs))
    plt.show()
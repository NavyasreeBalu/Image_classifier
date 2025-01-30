import torch
import argparse
from model_utils import load_data, create_model, train_model, save_checkpoint
import torch.optim as optim
import torch.nn as nn

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train an image classifier')
    parser.add_argument('data_dir', type=str, help='Path to the dataset')
    parser.add_argument('--arch', type=str, default='resnet34', help='Model architecture (resnet34, alexnet)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load the data
    dataloaders = load_data(args.data_dir)
    
    # Check if GPU is available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_model(arch=args.arch, hidden_units=args.hidden_units)
    
    # Move model to device
    model.to(device)
    
    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # Train the model
    train_model(model, dataloaders, criterion, optimizer, device, epochs=args.epochs)
    
    # Save the checkpoint
    save_checkpoint(model, arch=args.arch, hidden_units=args.hidden_units, learning_rate=args.learning_rate,
                    epochs=args.epochs, use_gpu=args.gpu)

if __name__ == "__main__":
    main()

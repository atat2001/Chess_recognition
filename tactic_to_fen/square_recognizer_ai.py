import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
from util import generate_image_hash, image_to_squares
import os
# Define the CNN model
class ChessPieceCNN(nn.Module):
    def __init__(self):
        super(ChessPieceCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the tensor after the pooling layer
        self._to_linear = 64 * 32 * 32  # 64 channels, 32x32 feature map size after pooling
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 13)  # 13 classes: black and white for each piece type + empty square
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, prints = False):
        if prints:
            print(f'Input: {x.size()}')  # Print input shape
        x = torch.relu(self.conv1(x))
        if prints:
            print(f'After conv1: {x.size()}')  # Print shape after conv1
        x = self.pool(torch.relu(self.conv2(x)))
        if prints:
            print(f'After conv2 and pool: {x.size()}')  # Print shape after conv2 and pool
        x = x.view(x.size(0), -1)
        
        if prints:
            print(f'After view: {x.size()}')  # Print shape after view
        x = torch.relu(self.fc1(x))
        
        if prints:
            print(f'After fc1: {x.size()}')  # Print shape after fc1
        x = self.dropout(x)
        x = self.fc2(x)
        
        if prints:
            print(f'Output: {x.size()}')  # Print output shape
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    print("starting training")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
        if epoch_loss < 0.01:
            break
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 1
        with torch.no_grad():
            if val_loader is not None:
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        val_loss /= 1 + len(val_loader.dataset)
        val_accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    

    # Save the trained model
    print('Finished Training')
    current_dir = os.path.dirname(os.path.abspath(__file__))

    torch.save(model.state_dict(), os.path.join(current_dir,'chess_piece_cnn.pth'))

def predict_image(image, model, transform):
    model.eval()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def get_model():
    model = ChessPieceCNN().to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model.load_state_dict(torch.load(os.path.join(current_dir,'chess_piece_cnn.pth')))
    return model

def image_prediction_to_fen_representation(image, model):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # ['b_b', 'b_k', 'b_n', 'b_p', 'b_q', 'b_r', 'empty', 'w_b', 'w_k', 'w_n', 'w_p', 'w_q', 'w_r']
    classifications = {0:'b', 1:'k', 2:'n', 3:'p', 4:'q', 5:'r', 6:'_', 7:'B', 8:'K', 9:'N', 10:'P', 11:'Q', 12:'R'}
    return classifications[predict_image(image, model, transform)]

def train_model_default():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    model = ChessPieceCNN().to(device)
    # Load the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = datasets.ImageFolder(root=os.path.join(current_dir,'data'), transform=transform)
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)


def fine_tune(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    model = get_model()
    # Load the dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def __main__():
    # Set device
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    training_mode = True 
    model = ChessPieceCNN().to(device)

    if training_mode:
        train_model_default()
        
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(current_dir,'woodpecker')
        prediction_folder = os.path.join(current_dir,'data_try')
        prediction_folders = os.listdir(prediction_folder)
        prediction_folders.sort()
        print(prediction_folders)
        exit()
        files = os.listdir(img_path)

        initial_index = 100
        last_index = len(files) - 1

        for index in range(initial_index, last_index+1):
            image = Image.open(str(img_path + "/"+files[index]))
            squares = image_to_squares(image, False)
            # Load the trained model
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model.load_state_dict(torch.load(os.path.join(current_dir,'chess_piece_cnn.pth')))
            for square in squares:
                # Predict the class of the image
                predicted_class = predict_image(square, model, transform)
                #print(f'Predicted class: {predicted_class}')
                square.save(prediction_folder + "/" +prediction_folders[predicted_class] + f'/{generate_image_hash(square)}.jpg')

if __name__ == "__main__":
    __main__()
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from gesture_model import GestureClassifier

def train_model(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    dataset_dir = config["dataset_dir"]
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    classes = config["classes"]
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureClassifier(num_classes=len(classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct.double() / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config["gesture_model_path"])
            print("Saved Best Model!")

    # Export to ONNX optionally
    print("Exporting best model to ONNX...")
    model.load_state_dict(torch.load(config["gesture_model_path"]))
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    onnx_path = config["gesture_model_path"].replace(".pth", ".onnx")
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'])
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    train_model()

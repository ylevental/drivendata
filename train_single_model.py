#!/usr/bin/env python3
"""
Train a single model for ensemble
Usage: python3 train_single_model.py <model_name>
Example: CUDA_VISIBLE_DEVICES=0 python3 train_single_model.py efficientnet_b3
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler

# Config
DATA_DIR = os.getcwd() if os.path.exists('train_labels.csv') else os.path.dirname(os.path.abspath(__file__))
TRAIN_LABELS = os.path.join(DATA_DIR, 'train_labels.csv')
TRAIN_DIR = os.path.join(DATA_DIR, 'train_features')
NUM_EPOCHS = 40
BATCH_SIZE = 64  # Reduced from 96 to prevent OOM
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
EARLY_STOP_PATIENCE = 8
LR_PATIENCE = 4

class WildlifeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.class_cols = [col for col in df.columns if col != 'id']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor([row[col] for col in self.class_cols], dtype=torch.float32)
        return image, labels

def get_transforms(mode='train', img_size=300):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # val
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def create_model(model_name, num_classes):
    if model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights='DEFAULT')
        img_size = 300
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights='DEFAULT')
        img_size = 380
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, 224
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model, img_size

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    
    return running_loss / len(loader)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 train_single_model.py <model_name>")
        print("Models: efficientnet_b3, efficientnet_b4, resnet50")
        sys.exit(1)
    
    model_name = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    # Load data
    train_df = pd.read_csv(TRAIN_LABELS)
    class_cols = [col for col in train_df.columns if col != 'id']
    num_classes = len(class_cols)
    
    # Create model
    model, img_size = create_model(model_name, num_classes)
    model = model.to(device)
    print(f"Image size: {img_size}x{img_size}")
    
    # Split data
    train_data, val_data = train_test_split(
        train_df, test_size=VAL_SPLIT, random_state=42, shuffle=True
    )
    
    # Create datasets
    train_dataset = WildlifeDataset(train_data, TRAIN_DIR, get_transforms('train', img_size))
    val_dataset = WildlifeDataset(val_data, TRAIN_DIR, get_transforms('val', img_size))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE
    )
    scaler = GradScaler('cuda')
    
    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f'best_model_{model_name}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"✓ Model saved! (Best Val Loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
        
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 60)
    print(f"{model_name} COMPLETE!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    main()

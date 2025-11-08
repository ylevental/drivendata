#!/usr/bin/env python3
"""
Generate ensemble predictions from trained models
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# Config
DATA_DIR = os.getcwd() if os.path.exists('train_labels.csv') else os.path.dirname(os.path.abspath(__file__))
TRAIN_LABELS = os.path.join(DATA_DIR, 'train_labels.csv')
SUBMISSION_FORMAT = os.path.join(DATA_DIR, 'submission_format.csv')
TEST_DIR = os.path.join(DATA_DIR, 'test_features')
BATCH_SIZE = 64  # Match training batch size
TTA_AUGMENTATIONS = 3
MODELS = ['efficientnet_b3', 'efficientnet_b4', 'resnet50']

class WildlifeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_id

def get_transforms(mode='val', img_size=300):
    if mode == 'tta':
        return transforms.Compose([
            transforms.Resize((img_size + 20, img_size + 20)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # val/test
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
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model, img_size

def predict_single_model_tta(model_name, submission_df, class_cols, num_classes, device):
    print(f"\nGenerating predictions for {model_name}...")
    
    # Load model
    model, img_size = create_model(model_name, num_classes)
    model = model.to(device)
    model_path = f'best_model_{model_name}.pth'
    
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found! Train this model first.")
        return None, None
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_predictions = []
    ids = []
    
    # Run TTA
    for tta_idx in range(TTA_AUGMENTATIONS):
        transform = get_transforms('tta' if tta_idx > 0 else 'val', img_size)
        test_dataset = WildlifeDataset(submission_df, TEST_DIR, transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=4)
        
        predictions = []
        
        with torch.no_grad():
            for images, img_ids in tqdm(test_loader, desc=f"{model_name} TTA {tta_idx+1}/{TTA_AUGMENTATIONS}"):
                images = images.to(device)
                outputs = torch.sigmoid(model(images))
                predictions.append(outputs.cpu().numpy())
                if tta_idx == 0:
                    ids.extend(img_ids)
        
        all_predictions.append(np.vstack(predictions))
    
    # Average TTA predictions
    return np.mean(all_predictions, axis=0), ids

def main():
    print("=" * 60)
    print("GENERATING ENSEMBLE PREDICTIONS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    submission_df = pd.read_csv(SUBMISSION_FORMAT)
    train_df = pd.read_csv(TRAIN_LABELS)
    class_cols = [col for col in train_df.columns if col != 'id']
    num_classes = len(class_cols)
    
    print(f"Test samples: {len(submission_df)}")
    print(f"Models in ensemble: {MODELS}")
    
    # Get predictions from each model
    all_model_predictions = []
    ids = None
    
    for model_name in MODELS:
        preds, img_ids = predict_single_model_tta(model_name, submission_df, 
                                                   class_cols, num_classes, device)
        if preds is None:
            print(f"\nERROR: Could not generate predictions for {model_name}")
            print("Make sure all models are trained first!")
            return
        
        all_model_predictions.append(preds)
        if ids is None:
            ids = img_ids
    
    # Average ensemble predictions
    print("\nAveraging ensemble predictions...")
    final_predictions = np.mean(all_model_predictions, axis=0)
    
    # Create submission
    submission = pd.DataFrame(final_predictions, columns=class_cols)
    submission.insert(0, 'id', ids)
    submission.to_csv('submission_ensemble.csv', index=False)
    
    print("\n✓ Ensemble predictions saved to submission_ensemble.csv")
    print(f"Submission shape: {submission.shape}")
    print("\nFirst few predictions:")
    print(submission.head())
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()

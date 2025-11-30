#!/usr/bin/env python3
"""
Wildlife Classification - Improved Version
Target: Get below 1.3 log loss

Key improvements:
1. Fixed dtype conversion error
2. Use CrossEntropyLoss (single-label classification)
3. Added label smoothing
4. Added Mixup augmentation
5. Added Test-Time Augmentation (TTA)
6. Better learning rate scheduling
7. Gradient accumulation for larger effective batch size
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = False

class Config:
    # Paths
    DATA_DIR = '.'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train_features')
    TEST_DIR = os.path.join(DATA_DIR, 'test_features')
    
    # Classes
    CLASSES = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']
    NUM_CLASSES = 8
    
    # Models - using strong pretrained models
    MODELS = [
        'tf_efficientnetv2_m.in21k_ft_in1k',
        'convnext_base.fb_in22k_ft_in1k',
        'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
    ]
    
    # Training
    N_FOLDS = 5
    EPOCHS = 25
    BATCH_SIZE = 24  # Reduced for 3x P40 GPUs with larger images
    ACCUMULATION_STEPS = 2  # Effective batch size = 24 * 2 = 48
    IMG_SIZE = 384
    NUM_WORKERS = 8
    DEVICE = 'cuda'
    SEED = 42
    
    # Optimizer settings
    LR = 2e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4
    
    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5
    
    # TTA
    TTA_TRANSFORMS = 4  # Number of TTA augmentations


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Generate random bounding box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


class WildlifeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_filename = row['id'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_filename)
        
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        if self.is_test:
            return img, row['id']
        else:
            # FIX: Explicitly convert to float32 to avoid object dtype error
            label_values = row[Config.CLASSES].values.astype(np.float32)
            # For single-label classification, get the class index
            label_idx = np.argmax(label_values)
            return img, label_idx


def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(size=(Config.IMG_SIZE, Config.IMG_SIZE), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=7),
            A.MotionBlur(blur_limit=7),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1),
            A.GridDistortion(distort_limit=0.1),
        ], p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                        min_holes=1, min_height=8, min_width=8, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_transforms():
    return A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms():
    """Test-Time Augmentation transforms"""
    transforms_list = [
        # Original
        A.Compose([
            A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Horizontal flip
        A.Compose([
            A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Vertical flip
        A.Compose([
            A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Both flips
        A.Compose([
            A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    return transforms_list


class WildlifeModel(nn.Module):
    def __init__(self, model_name, num_classes=8, pretrained=True, drop_rate=0.3):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.model.num_features, num_classes)
        
    def forward(self, x):
        features = self.model(x)
        features = self.dropout(features)
        return self.fc(features)


# Change the definition to include 'scheduler'
def train_epoch(model, loader, criterion, optimizer, device, scaler, scheduler, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc='Training')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply Mixup or CutMix with probability
        use_mixup = np.random.random() < Config.MIXUP_PROB
        if use_mixup:
            if np.random.random() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
            else:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, Config.CUTMIX_ALPHA)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            
            # --- CRITICAL FIX: Ensure loss is defined in BOTH cases ---
            if use_mixup:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            # -----------------------------------------------------------
            
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step() # Update LR every step
        
        running_loss += loss.item() * accumulation_steps
        
        # Optional: showing LR in the progress bar helps verify it's working
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'lr': f'{current_lr:.2e}'})
    
    return running_loss / len(loader)

def valid_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Convert to probabilities
            preds = F.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            
            # Convert labels to one-hot for log_loss calculation
            labels_onehot = F.one_hot(labels, num_classes=Config.NUM_CLASSES).cpu().numpy()
            all_labels.append(labels_onehot)
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    avg_loss = running_loss / len(loader)
    # Calculate log loss properly
    logloss = log_loss(all_labels, all_preds)
    
    return avg_loss, logloss


def train_model(model_name, train_df, fold):
    # Skip if checkpoint already exists
    model_path = f'{model_name.replace("/", "_")}_fold{fold}_best.pth'
    if os.path.exists(model_path):
        print(f"Skipping {model_name} fold {fold+1} - checkpoint exists")
        return 0.0
    print(f"\n{'='*60}")
    print(f"Training {model_name} - Fold {fold+1}/{Config.N_FOLDS}")
    print(f"{'='*60}\n")
    
    # Split data
    train_idx = train_df[train_df['fold'] != fold].index
    valid_idx = train_df[train_df['fold'] == fold].index
    
    train_data = train_df.loc[train_idx].reset_index(drop=True)
    valid_data = train_df.loc[valid_idx].reset_index(drop=True)
    
    print(f"Train samples: {len(train_data)}, Valid samples: {len(valid_data)}")
    
    # Datasets
    train_dataset = WildlifeDataset(train_data, Config.TRAIN_DIR, get_train_transforms())
    valid_dataset = WildlifeDataset(valid_data, Config.TRAIN_DIR, get_valid_transforms())
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Model
    model = WildlifeModel(model_name, num_classes=Config.NUM_CLASSES)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(Config.DEVICE)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Cosine annealing with warmup
    num_training_steps = len(train_loader) * Config.EPOCHS // Config.ACCUMULATION_STEPS
    num_warmup_steps = num_training_steps // 10
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(Config.MIN_LR / Config.LR, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()
    
    best_logloss = float('inf')
    patience = 7
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE, scaler, scheduler, Config.ACCUMULATION_STEPS)
        valid_loss, valid_logloss = valid_epoch(model, valid_loader, criterion, Config.DEVICE)
        
        #scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid LogLoss: {valid_logloss:.4f}")
        
        if valid_logloss < best_logloss:
            best_logloss = valid_logloss
            patience_counter = 0
            model_path = f'{model_name.replace("/", "_")}_fold{fold}_best.pth'
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            print(f"Model saved! Best LogLoss: {best_logloss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_logloss


def predict_with_tta(model, test_df, device, tta_transforms):
    """Predict with Test-Time Augmentation"""
    model.eval()
    
    all_preds = []
    all_ids = []
    
    for tta_idx, transform in enumerate(tta_transforms):
        print(f"TTA {tta_idx + 1}/{len(tta_transforms)}")
        
        test_dataset = WildlifeDataset(test_df, Config.TEST_DIR, transform, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        
        tta_preds = []
        ids = []
        
        with torch.no_grad():
            for images, img_ids in tqdm(test_loader, desc=f'TTA {tta_idx + 1}'):
                images = images.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    preds = F.softmax(outputs, dim=1).cpu().numpy()
                
                tta_preds.append(preds)
                if tta_idx == 0:
                    ids.extend(img_ids)
        
        tta_preds = np.vstack(tta_preds)
        all_preds.append(tta_preds)
        
        if tta_idx == 0:
            all_ids = ids
    
    # Average TTA predictions
    final_preds = np.mean(all_preds, axis=0)
    return all_ids, final_preds


def main():
    print("="*60)
    print("Wildlife Classification - Improved Pipeline")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train_labels.csv')
    test_df = pd.read_csv('test_features.csv')
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Class distribution
    print(f"\nClass distribution:")
    for cls in Config.CLASSES:
        count = train_df[cls].sum()
        print(f"  {cls}: {int(count)} ({100*count/len(train_df):.1f}%)")
    
    # Create class labels for stratification
    train_df['class_label'] = train_df[Config.CLASSES].values.argmax(axis=1)
    
    # Stratified folds
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    train_df['fold'] = -1
    for fold, (_, valid_idx) in enumerate(skf.split(train_df, train_df['class_label'])):
        train_df.loc[valid_idx, 'fold'] = fold
    
    # Train models
    all_model_scores = {}
    
    for model_name in Config.MODELS:
        fold_scores = []
        
        for fold in range(Config.N_FOLDS):
            score = train_model(model_name, train_df, fold)
            fold_scores.append(score)
        
        avg_score = np.mean(fold_scores)
        all_model_scores[model_name] = avg_score
        print(f"\n{model_name} Average CV LogLoss: {avg_score:.4f}")
    
    # Generate predictions with TTA
    print("\n" + "="*60)
    print("Generating Test Predictions with TTA")
    print("="*60)
    
    tta_transforms = get_tta_transforms()
    ensemble_preds = []
    
    for model_name in Config.MODELS:
        model_preds = []
        
        for fold in range(Config.N_FOLDS):
            model = WildlifeModel(model_name, num_classes=Config.NUM_CLASSES)
            
            checkpoint = f'{model_name.replace("/", "_")}_fold{fold}_best.pth'
            if os.path.exists(checkpoint):
                print(f"\nLoading {checkpoint}")
                model.load_state_dict(torch.load(checkpoint))
                
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(Config.DEVICE)
                
                ids, preds = predict_with_tta(model, test_df, Config.DEVICE, tta_transforms)
                model_preds.append(preds)
        
        if model_preds:
            ensemble_preds.append(np.mean(model_preds, axis=0))
    
    # Final ensemble
    final_preds = np.mean(ensemble_preds, axis=0)
    
    # Create submission
    submission = pd.DataFrame({'id': ids})
    for i, cls in enumerate(Config.CLASSES):
        submission[cls] = final_preds[:, i]
    
    submission.to_csv('submission.csv', index=False)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Submission saved to: submission.csv")
    print(f"Average CV score: {np.mean(list(all_model_scores.values())):.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

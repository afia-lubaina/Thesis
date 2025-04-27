#!/usr/bin/env python3
# filepath: /home/hpc12/Lubaina/FocusMAE/scripts/cv_with_holdout.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

# Path to your video directory
video_dir = "/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV/videos"
base_path = "/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV"
output_dir = os.path.join(base_path, "cv_with_holdout")
folds_dir = os.path.join(output_dir, "folds")
holdout_dir = os.path.join(output_dir, "holdout")

# Create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(folds_dir, exist_ok=True)
os.makedirs(holdout_dir, exist_ok=True)

# Get all video files
print("Reading video files from:", video_dir)
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

if len(video_files) == 0:
    print("ERROR: No video files found! Check the directory and file extensions.")
    exit(1)

# Create DataFrame with video paths and labels
data = []
for video in video_files:
    path = os.path.join(video_dir, video)
    # Determine label (0 for benign, 1 for malignant)
    label = 1 if video.startswith('malignant') else 0
    
    # Extract patient ID from filename
    if '_' in video:
        parts = video.split('_')
        patient_id = parts[1] if len(parts) > 1 else parts[0]
        patient_id = patient_id.split('.')[0]
    else:
        patient_id = video.split('.')[0]
    
    data.append({'path': path, 'label': label, 'patient_id': patient_id})

df = pd.DataFrame(data)

# Count total videos by class
benign_df = df[df['label'] == 0]
malignant_df = df[df['label'] == 1]
total_benign = len(benign_df)
total_malignant = len(malignant_df)
total_videos = len(df)

print(f"Total videos found: {total_videos}")
print(f"Total benign videos: {total_benign}")
print(f"Total malignant videos: {total_malignant}")

# Create a dataframe of unique patients and their labels
unique_patients = df.drop_duplicates('patient_id')[['patient_id', 'label']]
print(f"Total unique patients: {len(unique_patients)}")
print(f"Benign patients: {len(unique_patients[unique_patients['label'] == 0])}")
print(f"Malignant patients: {len(unique_patients[unique_patients['label'] == 1])}")

# STEP 1: First split patients into 80% for CV and 20% for holdout test
# This ensures no patient overlap between CV and holdout test
cv_patients, holdout_patients = train_test_split(
    unique_patients, 
    test_size=0.2, 
    random_state=42,
    stratify=unique_patients['label']
)

# Get videos for holdout patients
holdout_df = df[df['patient_id'].isin(holdout_patients['patient_id'])].copy()
cv_df = df[df['patient_id'].isin(cv_patients['patient_id'])].copy()

print(f"\n--- Initial 80/20 Split ---")
print(f"Cross-validation set: {len(cv_df)} videos ({len(cv_df)/total_videos:.1%})")
print(f"  - Benign: {len(cv_df[cv_df['label'] == 0])} ({len(cv_df[cv_df['label'] == 0])/len(cv_df):.1%})")
print(f"  - Malignant: {len(cv_df[cv_df['label'] == 1])} ({len(cv_df[cv_df['label'] == 1])/len(cv_df):.1%})")
print(f"  - Unique patients: {cv_df['patient_id'].nunique()}")

print(f"Holdout test set: {len(holdout_df)} videos ({len(holdout_df)/total_videos:.1%})")
print(f"  - Benign: {len(holdout_df[holdout_df['label'] == 0])} ({len(holdout_df[holdout_df['label'] == 0])/len(holdout_df):.1%})")
print(f"  - Malignant: {len(holdout_df[holdout_df['label'] == 1])} ({len(holdout_df[holdout_df['label'] == 1])/len(holdout_df):.1%})")
print(f"  - Unique patients: {holdout_df['patient_id'].nunique()}")

# Save the holdout set
holdout_df[['path', 'label']].to_csv(os.path.join(holdout_dir, 'test.csv'), index=False, header=False)
print(f"Holdout test set saved to: {os.path.join(holdout_dir, 'test.csv')}")

# STEP 2: Split the CV portion into 5 folds
# First, create patient groups for the 5-fold cross-validation
cv_patients_unique = cv_patients.copy()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(skf.split(cv_patients_unique, cv_patients_unique['label']))

# Assign each patient to a fold (0-4)
patient_folds = {}
for fold_idx, (_, test_idx) in enumerate(fold_indices):
    fold_patients = cv_patients_unique.iloc[test_idx]['patient_id'].values
    for patient in fold_patients:
        patient_folds[patient] = fold_idx

# Apply fold assignments to the CV dataframe
cv_df['fold'] = cv_df['patient_id'].map(patient_folds)

# Create files for each fold
for fold_idx in range(5):
    # For each fold:
    # - Current fold is for validation (20% of CV data)
    # - Other 4 folds are for training (80% of CV data)
    val_df = cv_df[cv_df['fold'] == fold_idx].copy()
    train_df = cv_df[cv_df['fold'] != fold_idx].copy()
    
    # Shuffle datasets
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print statistics
    print(f"\nFold {fold_idx+1} statistics:")
    print(f"Training set: {len(train_df)} videos ({len(train_df)/len(cv_df):.1%} of CV data)")
    print(f"  - Benign: {len(train_df[train_df['label'] == 0])} ({len(train_df[train_df['label'] == 0])/len(train_df):.1%})")
    print(f"  - Malignant: {len(train_df[train_df['label'] == 1])} ({len(train_df[train_df['label'] == 1])/len(train_df):.1%})")
    print(f"  - Unique patients: {train_df['patient_id'].nunique()}")
    
    print(f"Validation set: {len(val_df)} videos ({len(val_df)/len(cv_df):.1%} of CV data)")
    print(f"  - Benign: {len(val_df[val_df['label'] == 0])} ({len(val_df[val_df['label'] == 0])/len(val_df):.1%})")
    print(f"  - Malignant: {len(val_df[val_df['label'] == 1])} ({len(val_df[val_df['label'] == 1])/len(val_df):.1%})")
    print(f"  - Unique patients: {val_df['patient_id'].nunique()}")
    
    # Save CSV files
    fold_dir = os.path.join(folds_dir, f"fold_{fold_idx+1}")
    os.makedirs(fold_dir, exist_ok=True)
    
    train_csv_path = os.path.join(fold_dir, 'train.csv')
    val_csv_path = os.path.join(fold_dir, 'val.csv')
    test_csv_path = os.path.join(fold_dir, 'test.csv')
    
    # Save without headers
    train_df[['path', 'label']].to_csv(train_csv_path, index=False, header=False)
    val_df[['path', 'label']].to_csv(val_csv_path, index=False, header=False)
    
    # Save test file with same content as validation file
    # Note: In this setup, test and val are identical within the folds
    val_df[['path', 'label']].to_csv(test_csv_path, index=False, header=False)

# Verify no patient overlap between train and validation in each fold
for fold_idx in range(5):
    fold_dir = os.path.join(folds_dir, f"fold_{fold_idx+1}")
    train_df = pd.read_csv(os.path.join(fold_dir, 'train.csv'), header=None, names=['path', 'label'])
    val_df = pd.read_csv(os.path.join(fold_dir, 'val.csv'), header=None, names=['path', 'label'])
    
    # Extract patient IDs
    def extract_patient_id(path):
        filename = os.path.basename(path)
        if '_' in filename:
            parts = filename.split('_')
            patient_id = parts[1] if len(parts) > 1 else parts[0]
            return patient_id.split('.')[0]
        return filename.split('.')[0]
    
    train_patients = set(train_df['path'].apply(extract_patient_id))
    val_patients = set(val_df['path'].apply(extract_patient_id))
    
    overlap = train_patients.intersection(val_patients)
    
    if len(overlap) == 0:
        print(f"\nFold {fold_idx+1}: No patient overlap between train and validation ✓")
    else:
        print(f"\nFold {fold_idx+1}: WARNING! Patient overlap between train and validation: {overlap}")

# Verify holdout patients are not in any CV fold
holdout_patients_set = set(holdout_df['patient_id'].unique())
cv_patients_set = set(cv_df['patient_id'].unique())
overlap = holdout_patients_set.intersection(cv_patients_set)

if len(overlap) == 0:
    print("\nNo patient overlap between cross-validation and holdout test sets ✓")
else:
    print(f"\nWARNING! Patient overlap between CV and holdout: {overlap}")

# Save fold assignments and patient splits for reference
cv_df[['path', 'label', 'patient_id', 'fold']].to_csv(
    os.path.join(output_dir, 'cv_fold_assignments.csv'), index=False)
holdout_df[['path', 'label', 'patient_id']].to_csv(
    os.path.join(output_dir, 'holdout_assignments.csv'), index=False)

print("\nCross-validation with holdout setup created successfully in:", output_dir)
print("\nFinal Summary:")
print(f"Holdout test: {len(holdout_df)} videos ({holdout_df['patient_id'].nunique()} patients)")
for fold_idx in range(5):
    fold_dir = os.path.join(folds_dir, f"fold_{fold_idx+1}")
    train_df = pd.read_csv(os.path.join(fold_dir, 'train.csv'), header=None, names=['path', 'label'])
    val_df = pd.read_csv(os.path.join(fold_dir, 'val.csv'), header=None, names=['path', 'label'])
    
    print(f"Fold {fold_idx+1}: Train={len(train_df)} videos, Val={len(val_df)} videos")

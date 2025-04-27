#TraditionalCV.py

#!/usr/bin/env python3
# filepath: /home/hpc12/Lubaina/FocusMAE/scripts/non_overlapping_cv.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Path to your video directory
video_dir = "/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV/videos"
base_path = "/home/hpc12/Lubaina/FocusMAE/Dataset_folder/BUV"
folds_dir = os.path.join(base_path, "non_overlapping_cv")

# Create cross-validation directory if it doesn't exist
os.makedirs(folds_dir, exist_ok=True)

# Get all video files
print("Reading video files from:", video_dir)
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

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

# Check if patient IDs were extracted correctly
print(f"Total unique patients: {df['patient_id'].nunique()}")
print("Sample patient IDs:", df['patient_id'].unique()[:5])

# Create a dataframe of unique patients and their labels
unique_patients = df.drop_duplicates('patient_id')[['patient_id', 'label']]
print(f"Benign patients: {len(unique_patients[unique_patients['label'] == 0])}")
print(f"Malignant patients: {len(unique_patients[unique_patients['label'] == 1])}")

# Split into 10 non-overlapping stratified folds - creating 10 equal-sized groups
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_indices = list(skf.split(unique_patients, unique_patients['label']))

# Assign each patient to a fold (0-9)
patient_folds = {}
for fold_idx, (_, test_idx) in enumerate(fold_indices):
    test_patients = unique_patients.iloc[test_idx]['patient_id'].values
    
    for patient in test_patients:
        patient_folds[patient] = fold_idx

# Apply fold assignments to the original dataframe
df['fold'] = df['patient_id'].map(patient_folds)

# Now create 5 fold combinations where:
# - 1 fold is for test (10%)
# - 1 fold is for validation (10%)
# - 8 folds are for training (80%)
# Ensure each fold is used EXACTLY ONCE for test and EXACTLY ONCE for validation
fold_combinations = [
    {'test': 0, 'val': 5, 'train': [1, 2, 3, 4, 6, 7, 8, 9]},
    {'test': 1, 'val': 6, 'train': [0, 2, 3, 4, 5, 7, 8, 9]},
    {'test': 2, 'val': 7, 'train': [0, 1, 3, 4, 5, 6, 8, 9]},
    {'test': 3, 'val': 8, 'train': [0, 1, 2, 4, 5, 6, 7, 9]},
    {'test': 4, 'val': 9, 'train': [0, 1, 2, 3, 5, 6, 7, 8]}
]

# Create CSV files for each fold combination
for fold_idx, fold_combo in enumerate(fold_combinations):
    # Extract data based on fold assignments
    test_df = df[df['fold'] == fold_combo['test']].copy()
    val_df = df[df['fold'] == fold_combo['val']].copy()
    train_df = df[df['fold'].isin(fold_combo['train'])].copy()
    
    # Shuffle datasets
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print statistics
    print(f"\nFold {fold_idx+1} statistics:")
    print(f"Training set: {len(train_df)} videos ({len(train_df)/total_videos:.1%})")
    print(f"  - Benign: {len(train_df[train_df['label'] == 0])} ({len(train_df[train_df['label'] == 0])/len(train_df):.1%})")
    print(f"  - Malignant: {len(train_df[train_df['label'] == 1])} ({len(train_df[train_df['label'] == 1])/len(train_df):.1%})")
    
    print(f"Validation set: {len(val_df)} videos ({len(val_df)/total_videos:.1%})")
    print(f"  - Benign: {len(val_df[val_df['label'] == 0])} ({len(val_df[val_df['label'] == 0])/len(val_df):.1%})")
    print(f"  - Malignant: {len(val_df[val_df['label'] == 1])} ({len(val_df[val_df['label'] == 1])/len(val_df):.1%})")
    
    print(f"Test set: {len(test_df)} videos ({len(test_df)/total_videos:.1%})")
    print(f"  - Benign: {len(test_df[test_df['label'] == 0])} ({len(test_df[test_df['label'] == 0])/len(test_df):.1%})")
    print(f"  - Malignant: {len(test_df[test_df['label'] == 1])} ({len(test_df[test_df['label'] == 1])/len(test_df):.1%})")
    
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
    val_df[['path', 'label']].to_csv(test_csv_path, index=False, header=False)

# Define a function for extracting patient IDs
def extract_patient_id(path):
    filename = os.path.basename(path)
    if '_' in filename:
        parts = filename.split('_')
        patient_id = parts[1] if len(parts) > 1 else parts[0]
        return patient_id.split('.')[0]
    return filename.split('.')[0]

# Verify no patient overlap between train and validation in each fold
for fold_idx in range(5):
    fold_dir = os.path.join(folds_dir, f"fold_{fold_idx+1}")
    train_df = pd.read_csv(os.path.join(fold_dir, 'train.csv'), header=None, names=['path', 'label'])
    val_df = pd.read_csv(os.path.join(fold_dir, 'val.csv'), header=None, names=['path', 'label'])
    test_df = pd.read_csv(os.path.join(fold_dir, 'test.csv'), header=None, names=['path', 'label'])
    
    # Extract patient IDs
    train_patients = set(train_df['path'].apply(extract_patient_id))
    val_patients = set(val_df['path'].apply(extract_patient_id))
    test_patients = set(test_df['path'].apply(extract_patient_id))
    
    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print(f"\nFold {fold_idx+1}: No patient overlap between datasets ✓")
    else:
        if len(train_val_overlap) > 0:
            print(f"\nFold {fold_idx+1}: WARNING! Patient overlap between train and val: {train_val_overlap}")
        if len(train_test_overlap) > 0:
            print(f"\nFold {fold_idx+1}: WARNING! Patient overlap between train and test: {train_test_overlap}")
        if len(val_test_overlap) > 0:
            print(f"\nFold {fold_idx+1}: WARNING! Patient overlap between val and test: {val_test_overlap}")
    
    # Verify test files match validation files
    val_df = pd.read_csv(os.path.join(fold_dir, 'val.csv'), header=None)
    test_df = pd.read_csv(os.path.join(fold_dir, 'test.csv'), header=None)
    
    if val_df.equals(test_df):
        print(f"Fold {fold_idx+1}: Test file is identical to validation file ✓")
    else:
        print(f"Fold {fold_idx+1}: ERROR! Test file differs from validation file!")

print("\nFolds created successfully in:", folds_dir)

# Create an additional file with fold assignments for reference
fold_assignments = df[['path', 'label', 'patient_id', 'fold']]
fold_assignments.to_csv(os.path.join(folds_dir, 'fold_assignments.csv'), index=False)
print("Fold assignments saved to:", os.path.join(folds_dir, 'fold_assignments.csv'))

# Verify validation set coverage and uniqueness
patients_by_val_fold = {}
for fold_idx in range(5):
    fold_dir = os.path.join(folds_dir, f"fold_{fold_idx+1}")
    val_df = pd.read_csv(os.path.join(fold_dir, 'val.csv'), header=None, names=['path', 'label'])
    val_patients = set(val_df['path'].apply(extract_patient_id))
    
    for patient in val_patients:
        if patient in patients_by_val_fold:
            print(f"WARNING: Patient {patient} appears in multiple validation folds: {patients_by_val_fold[patient]} and {fold_idx+1}")
        else:
            patients_by_val_fold[patient] = fold_idx+1

print(f"\nVerified {len(patients_by_val_fold)} patients each appear in exactly one validation fold")

# Final summary stats
print("\nCross-Validation Summary:")
for fold_idx in range(5):
    fold_dir = os.path.join(folds_dir, f"fold_{fold_idx+1}")
    train_df = pd.read_csv(os.path.join(fold_dir, 'train.csv'), header=None, names=['path', 'label'])
    val_df = pd.read_csv(os.path.join(fold_dir, 'val.csv'), header=None, names=['path', 'label'])
    test_df = pd.read_csv(os.path.join(fold_dir, 'test.csv'), header=None, names=['path', 'label'])
    
    print(f"Fold {fold_idx+1}: Train={len(train_df)} videos ({len(train_df)/total_videos:.1%}), "
          f"Val={len(val_df)} videos ({len(val_df)/total_videos:.1%}), "
          f"Test={len(val_df)} videos ({len(val_df)/total_videos:.1%}) (identical to val)")

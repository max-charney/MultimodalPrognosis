import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time

data_folder = "Data"
pan_can_folder = os.path.join(data_folder, "PanCanAtlas")

gene_expr_file = os.path.join(pan_can_folder, "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv")
clinical_data_file = os.path.join(pan_can_folder, "TCGA-CDR-SupplementalTableS1.csv")

start_time = time.time()
gene_expr = pd.read_csv(gene_expr_file, sep='\t')
print(f'Gene expression data loaded in {time.time() - start_time:.2f} seconds.')

start_time = time.time()
clinical_data = pd.read_csv(clinical_data_file)
print(f'Clinical data loaded in {time.time() - start_time:.2f} seconds.')
# Define file paths
tiles_folder = 'Data/tiles'
cache_file = 'filtered_valid_patients_cache.json'

# Shorten gene expression column names to match TCGA-XX-XXXX format
def shorten_gene_columns(gene_expr):
    shortened_columns = {}
    for col in gene_expr.columns:
        if col.startswith('TCGA'):
            shortened_col = col[:12]
            shortened_columns[col] = shortened_col
    gene_expr = gene_expr.rename(columns=shortened_columns)
    return gene_expr

def preprocess_gene_data(gene_expr):
    # Apply log2(x + 1) transformation
    gene_expr = np.log2(gene_expr + 1)
    
    # Normalize each gene to have zero mean and unit variance
    gene_expr = (gene_expr - np.mean(gene_expr, axis=0)) / np.std(gene_expr, axis=0)
    
    return gene_expr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Dataset class with additional features and debugging
class MultiModalDataset(Dataset):
    def __init__(self, clinical_data, gene_expr, transform=None):
        self.clinical_data = clinical_data
        self.gene_expr = shorten_gene_columns(gene_expr)
        self.transform = transform

        # Ensure 'OS.time' column is present
        assert 'OS.time' in self.clinical_data.columns, "OS.time column is missing from clinical data"

        # Convert 'OS.time' to numeric, replacing non-numeric values with NaN
        self.clinical_data['OS.time'] = pd.to_numeric(self.clinical_data['OS.time'], errors='coerce')

        # Remove rows with NaN in 'OS.time'
        self.clinical_data = self.clinical_data.dropna(subset=['OS.time'])

        # Separate numeric and categorical columns
        num_cols = self.clinical_data.select_dtypes(include=['number']).columns
        cat_cols = self.clinical_data.select_dtypes(exclude=['number']).columns

        # Process numeric data
        self.clinical_data[num_cols] = self.clinical_data[num_cols].fillna(self.clinical_data[num_cols].mean())
        self.clinical_data[num_cols] = (self.clinical_data[num_cols] - self.clinical_data[num_cols].mean()) / self.clinical_data[num_cols].std()

        # Process categorical data
        self.label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            self.clinical_data[col] = self.clinical_data[col].fillna('Unknown')  # Fill NaN with 'Unknown'
            self.clinical_data[col] = le.fit_transform(self.clinical_data[col].astype(str))
            self.label_encoders[col] = le

        self.clinical_size = len(self.clinical_data.columns) - 2  # Exclude 'bcr_patient_barcode' and 'OS.time'
        self.gene_size = len(self.gene_expr.columns)

        print(f"Clinical feature size: {self.clinical_size}")
        print(f"Gene feature size: {self.gene_size}")
        print(f"Final number of samples: {len(self.clinical_data)}")

    def __len__(self):
        return len(self.clinical_data)

    def __getitem__(self, idx):
        patient_data = self.clinical_data.iloc[idx]
        patient_id = patient_data['bcr_patient_barcode']

        clinical_features = patient_data.drop(['bcr_patient_barcode', 'OS.time']).values
        gene_features = self.gene_expr[patient_id].values if patient_id in self.gene_expr.columns else np.zeros(self.gene_size)

        event_times = torch.tensor(patient_data['OS.time'], dtype=torch.float)
        censored = torch.tensor(1.0, dtype=torch.float)  # Assuming all patients are censored, adjust if you have this information

        # Load and transform image
        image_path = os.path.join('Data/tiles', f"{patient_id}_image.jpg")  # Adjust the file name pattern as needed
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        else:
            image = torch.zeros((3, 256, 256))  # Placeholder for missing images

        return {
            'clinical': torch.tensor(clinical_features, dtype=torch.float),
            'gene': torch.tensor(gene_features, dtype=torch.float),
            'image': image,
            'event_times': event_times,
            'censored': censored
        }

class FilteringDataLoader(DataLoader):
    def __iter__(self):
        return filter(lambda x: x is not None, super().__iter__())

# Define the network architectures
class ClinicalNet(nn.Module):
    def __init__(self, input_size):
        super(ClinicalNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64)
        )
    
    def forward(self, x):
        return self.fc(x)

class GeneNet(nn.Module):
    def __init__(self, input_size):
        super(GeneNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64)  # Ensure output size is consistent
        )
    
    def forward(self, x):
        return self.fc(x)

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 64 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64)  # Ensure output size is consistent
        )
    
    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

class MultiModalNet(nn.Module):
    def __init__(self, clinical_size, gene_size):
        super(MultiModalNet, self).__init__()
        self.clinical_net = ClinicalNet(clinical_size)
        self.gene_net = GeneNet(gene_size)
        self.image_net = ImageNet()
        self.fc = nn.Sequential(
            nn.Linear(64 * 3, 1)
        )
    
    def forward(self, clinical, gene, image):
        clinical_out = self.clinical_net(clinical)
        gene_out = self.gene_net(gene)
        image_out = self.image_net(image)
        
        # Ensure all features have the same size
        feature_size = 64  # This should match the output size of each individual network
        clinical_out = clinical_out[:, :feature_size]
        gene_out = gene_out[:, :feature_size]
        image_out = image_out[:, :feature_size]
        
        combined = torch.cat([clinical_out, gene_out, image_out], dim=1)
        return self.fc(combined), (clinical_out, gene_out, image_out)

class SimplifiedMultiModalNet(nn.Module):
    def __init__(self, clinical_size, gene_size):
        super(SimplifiedMultiModalNet, self).__init__()
        self.clinical_fc = nn.Linear(clinical_size, 64)
        self.gene_fc = nn.Linear(gene_size, 64)
        self.image_fc = nn.Linear(3 * 256 * 256, 64)  # Assuming 256x256 RGB images
        self.final_fc = nn.Linear(64 * 3, 1)
    
    def forward(self, clinical, gene, image):
        clinical_out = torch.relu(self.clinical_fc(clinical))
        gene_out = torch.relu(self.gene_fc(gene))
        image_out = torch.relu(self.image_fc(image.view(image.size(0), -1)))
        
        combined = torch.cat((clinical_out, gene_out, image_out), dim=1)
        output = self.final_fc(combined)
        
        # Return both the output and the individual features
        return output, (clinical_out, gene_out, image_out)

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, features):
        clinical_features, gene_features, image_features = features
        
        # Compute cosine similarity
        sim_clinical_gene = torch.nn.functional.cosine_similarity(clinical_features, gene_features, dim=1)
        sim_clinical_image = torch.nn.functional.cosine_similarity(clinical_features, image_features, dim=1)
        sim_gene_image = torch.nn.functional.cosine_similarity(gene_features, image_features, dim=1)
        
        # Compute loss
        loss_clinical_gene = torch.clamp(self.margin - sim_clinical_gene, min=0.0)
        loss_clinical_image = torch.clamp(self.margin - sim_clinical_image, min=0.0)
        loss_gene_image = torch.clamp(self.margin - sim_gene_image, min=0.0)
        
        return loss_clinical_gene.mean() + loss_clinical_image.mean() + loss_gene_image.mean()

class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, predictions, event_times, censored):
        epsilon = 1e-7
        
        # Sort in descending order
        _, indices = torch.sort(event_times, descending=True)
        predictions = predictions[indices]
        censored = censored[indices]
        event_times = event_times[indices]

        # Calculate risk scores
        risk_scores = torch.exp(predictions)
        
        # Calculate cumulative risk scores
        cumulative_risk_scores = torch.cumsum(risk_scores, dim=0)
        
        # Calculate log of cumulative risk scores
        log_risk = torch.log(cumulative_risk_scores + epsilon)
        
        # Calculate negative log likelihood
        uncensored_likelihood = predictions - log_risk
        censored_likelihood = uncensored_likelihood * censored
        neg_likelihood = -torch.sum(censored_likelihood)
        
        # Normalize by number of events
        num_events = torch.sum(censored)
        loss = neg_likelihood / (num_events + epsilon)
        
        return loss

# Evaluation function with C-index and AUC
def evaluate_model(dataloader, model, device):
    model.eval()
    all_outputs = []
    all_event_times = []
    all_censored = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            clinical = data['clinical'].to(device)
            gene = data['gene'].to(device)
            image = data['image'].to(device)
            event_times = data['event_times'].to(device)
            censored = data['censored'].to(device)
            
            if clinical is None or gene is None or image is None:
                continue
            
            outputs, _ = model(clinical, gene, image)
            outputs = outputs.squeeze()
            
            # Ensure outputs is at least 1-dimensional
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            all_outputs.append(outputs)
            all_event_times.append(event_times)
            all_censored.append(censored)
    
    # Concatenate results
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    all_event_times = torch.cat(all_event_times).cpu().numpy()
    all_censored = torch.cat(all_censored).cpu().numpy()
    
    # Compute C-index
    c_index = concordance_index(all_event_times, -all_outputs, all_censored)
    print(f"C-index: {c_index}")
    
    # Compute AUC if possible
    try:
        # Compute AUC
        auc = roc_auc_score(all_censored, -all_outputs)
        print(f"AUC: {auc}")
        return c_index, auc
    except ValueError as e:
        print(f"AUC Error: {e}")
        return c_index, None

# Training loop with debugging and evaluation
def train_and_evaluate(train_loader, test_loader, model, cox_criterion, contrastive_criterion, optimizer, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            clinical = batch['clinical'].to(device)
            gene = batch['gene'].to(device)
            image = batch['image'].to(device)
            event_times = batch['event_times'].to(device)
            censored = batch['censored'].to(device)

            optimizer.zero_grad()
            outputs, features = model(clinical, gene, image)
            
            cox_loss = cox_criterion(outputs, event_times, censored)
            contrastive_loss = contrastive_criterion(features)
            
            loss = cox_loss + contrastive_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf loss detected: cox_loss={cox_loss}, contrastive_loss={contrastive_loss}")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if torch.isnan(torch.tensor(avg_loss)):
            print("NaN loss detected. Stopping training.")
            break
        
        # Evaluate on test set
        # if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
        #     c_index, auc = evaluate_model(test_loader, model, device)
        #     print(f"Epoch {epoch + 1}/{epochs}, C-index: {c_index:.4f}, AUC: {auc:.4f}")

# Example usage
# Assume clinical_data and gene_expr are pre-loaded DataFrames
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = MultiModalDataset(clinical_data, gene_expr, transform=transform)

# Calculate the sizes for the split
total_size = len(dataset)
train_size = int(0.85 * total_size)
test_size = total_size - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Initialize and train the model
clinical_size = dataset.clinical_size  # Use the size from the dataset
gene_size = dataset.gene_size  # Use the size from the dataset

# Use this simplified model instead of the previous MultiModalNet
model = SimplifiedMultiModalNet(clinical_size=32, gene_size=dataset.gene_size)

cox_criterion = CoxLoss()
contrastive_criterion = ContrastiveLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

train_and_evaluate(train_loader, test_loader, model, cox_criterion, contrastive_criterion, optimizer, epochs=10)

torch.save(model.state_dict(), 'histo-clinical-gene-model.pth')
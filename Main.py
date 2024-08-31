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
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import random
from glob import glob
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Dataset class with additional features and debugging
class MultiModalDataset(Dataset):
    def __init__(self, clinical_data, gene_expr, cache_file, transform=None):
        self.clinical_data = clinical_data
        self.gene_expr = self.shorten_gene_columns(gene_expr)
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Load the cache file
        with open(cache_file, 'r') as f:
            self.cache = json.load(f)

        # Filter clinical data to include only valid patients
        self.clinical_data = self.clinical_data[self.clinical_data['bcr_patient_barcode'].isin(self.cache['valid_patients'])]

        # Ensure 'OS.time' column is present and process it
        assert 'OS.time' in self.clinical_data.columns, "OS.time column is missing from clinical data"
        self.clinical_data['OS.time'] = pd.to_numeric(self.clinical_data['OS.time'], errors='coerce')
        self.clinical_data = self.clinical_data.dropna(subset=['OS.time'])

        # Specify and process feature columns
        self.feature_columns = ["age_at_initial_pathologic_diagnosis", "gender", "race", "histological_grade"]
        self.process_feature_columns()

        self.clinical_size = len(self.feature_columns)
        self.gene_size = gene_expr.shape[1] if len(gene_expr.shape) > 1 else gene_expr.shape[0]

        # Process image paths
        self.patient_images = {}
        for idx, patient_id in enumerate(self.cache['valid_patients']):
            if idx < len(self.cache['image_paths']):
                self.patient_images[patient_id] = self.cache['image_paths'][idx]
            else:
                print(f"Warning: No image path found for patient {patient_id}")

        # Track patients without images
        self.patients_without_images = set(self.clinical_data['bcr_patient_barcode']) - set(self.patient_images.keys())

        print(f"Clinical feature size: {self.clinical_size}")
        print(f"Gene feature size: {self.gene_size}")
        print(f"Final number of samples: {len(self.clinical_data)}")
        print(f"Number of patients with images: {len(self.patient_images)}")

    def shorten_gene_columns(self, gene_expr):
        shortened_columns = {col: col[:12] for col in gene_expr.columns if col.startswith('TCGA')}
        return gene_expr.rename(columns=shortened_columns)

    def process_feature_columns(self):
        self.encoders = {}
        for col in self.feature_columns:
            if col not in self.clinical_data.columns:
                raise ValueError(f"Column {col} not found in clinical data")

            if self.clinical_data[col].dtype == 'object':
                self.clinical_data[col] = self.clinical_data[col].fillna('Unknown')
                le = LabelEncoder()
                self.clinical_data[col] = le.fit_transform(self.clinical_data[col].astype(str))
                self.encoders[col] = le
            else:
                self.clinical_data[col] = pd.to_numeric(self.clinical_data[col], errors='coerce')
                self.clinical_data[col] = self.clinical_data[col].fillna(self.clinical_data[col].median())

    def __len__(self):
        return len(self.clinical_data)

    def __getitem__(self, idx):
        patient_data = self.clinical_data.iloc[idx]
        patient_id = patient_data['bcr_patient_barcode']
        
        clinical_features = patient_data[self.feature_columns].values.astype(float)
        clinical_features = np.nan_to_num(clinical_features)  # Replace NaNs with 0

        gene_features = self.gene_expr.loc[patient_id].values if patient_id in self.gene_expr.index else np.zeros(self.gene_size)
        gene_features = np.pad(gene_features, (0, max(0, self.gene_size - len(gene_features))))[:self.gene_size]

        event_times = torch.tensor(patient_data['OS.time'], dtype=torch.float)
        censored = torch.tensor(1.0, dtype=torch.float)  # Assuming all patients are censored, adjust if needed

        patients_without_images = set(self.clinical_data['bcr_patient_barcode']) - set(self.patient_images.keys())

        # Load image for the patient
        if patient_id in self.patient_images:
            image_path = self.patient_images[patient_id]
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
            except Exception as e:
                print(f"Error loading image for patient {patient_id}: {str(e)}")
                image = torch.zeros((3, 224, 224))
        else:
            image = torch.zeros((3, 224, 224))

        return {
            'clinical': torch.tensor(clinical_features, dtype=torch.float),
            'gene': torch.tensor(gene_features, dtype=torch.float),
            'image': image,
            'event_times': event_times,
            'censored': censored
        }
    
    def print_summary(self):
        print(f"Number of patients without images: {len(self.patients_without_images)}")

    
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
        
        # Login to Hugging Face
        login(token="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")
        
        # Initialize UNI model
        self.uni_model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.uni_model.eval()  # Set to evaluation mode
        
        # Get the transform for UNI
        config = resolve_data_config(self.uni_model.pretrained_cfg, model=self.uni_model)
        self.transform = create_transform(**config)
        
        # Add a final linear layer to match the output size
        self.fc = nn.Linear(self.uni_model.num_features, 64)
    
    def forward(self, x):
        # Check if input is already a tensor
        if not isinstance(x, torch.Tensor):
            x = self.transform(x)
        else:
            # If it's already a tensor, just ensure it's the right shape and type
            if x.dim() == 3:
                x = x.unsqueeze(0)  # Add batch dimension if it's missing
            x = x.float()  # Ensure float type
        
        # Get features from UNI
        with torch.no_grad():
            features = self.uni_model(x)
        
        # Pass through final linear layer
        return self.fc(features)

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
        self.image_net = ImageNet()  # This now uses UNI
        self.final_fc = nn.Linear(64 * 3, 1)
    
    def forward(self, clinical, gene, image):        
        clinical_out = torch.relu(self.clinical_fc(clinical))
        gene_out = torch.relu(self.gene_fc(gene))
        image_out = self.image_net(image)
        
        combined = torch.cat((clinical_out, gene_out, image_out), dim=1)
        output = self.final_fc(combined)
        
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

# Training loop with debugging and evaluation
def train_and_evaluate(train_loader, test_loader, model, cox_criterion, contrastive_criterion, optimizer, epochs=40):
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
    
    # Evaluate C-index on test set after all epochs
    c_index = evaluate_c_index(test_loader, model, device)
    if c_index is not None:
        print(f"Final C-index: {c_index:.4f}")
    else:
        print("Unable to calculate C-index")

def evaluate_c_index(dataloader, model, device):
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
    
    if not all_outputs:
        print("No valid outputs for C-index calculation")
        return None

    # Concatenate results
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    all_event_times = torch.cat(all_event_times).cpu().numpy()
    all_censored = torch.cat(all_censored).cpu().numpy()
    
    try:
        # Compute C-index
        c_index = concordance_index(all_event_times, -all_outputs, all_censored)
        return c_index
    except Exception as e:
        print(f"Error calculating C-index: {str(e)}")
        return None

# Example usage
# Assume clinical_data and gene_expr are pre-loaded DataFrames
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = MultiModalDataset(clinical_data, gene_expr, cache_file, transform=transform)

# Get cancer types for each patient
cancer_types = clinical_data.set_index('bcr_patient_barcode')['type'].loc[dataset.clinical_data['bcr_patient_barcode']].values

# Perform stratified split
train_indices, test_indices = train_test_split(
    range(len(dataset)),
    test_size=0.15,
    stratify=cancer_types,
    random_state=42
)

# Calculate the sizes for the split
total_size = len(dataset)
train_size = int(0.85 * total_size)
test_size = total_size - train_size

# Create Subset datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Initialize and train the model
clinical_size = len(dataset.feature_columns)
gene_size = dataset.gene_size

# Use this simplified model instead of the previous MultiModalNet
model = SimplifiedMultiModalNet(clinical_size=clinical_size, gene_size=gene_size)

cox_criterion = CoxLoss()
contrastive_criterion = ContrastiveLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

train_and_evaluate(train_loader, test_loader, model, cox_criterion, contrastive_criterion, optimizer, epochs=40)
dataset.print_summary()

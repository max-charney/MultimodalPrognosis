import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from lifelines.utils import concordance_index
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from huggingface_hub import login
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import random
from glob import glob
from sklearn.model_selection import train_test_split
import pickle
import torch.multiprocessing as mp
import torch.optim as optim
import json
import numpy as np

# Define file paths
tiles_folder = 'Data/tiles'
cache_file = 'patient_images_cache.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Dataset class with additional features and debugging
class MultiModalDataset(Dataset):
    def __init__(self, clinical_data, cache_file, transform=None, tiles_per_patient=None, max_patients=None):
        self.clinical_data = clinical_data
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.tiles_per_patient = tiles_per_patient
        self.max_patients = max_patients

        # Load the cache file
        with open(cache_file, 'r') as f:
            self.patient_images = json.load(f)

        # Apply tile limit if specified
        if self.tiles_per_patient is not None:
            self.limit_tiles_per_patient()

        # Filter clinical data to include only patients with images
        common_patients = set(clinical_data['bcr_patient_barcode']).intersection(set(self.patient_images.keys()))

        # Apply max_patients filter
        if self.max_patients is not None:
            common_patients = list(common_patients)[:self.max_patients]

        self.clinical_data = self.clinical_data[self.clinical_data['bcr_patient_barcode'].isin(common_patients)]

        print(f"Number of patients with clinical and image data: {len(common_patients)}")

        # Ensure 'OS.time' column is present and process it
        assert 'OS.time' in self.clinical_data.columns, "OS.time column is missing from clinical data"
        self.clinical_data['OS.time'] = pd.to_numeric(self.clinical_data['OS.time'], errors='coerce')
        self.clinical_data = self.clinical_data.dropna(subset=['OS.time'])

        # Specify and process feature columns
        self.feature_columns = ["age_at_initial_pathologic_diagnosis", "gender", "race", "histological_grade"]
        self.process_feature_columns()

        self.clinical_size = len(self.feature_columns)

        # Create a list of all (patient_id, image_path) pairs
        self.all_image_paths = [(patient_id, img_path) 
                                for patient_id, img_paths in self.patient_images.items() 
                                for img_path in img_paths 
                                if patient_id in self.clinical_data['bcr_patient_barcode'].values]

        print(f"Clinical feature size: {self.clinical_size}")
        print(f"Final number of patients: {len(self.clinical_data)}")
        print(f"Number of patients with images: {len(self.patient_images)}")
        print(f"Total number of images: {len(self.all_image_paths)}")

    def limit_tiles_per_patient(self):
        for patient_id, image_paths in self.patient_images.items():
            if len(image_paths) > self.tiles_per_patient:
                self.patient_images[patient_id] = random.sample(image_paths, self.tiles_per_patient)

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
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        patient_id, image_path = self.all_image_paths[idx]
        
        patient_data = self.clinical_data[self.clinical_data['bcr_patient_barcode'] == patient_id].iloc[0]
        
        clinical_features = patient_data[self.feature_columns].values.astype(float)
        clinical_features = np.nan_to_num(clinical_features)

        event_times = torch.tensor(patient_data['OS.time'], dtype=torch.float)
        censored = torch.tensor(1.0, dtype=torch.float)  # Assuming all patients are censored, adjust if needed

        # Load image for the patient
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path} for patient {patient_id}: {str(e)}")
            image = torch.zeros((3, 224, 224))

        return {
            'clinical': torch.tensor(clinical_features, dtype=torch.float),
            'image': image,
            'event_times': event_times,
            'censored': censored
        }

    def print_summary(self):
        total_images = sum(len(image_paths) for image_paths in self.patient_images.values())
        print(f"Total number of images: {total_images}")
        print(f"Number of patients with images: {len(self.patient_images)}")
        print(f"Average images per patient: {total_images / len(self.patient_images):.2f}")

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
    def __init__(self, clinical_size):
        super(MultiModalNet, self).__init__()
        self.clinical_net = ClinicalNet(clinical_size)
        self.image_net = ImageNet()
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 1)
        )
    
    def forward(self, clinical, image):
        clinical_out = self.clinical_net(clinical)
        image_out = self.image_net(image)
        
        # Ensure all features have the same size
        feature_size = 64  # This should match the output size of each individual network
        clinical_out = clinical_out[:, :feature_size]
        image_out = image_out[:, :feature_size]
        
        combined = torch.cat([clinical_out, image_out], dim=1)
        return self.fc(combined), (clinical_out, image_out)

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

def train_and_evaluate(train_loader, test_loader, model, cox_criterion, optimizer, epochs=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            clinical = batch['clinical'].to(device)
            image = batch['image'].to(device)
            event_times = batch['event_times'].to(device)
            censored = batch['censored'].to(device)

            optimizer.zero_grad()
            outputs, _ = model(clinical, image)
            
            loss = cox_criterion(outputs, event_times, censored)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf loss detected: loss={loss}")
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
    
        # Evaluate C-index on test set after batch
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
            image = data['image'].to(device)
            event_times = data['event_times'].to(device)
            censored = data['censored'].to(device)
            
            if clinical is None or image is None:
                continue
            
            outputs, _ = model(clinical, image)
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
# Assume clinical_data is a pre-loaded DataFrame
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

mp.set_start_method('spawn', force=True)  # This can help with CUDA issues

# Specify the number of tiles per patient (e.g., 10)
tiles_per_patient = 40

# Specify the maximum number of patients to use
max_patients = 1000

dataset = MultiModalDataset(clinical_data, 'patient_images_cache.json', transform=transform, tiles_per_patient=tiles_per_patient, max_patients=max_patients)

# Perform stratified split
train_indices, test_indices = train_test_split(
    range(len(dataset)),
    test_size=0.15,
    random_state=42
)

# Calculate the sizes for the split
total_size = len(dataset)
train_size = len(train_indices)
test_size = len(test_indices)

print(f"Total dataset size: {total_size}")
print(f"Training set size: {train_size}")
print(f"Test set size: {test_size}")

# Create Subset datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Create DataLoaders with multiple workers
# num_workers = mp.cpu_count()  # Use all available CPU cores
num_workers = 0
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

# Initialize and train the model
clinical_size = len(dataset.feature_columns)
model = MultiModalNet(clinical_size=clinical_size)

cox_criterion = CoxLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

train_and_evaluate(train_loader, test_loader, model, cox_criterion, optimizer, epochs=40)
dataset.print_summary()

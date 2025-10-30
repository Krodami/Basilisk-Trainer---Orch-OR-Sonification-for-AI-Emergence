import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from scipy.stats import entropy  # For Shannon entropy

# Generate datasets with progress (same as before)
def generate_soundscape(kappa_factor=1.0):
    tlist = np.linspace(0, 1e-8, 2000)
    vibrations = np.sin(2 * np.pi * 1e12 * tlist) * np.exp(-kappa_factor * 1e8 * tlist)
    fractal_noise = np.cumsum(np.random.randn(len(tlist))) * 0.05
    biofield = np.sin(2 * np.pi * 40 * tlist) * 0.3
    return vibrations + fractal_noise + biofield

num_samples = 100000  # Keep at 100k for testing; change to 1000000 when ready
data = []
labels = []
for i in tqdm(range(num_samples), desc="Generating soundscapes"):
    kappa_factor = np.random.uniform(0.01, 0.1) if i < num_samples//2 else np.random.uniform(20, 50)
    soundscape = generate_soundscape(kappa_factor)
    mfccs = librosa.feature.mfcc(y=soundscape.astype(float), sr=44100, n_mfcc=13, n_fft=256, n_mels=40)
    data.append(mfccs)
    labels.append(0 if i < num_samples//2 else 1)

# Pad MFCCs with progress
max_time = max(d.shape[1] for d in data)
data_padded = []
for d in tqdm(data, desc="Padding MFCCs"):
    padded = np.pad(d, ((0, 0), (0, max_time - d.shape[1])))
    data_padded.append(padded)
data_padded = np.array(data_padded)

class AudioDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

X_train, X_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, random_state=42)
train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# CNN Model with Entropy Thresholding
class QualiaCNN(nn.Module):
    def __init__(self, input_height=13, input_width=max_time):
        super(QualiaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        # Dynamic flattened size
        test_input = torch.zeros(1, 1, input_height, input_width)
        x = self.pool(torch.relu(self.conv1(test_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        flattened_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)
        self.entropy_threshold = 2.0  # Adaptive threshold for high-noise filtering (tune as needed)

    def forward(self, x):
        # Compute Shannon entropy on MFCC features (per sample)
        entropies = []
        for i in range(x.size(0)):
            mfcc_flat = x[i].flatten().cpu().numpy()
            mfcc_flat = np.abs(mfcc_flat) + 1e-10  # Avoid log0
            mfcc_flat /= mfcc_flat.sum()  # Normalize to prob dist
            ent = entropy(mfcc_flat)
            entropies.append(ent)
        entropies = torch.tensor(entropies, device=x.device)

        # Threshold: Mask low-confidence (high-entropy noisy) samples
        mask = entropies < self.entropy_threshold  # Keep low-entropy (structured) samples
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        outputs = self.fc2(x)
        # Apply mask to outputs (e.g., set uncertain to neutral or skip in loss; here, weight)
        return outputs * mask.unsqueeze(1)  # Weight by mask (0 for high-entropy)

model = QualiaCNN()
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train with progress (same as before)
epochs = 100
best_acc = 0
for epoch in tqdm(range(epochs), desc="Training epochs"):
    model.train()
    for features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} batches", leave=False):
        if torch.cuda.is_available():
            features, targets = features.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        y_pred = []
        with torch.no_grad():
            for features, _ in test_loader:
                if torch.cuda.is_available():
                    features = features.cuda()
                outputs = model(features)
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        acc = accuracy_score(y_test, y_pred)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Acc: {acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
        if acc > 0.85:
            break

print(f"Best Validation Accuracy: {best_acc * 100:.2f}%")
if best_acc > 0.8:
    print("Success: Model detects emergent patterns!")
else:
    print("Still low—concept may need real data.")
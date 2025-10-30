import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Generate datasets with progress
def generate_soundscape(kappa_factor=1.0):
    tlist = np.linspace(0, 1e-8, 2000)
    vibrations = np.sin(2 * np.pi * 1e12 * tlist) * np.exp(-kappa_factor * 1e8 * tlist)
    fractal_noise = np.cumsum(np.random.randn(len(tlist))) * 0.05
    biofield = np.sin(2 * np.pi * 40 * tlist) * 0.3
    return vibrations + fractal_noise + biofield

num_samples = 10000  # Small for test; scale later
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
        self.features = torch.tensor(features, dtype=torch.float32).permute(0, 2, 1)  # (batch, seq, feat) for transformer
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

# Transformer Model
class QualiaTransformer(nn.Module):
    def __init__(self, input_dim=13, d_model=64, nhead=4, num_layers=2):
        super(QualiaTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, batch_first=True), num_layers)  # Fixed warning
        self.fc = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global avg pool over seq
        x = self.dropout(x)
        return self.fc(x)

# Hyperopt objective
def objective(params):
    model = QualiaTransformer()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()
    epochs = int(params['epochs'])

    for epoch in range(epochs):
        model.train()
        for features, targets in train_loader:
            if torch.cuda.is_available():
                features, targets = features.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    y_pred = []
    with torch.no_grad():
        for features, _ in test_loader:
            if torch.cuda.is_available():
                features = features.cuda()
            outputs = model(features)
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    acc = accuracy_score(y_test, y_pred)
    return {'loss': -acc, 'status': STATUS_OK}

space = {'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)), 'epochs': hp.quniform('epochs', 50, 150, 10)}
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)
print("Best Params: LR", best['lr'], "Epochs", best['epochs'])

# Full Train with Best Params
model = QualiaTransformer()
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=best['lr'])
criterion = nn.CrossEntropyLoss()
epochs = int(best['epochs'])

best_acc = 0
for epoch in tqdm(range(epochs), desc="Full training epochs"):
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

print(f"Best Validation Accuracy: {best_acc * 100:.2f}%")
if best_acc > 0.8:
    print("Success: Model detects emergent patterns!")
else:
    print("Still low—concept may need real data.")
import torch
import torch.nn as nn
from mamba_ssm import Mamba
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns

class StateTransitionModel(nn.Module):
    def __init__(self, latent_dim, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([Mamba(d_model=latent_dim) for _ in range(n_layers)])

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z

class RealNVPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Sequential(nn.Linear(dim // 2, 128), nn.ReLU(), nn.Linear(128, dim // 2))
        self.translate = nn.Sequential(nn.Linear(dim // 2, 128), nn.ReLU(), nn.Linear(128, dim // 2))
        nn.init.zeros_(self.scale[-1].weight)
        nn.init.zeros_(self.scale[-1].bias)
        nn.init.zeros_(self.translate[-1].weight)
        nn.init.zeros_(self.translate[-1].bias)

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        if reverse:
            s = self.scale(x1)
            t = self.translate(x1)
            x2 = (x2 - t) * torch.exp(-s)
        else:
            s = self.scale(x1)
            t = self.translate(x1)
            x2 = x2 * torch.exp(s) + t
        return torch.cat([x1, x2], dim=1)

class NormalizingFlow(nn.Module):
    def __init__(self, dim, n_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([RealNVPBlock(dim) for _ in range(n_blocks)])

    def forward(self, x, reverse=False):
        for block in (reversed(self.blocks) if reverse else self.blocks):
            x = block(x, reverse)
        return x

class AnomalyDetectionModel(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.transition = StateTransitionModel(latent_dim)
        self.flow = NormalizingFlow(latent_dim, n_blocks=4)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, input_dim))

    def forward(self, x):
        z = self.encoder(x)
        z_next = self.transition(z)
        z_flow = self.flow(z_next, reverse=True)
        z_flow = torch.tanh(z_flow)
        x_recon = self.decoder(z_flow)
        return x_recon, z_next

df_cc = pd.read_csv('/data/creditcard.csv')
df_cc_sorted = df_cc.sort_values('Time').reset_index(drop=True)
window_size = 30
windows, labels = [], []

for i in range(len(df_cc_sorted) - window_size + 1):
    window = df_cc_sorted.iloc[i:i + window_size]
    X_window = window.drop(columns=['Time', 'Class']).values
    label = 1 if window['Class'].sum() > 0 else 0
    windows.append(X_window)
    labels.append(label)

X_windows = np.array(windows).astype(np.float32)
y_windows = np.array(labels)
scaler = StandardScaler()
n_samples, seq_len, n_features = X_windows.shape
X_windows = scaler.fit_transform(X_windows.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
X_windows = np.transpose(X_windows, (0, 2, 1))
X_cc_train, X_cc_test, y_cc_train, y_cc_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42, stratify=y_windows)

rocket = MiniRocket()
rocket.fit(X_cc_train)
X_train_trans = rocket.transform(X_cc_train)
X_test_trans = rocket.transform(X_cc_test)
clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
clf.fit(X_train_trans, y_cc_train)
y_pred_cc = clf.predict(X_test_trans)
auc_cc = roc_auc_score(y_cc_test, y_pred_cc)

df_train_ecg = pd.read_csv('data/mitbih_train.csv', header=None)
df_test_ecg = pd.read_csv('/data/mitbih_test.csv', header=None)
X_ecg = pd.concat([df_train_ecg, df_test_ecg])
y_ecg = (X_ecg.iloc[:, -1].values != 0).astype(int)
X_ecg = X_ecg.iloc[:, :-1].values
X_ecg = StandardScaler().fit_transform(X_ecg)
X_ecg_train, X_ecg_test, y_ecg_train, y_ecg_test = train_test_split(X_ecg, y_ecg, test_size=0.2, random_state=42, stratify=y_ecg)

rocket_ecg = MiniRocket()
rocket_ecg.fit(X_ecg_train.reshape(-1, 1, X_ecg_train.shape[1]))
X_train_trans_ecg = rocket_ecg.transform(X_ecg_train.reshape(-1, 1, X_ecg_train.shape[1]))
X_test_trans_ecg = rocket_ecg.transform(X_ecg_test.reshape(-1, 1, X_ecg_test.shape[1]))
clf_ecg = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
clf_ecg.fit(X_train_trans_ecg, y_ecg_train)
y_pred_ecg = clf_ecg.predict(X_test_trans_ecg)
acc_ecg = accuracy_score(y_ecg_test, y_pred_ecg)

model_ecg = AnomalyDetectionModel(input_dim=187, latent_dim=32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ecg = model_ecg.to(device)
optimizer = torch.optim.Adam(model_ecg.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

X_tensor = torch.tensor(X_ecg_train, dtype=torch.float32).to(device)
y_tensor = torch.tensor(X_ecg_train, dtype=torch.float32).to(device)
model_ecg.train()
for epoch in range(10):
    optimizer.zero_grad()
    x_recon, _ = model_ecg(X_tensor)
    loss = loss_fn(x_recon, y_tensor)
    loss.backward()
    optimizer.step()

model_ecg.eval()
with torch.no_grad():
    test_tensor = torch.tensor(X_ecg_test, dtype=torch.float32).to(device)
    recon, _ = model_ecg(test_tensor)
    recon_error = torch.mean((test_tensor - recon) ** 2, dim=1).cpu().numpy()

threshold = np.percentile(recon_error, 95)
y_pred_ours = (recon_error > threshold).astype(int)
acc_ours = accuracy_score(y_ecg_test, y_pred_ours)

results = [
    {"Dataset": "Credit Card Fraud", "Model": "MiniRocket", "Metric": "AUC", "Score": auc_cc},
    {"Dataset": "MIT-BIH ECG", "Model": "MiniRocket", "Metric": "Accuracy", "Score": acc_ecg},
    {"Dataset": "MIT-BIH ECG", "Model": "SSM+Flow (Ours)", "Metric": "Accuracy", "Score": acc_ours}
]

results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Dataset", y="Score", hue="Model")
plt.title("Model Performance Comparison Across Datasets")
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.show()

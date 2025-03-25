import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Appareil utilisé :", device)

path = kagglehub.dataset_download("suvroo/ai-for-sustainable-agriculture-dataset")
file_path = f"{path}/farmer_advisor_dataset.csv"

scaler = StandardScaler()

df = pd.read_csv(file_path, usecols=['Soil_pH', 'Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Crop_Type', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 'Crop_Yield_ton'])
df2 = pd.read_csv(file_path, usecols=['Sustainability_Score'])

df = df.fillna(0)
df2 = df2.fillna(0)
df["Crop_Type"] = df["Crop_Type"].map({"Wheat": 0, "Corn": 1, "Rice": 2, "Soybean": 3})

#++++++++++++++++++++++++++++++++++++++++++++++++++++#
"""
scaler.fit_transform =

X = valeur brute d’une cellule (par exemple, pH = 6.5)
μ = moyenne de toutes les valeurs de cette colonne
σ = écart-type de la colonne

x = ( X - μ ) / σ
"""
#++++++++++++++++++++++++++++++++++++++++++++++++++++#

df[['Soil_pH', 'Rainfall_mm', 'Temperature_C', 'Soil_Moisture', 'Crop_Type']] = scaler.fit_transform(df[['Soil_pH', 'Rainfall_mm', 'Temperature_C', 'Soil_Moisture', 'Crop_Type']])
df[["Crop_Yield_ton", "Fertilizer_Usage_kg", "Pesticide_Usage_kg"]] = np.log1p(df[["Crop_Yield_ton", "Fertilizer_Usage_kg", "Pesticide_Usage_kg"]])
df["Fertilizer_per_yield"] = df["Fertilizer_Usage_kg"] / (df["Crop_Yield_ton"] + 1e-5)
df["Pesticide_per_yield"] = df["Pesticide_Usage_kg"] / (df["Crop_Yield_ton"] + 1e-5)

#++++++++++++++++++++++++++++++++++++++++++++++++++++#
"""
np.log1p(x) ≡ log(1 + x)

→ log(0) = -∞ ❌
→ log(1 + 0) = 0 ✅
"""
#++++++++++++++++++++++++++++++++++++++++++++++++++++#

df2[['Sustainability_Score']] = scaler.fit_transform(df2[['Sustainability_Score']])

corr = pd.concat([df, df2], axis=1).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(
    df.values, df2.values.flatten(), test_size=0.2, random_state=42
)

x_train = torch.tensor(x_train, dtype=torch.float32 )
x_test = torch.tensor(x_test, dtype=torch.float32 )
y_train = torch.tensor(y_train, dtype=torch.float32 )
y_test = torch.tensor(y_test, dtype=torch.float32 )

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)

class Agriculture(nn.Module):
    def __init__(self):
        super(Agriculture, self).__init__()
        self.hidden1 = nn.Linear(10, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)

        self.hidden2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.3)

        self.hidden3 = nn.Linear(1024, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(0.2)

        self.hidden4 = nn.Linear(2048, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.1)

        self.output = nn.Linear(512, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.bn1(self.hidden1(x)))
        x = self.dropout1(x)

        x = self.act(self.bn2(self.hidden2(x)))
        x = self.dropout2(x)

        x = self.act(self.bn3(self.hidden3(x)))
        x = self.dropout3(x)

        x = self.act(self.bn4(self.hidden4(x)))
        x = self.dropout4(x)

        x = self.output(x)
        return x
    
model = Agriculture().to(device)

x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
schreduler = StepLR(optimizer=optimizer, step_size=2000, gamma=0.7)

print("Min y:", y_train.min().item(), "| Max y:", y_train.max().item())

epochs = 15000
best_loss = float('inf')
epochs_no_improve = 0
patience = 100 
early_stop = False
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred.squeeze(1), y_batch)
        loss.backward()
        optimizer.step()
    schreduler.step()
    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss
        epochs_no_improve = 0
        # Sauvegarde du meilleur modèle
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss:.4f} ✅ (sauvegardé)")
    else:
        epochs_no_improve += 1
        print(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss:.4f} ❌ (no improve)")

    if epochs_no_improve >= patience:
        print("Early stopping déclenché 🚨")
        break
        

model.load_state_dict(torch.load('best_model.pt'))
model.eval()
with torch.no_grad():
    y_pred_test = model(x_test).squeeze() 
    mse = nn.L1Loss()(y_pred_test, y_test)
    print(f"Test MSE: {mse.item():.4f}")
    pred = y_pred_test.cpu().numpy().reshape(-1, 1)
    real_pred = scaler.inverse_transform(pred)
    print("Exemples de prédictions réelles :", real_pred[:5])

    y_true_np = y_test.cpu().numpy().reshape(-1, 1)
    real_y = scaler.inverse_transform(y_true_np)

    epsilon = 1e-5
    mape_total = np.mean(np.abs((real_y - real_pred) / (real_y + epsilon))) * 100
    print(f"MAPE (global) : {mape_total:.2f}%")
    mask = real_y > 10
    mape_filtered = np.mean(np.abs((real_y[mask] - real_pred[mask]) / (real_y[mask] + epsilon))) * 100
    print(f"MAPE (>10) : {mape_filtered:.2f}%")
    print(f"Précision estimée (~) : {100 - mape_filtered:.2f}%")


    # Scatter plot des 100 premières prédictions vs réelles
    plt.figure(figsize=(8, 6))
    plt.scatter(real_y[:100], real_pred[:100], alpha=0.7, label="Prédictions")
    plt.plot([0, 100], [0, 100], color='red', linestyle='--', label="Ligne parfaite")

    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title("Prédictions vs Réalité (Sustainability Score)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

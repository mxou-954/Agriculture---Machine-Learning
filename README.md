# 🌾 Agriculture - Machine Learning

Projet de prédiction du **Sustainability Score** d'une culture agricole à l'aide d'un réseau de neurones en **PyTorch**.  
Ce projet utilise un jeu de données provenant de [Kaggle](https://www.kaggle.com/datasets/suvroo/ai-for-sustainable-agriculture-dataset), contenant des caractéristiques liées aux sols, aux cultures et aux intrants utilisés.

---

## 🌟 Objectif

> Prédire avec précision le `Sustainability_Score` en fonction des données mesurées sur les sols, les cultures et les intrants.

---

## 📊 Données utilisées

- `Soil_pH`
- `Soil_Moisture`
- `Rainfall_mm`
- `Temperature_C`
- `Crop_Type` (encodé : Wheat = 0, Corn = 1, Rice = 2, Soybean = 3)
- `Fertilizer_Usage_kg`
- `Pesticide_Usage_kg`
- `Crop_Yield_ton`

### Variables dérivées :
- `Fertilizer_per_yield` = Fertilizer / Yield
- `Pesticide_per_yield` = Pesticide / Yield

---

## 🧪 Prétraitements

```python
# Standardisation des données numériques
StandardScaler()

# Transformation logarithmique sur les grandes valeurs
np.log1p(x)  # log(1 + x), pour éviter log(0)

# Encodage des cultures (Wheat, Corn, etc.)
# Ajout de variables dérivées (rapport engrais/rendement, pesticides/rendement)
```

---

## 🧠 Architecture du modèle

- Framework : PyTorch
- Architecture : réseau dense profond (deep fully connected)
- Activations : `SiLU` (plus fluide que ReLU)
- Normalisation : `BatchNorm1d`
- Régularisation : `Dropout` (de 10% à 30%)
- Optimiseur : `Adam`
- Fonction de perte : `MSELoss` + `L1Loss` pour évaluation

```python
Input (10 features)
   ↓
Linear(10 → 512) → BatchNorm → SiLU → Dropout
   ↓
Linear(512 → 1024) → BatchNorm → SiLU → Dropout
   ↓
Linear(1024 → 2048) → BatchNorm → SiLU → Dropout
   ↓
Linear(2048 → 512) → BatchNorm → SiLU → Dropout
   ↓
Output(512 → 1)
```

---

## 📉 Résultats

- **MAPE (Mean Absolute Percentage Error)** : ~72.65%
- **Précision estimée** : ~27.35%
- **Test L1Loss** : variable selon les runs (~0.04–0.07)
- Les prédictions ont tendance à être centrées autour de la moyenne, avec une sous-performance sur les valeurs extrêmes.

---

## 📌 Hypothèses et remarques

- Le **Sustainability Score** est influencé par des facteurs non présents dans le dataset (ex : méthodes culturales, variétés, irrigation, rotation, etc.)
- Peu de corrélation entre les variables disponibles et la variable cible (confirmé par la **matrice de corrélation**).
- Malgré des efforts sur la profondeur du réseau, le modèle plafonne : hypothèse de **sous-information dans les données**.
- Overfitting contrôlé avec Dropout + Early Stopping.

---

## ▶️ Exécution

### 🔧 Prérequis
```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn kagglehub
```

### 🚀 Lancer le script
```bash
python Agriculture.py
```

---

## 📈 Visualisation

- Matrice de corrélation
- Scatter plot : Prédictions vs Valeurs réelles (100 premiers points)

---

## 📂 Structure du projet

```
📆 Agriculture - Machine Learning
 ├️ 📜 Agriculture.py
 ├️ 📜 README.md
 ├️ 📊 Graphiques (sauvegardés si besoin)
 └️ 📁 best_model.pt (modèle sauvegardé automatiquement)
```

---

## 📬 Auteur

- [mxou-954](https://github.com/mxou-954)


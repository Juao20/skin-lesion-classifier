# Skin Lesion Classifier

Classification d'images de lésions cutanées par deep learning.

## Structure

```
configs/          # fichiers de configuration (hyperparamètres)
src/
  data.py         # dataset, transforms, split
  model.py        # construction du modèle
  train.py        # boucle d'entraînement
  evaluate.py     # métriques, matrice de confusion
notebooks/        # EDA et expérimentations
app/app.py        # démo interactive
reports/figures/  # figures générées
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Utilisation

```bash
python -m src.train --config configs/baseline.yaml
python -m src.evaluate --config configs/baseline.yaml
python app/app.py
```

Les données sont attendues dans `data/` (non versionné).

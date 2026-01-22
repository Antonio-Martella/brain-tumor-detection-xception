# Brain Tumor Classification via Transfer Learning (Xception)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-99.5%25-brightgreen)

## Project Overview
Questo progetto implementa un sistema di **Deep Learning** per la classificazione automatica di tumori cerebrali a partire da immagini di Risonanza Magnetica (MRI).

L'obiettivo è fornire uno strumento di supporto alla diagnosi (CAD) capace di distinguere tra quattro classi principali con elevata precisione, minimizzando i falsi negativi. Il modello sfrutta l'architettura **Xception** pre-addestrata su ImageNet, integrata con un blocco convoluzionale custom (Hybrid Transfer Learning).

##  Dataset
Il dataset è composto da immagini MRI suddivise in quattro classi:
* **Glioma Tumor**
* **Meningioma Tumor**
* **Pituitary Tumor**
* **No Tumor** (Paziente sano)

Le immagini sono state ridimensionate a **224x224 pixel** e normalizzate. Il dataset è stato diviso in Training, Validation e Test set per garantire una valutazione robusta.

## Model Architecture
Abbiamo adottato un approccio ibrido che combina un potente feature extractor con layer specifici per catturare pattern spaziali medici.

1.  **Backbone:** `Xception` (ImageNet weights, frozen backbone).
2.  **Custom Head:**
    * **SeparableConv2D** (512 filtri, kernel 3x3): Riduce i parametri mantenendo l'efficienza spaziale.
    * **BatchNormalization**: Per la stabilità dei gradienti.
    * **GlobalAveragePooling2D**: Riduzione dimensionale da 3D a vettore 1D.
    * **Dense Layer** (128 neuroni, ReLU) + **Dropout**.
    * **Output Layer**: Dense (4 neuroni, Softmax).

## Training Strategy
Il training è stato ottimizzato per prevenire l'overfitting e massimizzare la convergenza:

* **Optimizer:** Adamax (Learning Rate iniziale: 0.001).
* **Loss Function:** Categorical Crossentropy.
* **Callbacks:**
    * `ReduceLROnPlateau`: Riduce il LR se la loss stagna.
    * `EarlyStopping`: Ferma il training se non ci sono miglioramenti (Restore best weights attivo).
    * `ModelCheckpoint`: Salva solo il modello migliore.

##  Results & Performance
Il modello ha dimostrato prestazioni eccezionali, con uno scarto minimo tra Training e Test, indicando assenza di overfitting.

| Metric | Training | Validation | Test |
| :--- | :---: | :---: | :---: |
| **Loss** | 0.0050 | 0.0222 | **0.0358** |
| **Accuracy** | 100.0% | 99.5% | **99.5%** |

### Classification Report (Test Set)
| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Glioma** | 0.99 | 0.99 | 0.99 |
| **Meningioma** | 0.99 | 0.99 | 0.99 |
| **No Tumor** | **1.00** | **1.00** | **1.00** |
| **Pituitary** | **1.00** | **1.00** | **1.00** |

### Error Analysis
Dalla **Matrice di Confusione**, su oltre 1300 immagini di test:
* **0 Errori** sulla classe "No Tumor" (Affidabilità massima sui sani).
* **0 Errori** sulla classe "Pituitary".
* Lieve confusione (solo 4 casi totali scambiati) tra **Glioma** e **Meningioma**, dovuta a similarità morfologiche in specifiche scansioni.

## Requirements & Installation
Per replicare l'ambiente, sono necessarie le seguenti librerie:

```python
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Installation & Setup

Per eseguire questo progetto in locale, segui questi passaggi:

**1. Clona la repository**
Apri il terminale e scrivi:
```bash
git clone https://github.com/Antonio-Martella/brain-tumor-classification-xception.git
```

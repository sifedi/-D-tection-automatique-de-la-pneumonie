🎯 Objectif
Développer un modèle de classification binaire (pneumonie vs normal) à partir d’images de radiographies pulmonaires, en utilisant EfficientNetB0 pré-entraîné sur ImageNet, combiné à un prétraitement robuste et une phase de fine-tuning pour optimiser la performance.

⚙️ Méthodologie
Utilisation de ImageDataGenerator pour :

la normalisation des images,

l’augmentation des données (rotations, zoom, flips...).

Architecture basée sur :

EfficientNetB0 (gelé dans un premier temps),

couches personnalisées avec GlobalAveragePooling, Dropout, et Dense.

Entraînement en deux phases :

Phase 1 : entraînement avec EfficientNet gelé,

Phase 2 : fine-tuning avec des couches débloquées.

Optimisation avec :

EarlyStopping pour éviter le surapprentissage,

ModelCheckpoint pour sauvegarder les meilleurs poids.

data est accessible sur ce site : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

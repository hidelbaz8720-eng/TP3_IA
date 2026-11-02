# TP 3 : Du vecteur à la décision – Classification binaire et entraînement de modèle

##  Module
**Introduction à l'Intelligence Artificielle – MLIAEdu**

##  Objectif du TP

L'objectif de ce TP est de mettre en pratique les techniques de NLP et de Machine Learning pour réaliser une classification binaire de données textuelles.

Le travail consiste à construire un système capable de classifier automatiquement des messages en deux catégories :
- **Ham :** messages normaux.
- **Spam :** messages indésirables.

---

##  Partie 1 : Classification binaire

### 1. Choix du jeu de données

Dataset utilisé : **SMS Spam Collection**

Ce jeu de données contient des messages SMS répartis en deux classes :
- `ham` : message normal.
- `spam` : message indésirable.

Les deux variables principales sont :
- `text` : contenu du message.
- `label` : catégorie du message à prédire.

### 2. Prétraitement du texte (NLP)

Le prétraitement permet de nettoyer et normaliser les textes avant leur transformation en vecteurs numériques.

Les opérations réalisées :

- Conversion des lettres en minuscules avec `lower()`.
- Suppression des chiffres avec les expressions régulières.
- Suppression de la ponctuation.
- Suppression des espaces inutiles.



### 3. Vectorisation du texte

La méthode **TF-IDF (Term Frequency – Inverse Document Frequency)** est utilisée pour transformer les messages en vecteurs numériques.

Bibliothèque utilisée :

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

TF-IDF attribue un poids aux mots selon leur fréquence dans un message et leur rareté dans l'ensemble des documents.

### 4. Séparation des données

Les données sont divisées en deux ensembles :

- **80 % :** données d'entraînement (`X_train`, `y_train`).
- **20 % :** données de test (`X_test`, `y_test`).

Fonction utilisée :

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Explication :
- `test_size=0.2` : réserve 20 % des données pour le test.
- `random_state=42` : garantit la reproductibilité de la séparation.
- `stratify=y` : conserve les proportions des classes.

### 5. Entraînement des modèles

Trois algorithmes de classification sont utilisés pour comparer leurs performances :

| Algorithme | Principe |
|---|---|
| Naive Bayes | Classification basée sur les probabilités |
| Régression logistique | Prédit la probabilité d'appartenance à une classe |
| SVM | Cherche une frontière optimale entre les classes |

Bibliothèques utilisées :

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
```

### 6. Validation croisée (K-Fold)

La validation croisée permet d'évaluer les performances des modèles sur plusieurs divisions des données d'entraînement.

Dans ce TP, on utilise une validation croisée à 5 plis (`k=5`).

Fonctions utilisées :

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

La fonction `cross_val_score()` permet de calculer les scores de performance pour chaque pli.



## 🧠 Notions importantes à retenir

| Notion | Définition |
|---|---|
| Classification binaire | Prédire une observation parmi deux classes |
| Prétraitement NLP | Nettoyer et normaliser les textes |
| Vectorisation | Transformer les textes en données numériques |
| TF-IDF | Mesurer l'importance des mots |
| Train set | Ensemble utilisé pour entraîner le modèle |
| Test set | Ensemble utilisé pour évaluer le modèle |
| K-Fold | Technique de validation croisée |
| Overfitting | Le modèle apprend trop les données d'entraînement et généralise mal |

##  Conclusion

Ce TP m'a permis de comprendre les différentes étapes nécessaires pour construire un modèle de classification binaire à partir de données textuelles.

J'ai appris à :
- Nettoyer et préparer des données textuelles.
- Transformer les textes en vecteurs numériques avec TF-IDF.
- Séparer les données en ensembles d'entraînement et de test.
- Entraîner plusieurs algorithmes de classification.
- Évaluer les modèles avec la validation croisée à 5 plis.
- Comparer les performances de différents algorithmes.


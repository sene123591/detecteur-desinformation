# 🔍 Détecteur de désinformation

Projet final NLP — TP10, Sujet A. Système de classification binaire qui évalue
la vraisemblance qu'un texte soit de la désinformation (**Fake**) ou une
information vérifiée (**Real**), avec une explication des mots ayant le plus
influencé la décision.

**🚀 Application en ligne :** https://detecteur-desinformation-3occrupuhtrqussyvzhjmj.streamlit.app

## Sommaire

- [Aperçu](#aperçu)
- [Données](#données)
- [Approche](#approche)
- [Résultats](#résultats)
- [Structure du projet](#structure-du-projet)
- [Installation et relance en local](#installation-et-relance-en-local)
- [Limites et analyse honnête](#limites-et-analyse-honnête)

## Aperçu

L'application permet de coller le texte d'un article et d'obtenir :
- un verdict (**Fake** / **Real**)
- un score de confiance
- les mots ayant le plus influencé la décision, avec un code couleur
  (vert = pousse vers *Real*, rouge = pousse vers *Fake*)

Deux modèles sont proposés au choix dans l'application :
1. **Modèle simple** : TF-IDF + LinearSVC
2. **Modèle avancé** : DistilBERT fine-tuné

## Données

Le projet utilise le dataset **[WELFake](https://huggingface.co/datasets/davanstrien/WELFake)**,
une combinaison de plusieurs datasets fake news, contenant environ 72 000
articles avec :
- `title` : titre de l'article
- `text` : contenu de l'article
- `label` : `0` = Fake, `1` = Real

## Approche

| Étape | Détail |
|---|---|
| Exploration | distribution des classes, longueur des textes, doublons/valeurs manquantes, exemples |
| Nettoyage | suppression des doublons/valeurs manquantes, fusion titre + texte |
| Baseline | TF-IDF (20 000 features, bigrammes) + LinearSVC |
| Modèle avancé | DistilBERT (`distilbert-base-uncased`) fine-tuné, 2 epochs, `fp16` |
| Évaluation | accuracy, F1-score, matrice de confusion, analyse des erreurs |
| Interprétabilité | poids TF-IDF (modèle simple) / méthode leave-one-out (modèle avancé) |

Le modèle avancé a été entraîné sur un sous-échantillon de 8 000 exemples
(sur les 72 000 disponibles) pour respecter la contrainte de temps du projet
(6h) — un choix documenté et assumé plutôt qu'une limite cachée.

## Résultats

| Modèle | Accuracy | F1-score |
|---|---|---|
| TF-IDF + LinearSVC (baseline) | *voir notebook, Étape 7* | *voir notebook, Étape 7* |
| DistilBERT fine-tuné | 98.6 % | 98.4 % |

> ℹ️ Complétez la ligne de la baseline avec les chiffres exacts affichés
> dans le notebook (section "Comparer les deux modèles") avant de rendre
> le projet.

## Structure du projet

```
detecteur-desinformation/
├── app.py                          # Application Streamlit
├── requirements.txt                # Dépendances de l'application
├── modele_simple.joblib            # Modèle TF-IDF + LinearSVC entraîné
├── tfidf_vectoriseur.joblib        # Vectoriseur TF-IDF entraîné
├── README.md                       # Ce fichier
└── projet_desinformation_debutant.ipynb   # Notebook complet (exploration → évaluation)
```

Le modèle DistilBERT fine-tuné (~260 Mo) est hébergé séparément sur le
Hugging Face Hub, car trop volumineux pour GitHub :
👉 https://huggingface.co/sene90/distilbert-welfake-desinformation

L'application le télécharge automatiquement au premier lancement.

## Installation et relance en local

```bash
# 1. Cloner le repo
git clone https://github.com/sene123591/detecteur-desinformation.git
cd detecteur-desinformation

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse
`http://localhost:8501`. Le modèle DistilBERT est téléchargé automatiquement
depuis le Hugging Face Hub au premier lancement (connexion internet requise).

Pour relancer l'entraînement depuis zéro (exploration, baseline, fine-tuning),
ouvrez `projet_desinformation_debutant.ipynb` sur
[Google Colab](https://colab.research.google.com) avec un GPU activé
(`Exécution > Modifier le type d'exécution > GPU`).

## Limites et analyse honnête

- **Sous-échantillonnage** : le modèle avancé a été entraîné sur 8 000
  exemples sur les 72 000 disponibles, pour tenir dans le temps imparti.
  Un entraînement sur le dataset complet améliorerait probablement encore
  les scores.
- **Textes hors distribution** : sur des textes courts ou neutres qui
  diffèrent du style des articles du dataset d'entraînement, les deux
  modèles peuvent être en désaccord (observé en test manuel), ce qui
  illustre les limites de généralisation d'un modèle entraîné sur un
  dataset spécifique.
- **Outil d'aide à la réflexion, pas un vérificateur de faits officiel** :
  l'application détecte des *patterns linguistiques* statistiquement
  associés à la désinformation dans le dataset d'entraînement, elle ne
  vérifie pas les faits eux-mêmes.

Voir le notebook (section "Analyse des erreurs") pour des exemples concrets
de textes mal classés et leurs points communs.

---

Projet réalisé dans le cadre du TP10 — Projet Final NLP (Licence).

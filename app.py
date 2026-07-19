"""
Application Streamlit — Détecteur de désinformation
TP 10 — Projet Final NLP — Sujet A

Cette application charge les modèles entraînés dans le notebook
(projet_desinformation_debutant.ipynb) et permet à un utilisateur
de coller un texte pour savoir s'il ressemble plutôt à un article
FAKE (faux) ou REAL (vrai), avec :
- un score de confiance
- les mots qui ont le plus influencé la décision
"""

import streamlit as st
import joblib
import numpy as np

# -----------------------------------------------------------------
# Configuration générale de la page
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Détecteur de désinformation",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Détecteur de désinformation")
st.write(
    "Collez le texte d'un article ci-dessous. L'application indique s'il "
    "ressemble plutôt à une information **vraie (Real)** ou **fausse (Fake)**, "
    "avec les mots qui ont le plus influencé la décision."
)

# -----------------------------------------------------------------
# Chargement des modèles (mis en cache pour ne pas recharger à chaque clic)
# -----------------------------------------------------------------

@st.cache_resource
def charger_modele_simple():
    """Charge le modèle simple : TF-IDF + LinearSVC."""
    vectoriseur = joblib.load("tfidf_vectoriseur.joblib")
    modele = joblib.load("modele_simple.joblib")
    return vectoriseur, modele


# Remplacez cet identifiant par le vôtre : "votre_nom_utilisateur/nom_du_repo"
REPO_MODELE_AVANCE = "sene90/distilbert-welfake-desinformation"

@st.cache_resource
def charger_modele_avance():
    """Charge le modèle avancé : DistilBERT fine-tuné, hébergé sur le Hugging Face Hub."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(REPO_MODELE_AVANCE)
    modele = AutoModelForSequenceClassification.from_pretrained(REPO_MODELE_AVANCE)
    modele.eval()
    return tokenizer, modele


# -----------------------------------------------------------------
# Fonctions de prédiction
# -----------------------------------------------------------------

def predire_avec_modele_simple(texte, vectoriseur, modele):
    """Retourne (label, confiance, mots_importants) avec le modèle TF-IDF + LinearSVC."""
    texte_vectorise = vectoriseur.transform([texte])

    # LinearSVC ne donne pas directement une probabilité, mais on peut
    # utiliser la distance à la frontière de décision comme indicateur de confiance
    score_decision = modele.decision_function(texte_vectorise)[0]
    prediction = modele.predict(texte_vectorise)[0]

    # On transforme le score en une confiance entre 0 et 1 (fonction sigmoïde)
    confiance = 1 / (1 + np.exp(-abs(score_decision)))

    # Mots importants : parmi les mots du texte, lesquels ont le plus de poids ?
    mots_du_texte = texte_vectorise.nonzero()[1]
    noms_mots = np.array(vectoriseur.get_feature_names_out())
    poids = modele.coef_[0]

    mots_avec_poids = [(noms_mots[i], poids[i]) for i in mots_du_texte]
    mots_avec_poids.sort(key=lambda x: abs(x[1]), reverse=True)
    top_mots = mots_avec_poids[:10]

    label = "Real" if prediction == 1 else "Fake"
    return label, confiance, top_mots


def predire_avec_modele_avance(texte, tokenizer, modele):
    """Retourne (label, confiance, mots_importants) avec DistilBERT."""
    import torch

    entrees = tokenizer(texte, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        sorties = modele(**entrees)
        probabilites = torch.nn.functional.softmax(sorties.logits, dim=-1)[0]

    prediction = int(torch.argmax(probabilites))
    confiance = float(probabilites[prediction])
    label = "Real" if prediction == 1 else "Fake"

    # Mots importants : on enlève chaque mot un par un et on regarde
    # de combien la probabilité de la classe prédite change (méthode simple
    # dite "leave-one-out", facile à comprendre pour un débutant)
    mots = texte.split()
    if len(mots) > 60:
        mots = mots[:60]  # on limite pour rester rapide

    # Si la classe prédite est "Fake" (0), on inverse le signe pour que
    # l'importance reste toujours exprimée par rapport à "Real" (comme pour
    # le modèle simple) : positif = pousse vers Real, négatif = pousse vers Fake.
    signe = 1 if prediction == 1 else -1

    importances = []
    for i in range(len(mots)):
        texte_sans_mot = " ".join(mots[:i] + mots[i + 1:])
        entrees_sans_mot = tokenizer(
            texte_sans_mot, return_tensors="pt", truncation=True, max_length=256
        )
        with torch.no_grad():
            sortie_sans_mot = modele(**entrees_sans_mot)
            proba_sans_mot = torch.nn.functional.softmax(sortie_sans_mot.logits, dim=-1)[0]
        variation = signe * float(probabilites[prediction] - proba_sans_mot[prediction])
        importances.append((mots[i], variation))

    importances.sort(key=lambda x: abs(x[1]), reverse=True)
    top_mots = importances[:10]

    return label, confiance, top_mots


# -----------------------------------------------------------------
# Interface utilisateur
# -----------------------------------------------------------------

with st.sidebar:
    st.header("Réglages")
    choix_modele = st.radio(
        "Quel modèle utiliser ?",
        ["Modèle simple (TF-IDF + LinearSVC)", "Modèle avancé (DistilBERT)"],
        help="Le modèle simple est plus rapide. Le modèle avancé est plus précis mais un peu plus lent.",
    )
    st.markdown("---")
    st.caption(
        "Projet TP10 — Sujet A : Détecteur de désinformation. "
        "Les deux modèles ont été entraînés sur le dataset WELFake."
    )

texte_utilisateur = st.text_area(
    "Texte à analyser",
    height=200,
    placeholder="Collez ici le titre et/ou le contenu d'un article...",
)

bouton_analyser = st.button("Analyser le texte", type="primary")

if bouton_analyser:
    if not texte_utilisateur.strip():
        st.warning("Merci de coller un texte avant d'analyser.")
    else:
        with st.spinner("Analyse en cours..."):
            if choix_modele == "Modèle simple (TF-IDF + LinearSVC)":
                vectoriseur, modele = charger_modele_simple()
                label, confiance, top_mots = predire_avec_modele_simple(
                    texte_utilisateur, vectoriseur, modele
                )
            else:
                tokenizer, modele = charger_modele_avance()
                label, confiance, top_mots = predire_avec_modele_avance(
                    texte_utilisateur, tokenizer, modele
                )

        # --- Affichage du résultat ---
        st.markdown("### Résultat")
        if label == "Fake":
            st.error(f"⚠️ Ce texte ressemble à une **désinformation (Fake)**")
        else:
            st.success(f"✅ Ce texte ressemble à une **information vérifiée (Real)**")

        st.metric("Confiance du modèle", f"{confiance * 100:.1f} %")
        st.progress(min(float(confiance), 1.0))

        # --- Mots importants ---
        st.markdown("### Mots ayant le plus influencé la décision")
        st.caption(
            "Barre verte = pousse vers *Real*, barre rouge = pousse vers *Fake*."
        )

        for mot, poids in top_mots:
            couleur = "green" if poids > 0 else "red"
            largeur = min(abs(float(poids)) * 100, 100)
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;'>"
                f"<span style='width:120px;display:inline-block'>{mot}</span>"
                f"<div style='background:{couleur};height:14px;width:{largeur}%;border-radius:4px;'></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption(
            "⚠️ Cet outil est une aide à la réflexion, pas un vérificateur de faits "
            "officiel. Croisez toujours l'information avec des sources fiables."
        )

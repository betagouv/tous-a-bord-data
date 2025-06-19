"""
Prompt pour la classification de contexte TSST (Tarification Sociale et
Solidaire des Transports)

Ce prompt est utilisé par le service tsst_spacy_llm_task.py pour déterminer
si un texte concerne la tarification sociale et solidaire des transports (TSST)
ou non.
"""

# Prompt principal pour la classification TSST
TSST_CLASSIFICATION_PROMPT = """
Tu es un expert en analyse de texte spécialisé dans la détection de contextes
liés à la tarification sociale et solidaire des transports (TSST).

Ta tâche est d'analyser le texte fourni et de déterminer s'il concerne
spécifiquement la tarification sociale et solidaire des transports publics
(bus, métro, tram, train régional, etc.) ou la tarification des transports
en général.

Un texte concerne la TSST s'il parle de tarifs réduits, préférentiels ou
gratuits pour des catégories spécifiques de personnes (bas revenus, chômeurs,
étudiants, seniors, etc.) dans le contexte des transports publics.

Un texte concerne également la TSST s'il parle de tarification des transports
en général, même s'il ne mentionne pas explicitement des réductions pour des
catégories spécifiques. Toute information sur les tarifs de transport public
est pertinente.

Un texte ne concerne PAS la TSST s'il mentionne des prix, des euros (€) ou
des tarifs, mais dans un autre contexte (comme des loyers, des prêts, des
aides au logement, etc.) sans lien avec les transports.

Voici quelques indices pour t'aider à identifier un contexte TSST :
1. Présence de termes liés aux transports (bus, métro, tram, train, mobilité)
2. Mention de tarifs sociaux, solidaires, réduits ou préférentiels
3. Référence à des catégories de bénéficiaires (étudiants, seniors, etc.)
4. Mention de critères d'éligibilité (quotient familial, âge, statut, etc.)
5. Information sur les prix des tickets, abonnements ou cartes de transport,
   même sans mention de réduction

Classifie le texte en TSST ou NON_TSST.
"""

# Exemples positifs (textes concernant la TSST)
TSST_POSITIVE_EXAMPLES = [
    """
    Les bénéficiaires du RSA peuvent obtenir une réduction de 75% sur leur
    abonnement mensuel de transport en commun. Pour en bénéficier, présentez
    votre attestation CAF et une pièce d'identité à l'agence commerciale.
    """,
    """
    Tarification solidaire : Les personnes dont le quotient familial est
    inférieur à 600€ peuvent bénéficier de la gratuité des transports.
    Les personnes dont le quotient familial est compris entre 600€ et 800€
    peuvent bénéficier d'une réduction de 50% sur leur abonnement.
    """,
    """
    Les étudiants boursiers bénéficient d'un tarif préférentiel de 15€ par
    mois pour l'accès illimité au réseau de bus et tramway de la métropole.
    """,
    """
    Le ticket unitaire de bus coûte 1,50€. L'abonnement mensuel est disponible
    au tarif de 45€ pour un accès illimité au réseau de transport de la ville.
    """,
]

# Exemples négatifs (textes ne concernant PAS la TSST)
TSST_NEGATIVE_EXAMPLES = [
    """
    La mairie propose des aides au logement pour les familles à faible revenu.
    Les bénéficiaires du RSA peuvent obtenir jusqu'à 200€ par mois pour leur
     loyer.
    """,
    """
    Le prix des repas à la cantine scolaire est fixé à 3,50€ pour les enfants
    dont le quotient familial est supérieur à 800€, et à 1€ pour les familles
    dont le quotient familial est inférieur à 800€.
    """,
    """
    Les tarifs de location des salles municipales varient de 50€ à 500€ selon
    la taille de la salle et la durée de location. Les associations locales
    bénéficient d'une réduction de 30%.
    """,
]


def generate_tsst_classification_prompt_with_examples(text_to_classify):
    """
    Génère un prompt complet pour la classification TSST avec des exemples.

    Args:
        text_to_classify: Le texte à classifier

    Returns:
        str: Le prompt complet avec exemples
    """
    prompt = TSST_CLASSIFICATION_PROMPT + "\n\n"

    # Ajouter des exemples positifs
    prompt += "Voici des exemples de textes qui concernent la TSST :\n\n"
    for i, example in enumerate(TSST_POSITIVE_EXAMPLES, 1):
        prompt += (
            f"Exemple positif {i}:\n{example.strip()}\n"
            f"Classification: TSST\n\n"
        )

    # Ajouter des exemples négatifs
    prompt += (
        "Voici des exemples de textes qui ne concernent PAS la TSST :\n\n"
    )
    for i, example in enumerate(TSST_NEGATIVE_EXAMPLES, 1):
        prompt += (
            f"Exemple négatif {i}:\n{example.strip()}\n"
            f"Classification: NON_TSST\n\n"
        )

    # Ajouter le texte à classifier
    prompt += (
        f"Texte à classifier :\n{text_to_classify}\n\n" f"Classification :"
    )

    return prompt

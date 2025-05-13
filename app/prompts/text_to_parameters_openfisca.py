def text_to_openfisca_parameters(text):
    return f"""
Tu es un expert en modélisation OpenFisca. À partir des phrases
suivantes extraites d'un site web de transport en commun,
génère un fichier YAML de paramètres pour OpenFisca.
Ce fichier doit contenir les barèmes, les seuils et les conditions
d'éligibilité pour les tarifs standards, réduits et solidaires.

Exemple de structure attendue :
```yaml
artois_mobilites:
  tarifs:
    abonnement:
      standard:
        jeune: 5  # €/mois
        tout_public: 14  # €/mois
        senior: 5  # €/mois
      solidaire:
        montant: 5  # €/mois
        conditions:
          - demandeur_emploi
          - beneficiaire_cmi
          - beneficiaire_asf
    ticket:
      unitaire: 1.5  # €
      carnet_10: 12.6  # €
```

IMPORTANT :
- Conserve les montants exacts et les unités (€, /mois, /an).
- Respecte la hiérarchie des barèmes (standard, réduit, solidaire).
- Inclus toutes les conditions d'éligibilité mentionnées.
- Ne fais pas d'interprétation ou de résumé.

Phrases extraites :
{text}
"""

def text_to_openfisca_rules(text):
    return f"""
Tu es un expert en modélisation OpenFisca. À partir des phrases
suivantes extraites d'un site web de transport en commun,
génère une règle Python pour OpenFisca qui calcule le montant
de l'abonnement ou du ticket en fonction des conditions d'éligibilité.

Exemple de structure attendue :
```python
class artois_mobilites_tarif_transport(Variable):
    value_type = float
    entity = Menage
    definition_period = MONTH
    label = "Tarif de transport Artois Mobilités"

    def formula(menage, period):
        age = menage.personne_de_reference('age', period)
        demandeur_emploi = menage.personne_de_reference(
            'demandeur_emploi',
            period
        )
        beneficiaire_cmi = menage.personne_de_reference(
            'beneficiaire_cmi',
            period
        )
        beneficiaire_asf = menage.personne_de_reference(
            'beneficiaire_asf',
            period
        )

        # Tarif standard
        tarif_standard = select(
            [age < 26, age >= 65],
            [5, 5],
            default=14
        )

        # Tarif solidaire
        tarif_solidaire = 5
        eligibilite_solidaire = (
            demandeur_emploi
            + beneficiaire_cmi
            + beneficiaire_asf
        )

        return select(
            [eligibilite_solidaire],
            [tarif_solidaire],
            default=tarif_standard
        )
```

IMPORTANT :
- Utilise les variables OpenFisca standard (age, demandeur_emploi, etc.).
- Respecte la logique des barèmes (standard, réduit, solidaire).
- Inclus toutes les conditions d'éligibilité mentionnées.
- Ne fais pas d'interprétation ou de résumé.

Phrases extraites :
{text}
"""

def text_to_openfisca_parameters(text):
    return """
Tu es un expert en modélisation OpenFisca. À partir des phrases
suivantes extraites d'un site web de transport en commun,
génère un ensemble de fichiers YAML de paramètres pour OpenFisca.

Structure des fichiers :

1. tarifs_tickets.yaml :
```yaml
description: Tarifs des tickets pour [AOM]
metadata:
  date: 2024-03-20
  source: [source]

tarifs_tickets:
  unite:
    description: Ticket unitaire
    valeur: 1.80
  dix_voyages:
    description: Carnet 10 voyages
    valeur: 17.50
  # autres types de tickets...
```

2. tarifs_abonnements.yaml:
```yaml
description: Tarifs des abonnements standards pour [AOM]
metadata:
  date: 2024-03-20
  source: [source]

tarifs_abonnements:
  mensuel:
    description: Abonnement mensuel standard
    valeur: 62.00
```

3. tarifs_scolaires.yaml :
description: Tarifs scolaires pour [AOM]
metadata:
  date: 2024-03-20
  source: [source]

tarifs_scolaires:
  description: Tarifs spécifiques pour les scolaires
  montant_annuel: 374.40
  frais_dossier: 8.00
  conditions:
    age_max: 26
    statut: etudiant_ou_scolaire

4. baremes.yaml:
```yaml
description: Barèmes de réduction pour [AOM]
metadata:
  date: 2024-03-20
  source: [source]

baremes:
  tranches_age:
    enfant:
      description: Tarif enfant
      age_max: 12
      reduction: 45.00
      type: montant  # type de réduction : montant ou pourcentage
    adolescent:
      description: Tarif adolescent
      age_min: 12
      age_max: 18
      reduction: 33.00
      type: montant

  quotient_familial:
    type: CAF
    type_reduction: pourcentage  # type de réduction pour l'ensemble des seuils
    seuils:
      - valeur: 350
        reduction: 1.00
        type: pourcentage
      - valeur: 500
        reduction: 0.90
        type: pourcentage
```

5. conditions_eligibilite.yaml :
```yaml
description: Conditions d'éligibilité pour les tarifs solidaires de [AOM]
metadata:
  date: 2024-03-20
  source: [source]

statuts:
  invalidite:
    description: Invalidité
    taux_minimum: 80
    reduction: 1.00
    type: pourcentage
```

6. zones.yaml:
```yaml
description: Zones géographiques et tarifs associés pour [AOM]
metadata:
  date: 2024-03-20
  source: [source]

zones:
  type: multiple
  liste:
    - nom: Zone 1
      description: Centre-ville
      tarifs:
        ticket: 1.80
        abonnement_mensuel: 62.00
        abonnement_annuel: 620.00
    - nom: Zone 2
      description: Première couronne
      tarifs:
        ticket: 2.20
        abonnement_mensuel: 72.00
        abonnement_annuel: 720.00
    - nom: Zone 3
      description: Deuxième couronne
      tarifs:
        ticket: 2.60
        abonnement_mensuel: 82.00
        abonnement_annuel: 820.00
```

7. conditions_specifiques.yaml :
```yaml
description: Conditions spécifiques pour [AOM]
metadata:
  date: 2024-03-20
  source: [source]

conditions_specifiques:
  gratuit_weekend:
    description: Gratuité les week-ends
    active: true
    jours:
      - samedi
      - dimanche
```

RÈGLES IMPORTANTES :
1. Tarifs standards :
   - Les tarifs de base (tickets et abonnements) doivent être
   dans leurs fichiers respectifs
   - Ne pas inclure les tarifs réduits dans ces fichiers

2. Barèmes et réductions :
   - Toutes les réductions par âge doivent être dans `baremes.yaml`
   sous forme de montants en euros
   - Les réductions en pourcentage (QF, statuts spéciaux) doivent
   être exprimées en décimal
   (0.75 pour 75%)
   - Les réductions en montant doivent être exprimées en euros
   - Toujours spécifier le type de réduction (montant ou pourcentage)
   pour chaque barème

3. Structure des fichiers :
   - Ne créer que les fichiers pertinents selon les informations disponibles
   - Inclure les métadonnées (date, source) pour chaque fichier
   - Respecter la structure exacte des fichiers

4. Format des réductions :
   - Pour les réductions par âge : montant en euros
   (différence entre tarif plein et tarif réduit)
   - Pour les réductions en pourcentage : valeur décimale (0.75 pour 75%)
   - Pour les gratuités : 1.00 (100% de réduction)

5. Conditions spécifiques :
   - Inclure les conditions particulières
   (gratuité week-end, intermodalite, etc.)
   - Spécifier les jours de gratuité si applicable
"""

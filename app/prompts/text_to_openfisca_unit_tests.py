def text_to_openfisca_unit_tests(text):
    return f"""
Tu es un expert en modélisation OpenFisca. À partir des phrases
suivantes extraites d'un site web de transport en commun,
génère un test YAML pour OpenFisca qui valide le calcul du tarif de transport.

Exemple de structure attendue :
```yaml
- name: Test tarif standard jeune
  period: 2024-01
  input:
    menage:
      personne_de_reference:
        age: 20
        demandeur_emploi: false
        beneficiaire_cmi: false
        beneficiaire_asf: false
  output:
    artois_mobilites_tarif_transport: 5

- name: Test tarif standard senior
  period: 2024-01
  input:
    menage:
      personne_de_reference:
        age: 70
        demandeur_emploi: false
        beneficiaire_cmi: false
        beneficiaire_asf: false
  output:
    artois_mobilites_tarif_transport: 5

- name: Test tarif solidaire demandeur emploi
  period: 2024-01
  input:
    menage:
      personne_de_reference:
        age: 30
        demandeur_emploi: true
        beneficiaire_cmi: false
        beneficiaire_asf: false
  output:
    artois_mobilites_tarif_transport: 5
```

IMPORTANT :
- Inclus des tests pour chaque barème (standard, réduit, solidaire).
- Vérifie les conditions d'éligibilité (âge, demandeur d'emploi, CMI, ASF).
- Ne fais pas d'interprétation ou de résumé.

Phrases extraites :
{text}
"""

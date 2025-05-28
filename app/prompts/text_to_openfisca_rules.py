def text_to_openfisca_rules(text):
    return f"""
Tu es un expert en modélisation OpenFisca. À partir paramètres yaml suivants,
et en t'appuyant sur la librairie openfisca-france-local,
génère les règles de calcul de tarification et tarification sociale des
transports pour cet AOM.

IMPORTANT :
- Utilise les variables OpenFisca standard
(age, demandeur_emploi, aah, css, rsa etc.) comme par exemple dans
@tarification_solidaire_transport.py
- Respecte la logique des tarifs (tickets, abonnements, scolaires,
zones) comme éventuellement spécifié dans les parameters
@tarifs_tickets.yaml @tarifs_abonnements.yaml, @tarifs_scolaires.yaml,
@zones.yaml
- Inclus toutes les conditions d'éligibilité éventuellement
mentionnées dans les parameters comme dans @conditions_eligibilite.yaml
- Inclus les éventuelles conditions specifiques éventuellement spécifiées dans
les parameters @conditions_specifiques.yaml
- Les réductions sont appliquées de manière non cumulable (on prend la plus
avantageuse) comme c'est souvent le cas dans les politiques tarifaires des
transports.
TRES IMPORTANT: surtout ne code pas les seuils en dur dans le code,
utiliser tous les barèmes spécifiés comme dans @baremes.yaml,
tu peux en ajouter si besoin

Paramètres yaml:
{text}
"""

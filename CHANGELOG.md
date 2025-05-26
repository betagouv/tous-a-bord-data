#[0.0.4](https://github.com/betagouv/tous-a-bord-data/pull/15)
- Evolution techniques et ajout de fonctionnalité
- Détail :
   - itération sur les étapes de traitement du pipeline avec les LLMs disponibles sur scaleway / ollama / claude
   - amélioration du module de filtrage avec spacy

#[0.0.3](https://github.com/betagouv/tous-a-bord-data/pull/11)
- Evolution techniques et ajout de fonctionnalité
- Détail :
   - amélioration du module de scraping
   - persistance des données collecter sur l'ensemble des AOMs du jeu de données
   - préparation du pipeline de traitement des données scrapées (filtre / nettoyage / mise en forme)
   - intégration des API scaleway / OLLAMA / Claude

#[0.0.2](https://github.com/betagouv/tous-a-bord-data/pull/10)
- Ajout de fonctionnalité
- Détail :
   - création d’une clé de correspondance des données (fuzzy match par le nom des AOMs)
   - mise en place d'un scraper pour collecter les données de tarification et les critères d'éligibilité à la TSST

#[0.0.1](https://github.com/betagouv/tous-a-bord-data/pull/8)
- Evolution techniques et ajout de fonctionnalité
- Détail :
   - ingestion et persistance des données data.gouv.fr des AOMs / communes dans postgresql
   - ingestion et persistance des données PASSIM / CEREMA des url des sites de billétique dans postgresql
   - page streamlit pour documenter et mettre mise à jour des données transport.data.gouv.fr et passim pour les sites de billétique (upload du dernier fichier)
   - documentation du processus de mise à jour


#[0.0.0](https://github.com/betagouv/tous-a-bord-data/pull/4)
- Ajout de fonctionnalité
- Détail :
   - initialisation d'une configuration docker / postgresql / streamlit
   - documentation de l'installation de l'environnement
   - préparation de l'interface streamlit
   - connexion à GRIST et exposition des données existantes TAB dans une page streamlit, avec moteur de recherche
   - connexion à l'API transport.data.gouv et exposition des données AOMs / Communes dans une page streamlit


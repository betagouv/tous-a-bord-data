# Architecture du Projet AOM Data

## Contexte et Objectifs

### Notions Clés

- **AOM (Autorité Organisatrice de la Mobilité)** : Collectivité territoriale chargée de l'organisation des transports publics
- **Sociétés de transport** : Opérateurs délégués par l'AOM
- **Éditeurs de billets** : Sociétés gérant la billettique

### Exemple Type : Île-de-France Mobilités (IDFM)

- **AOM** : IDFM
- **Sociétés de transport** : RATP, SNCF, Transdev, Keolis
- **Billettique** : Système Navigo

### Objectifs du Projet

1. Constituer une base de données consolidée des AOM
2. Collecter et structurer :
   - Informations AOM (SIREN, nom, population)
   - Sociétés de transport associées
   - Tarifs et abonnements
   - Critères d'éligibilité aux tarifs sociaux
   - Codes postaux des zones de chalandise

### Sources de Données

1. **Structurées** :
   - Fichier Excel de base (AOM/SIREN)
   - Fichiers Excel de collecte manuelle (critères/tarifs) sur grist
2. **Non Structurées** :
   - Site Bannatic (données AOM)
   - Site France Mobilité (sociétés de transport)
   - Sites web des transporteurs (tarifs/critères)

## Architecture Technique

### 1. Structure de la Base de Données

```
sql
-- Tables principales
CREATE TABLE aoms (
    siren VARCHAR(9) PRIMARY KEY,
    nom VARCHAR(255),
    population INTEGER,
    forme_juridique VARCHAR(100),
    date_maj TIMESTAMP
);

CREATE TABLE zones_desserte (
    id SERIAL PRIMARY KEY,
    aom_siren VARCHAR(9) REFERENCES aoms(siren),
    code_postal VARCHAR(5)
);

CREATE TABLE societes_transport (
    id SERIAL PRIMARY KEY,
    nom VARCHAR(255),
    site_web TEXT,
    aom_siren VARCHAR(9) REFERENCES aoms(siren)
);

CREATE TABLE editeurs_billets (
    id SERIAL PRIMARY KEY,
    nom VARCHAR(255),
    societe_transport_id INTEGER REFERENCES societes_transport(id)
);

-- Table vectorielle
CREATE TABLE raw_data (
    id SERIAL PRIMARY KEY,
    aom_siren VARCHAR(9) REFERENCES aoms(siren),
    content TEXT,
    embedding vector(1536),
    source_type VARCHAR(50),
    url TEXT,
    date_extraction TIMESTAMP
);
```

### 2. Pipeline de Traitement

```
python
class DataPipeline:
    def __init__(self):
        self.structured_sources = {
            'base_aom': self.process_base_excel,
            'criteres_tarifs': self.process_criteres_excel
        }
        self.unstructured_sources = {
            'bannatic': self.scrape_bannatic,
            'france_mobilite': self.scrape_france_mobilite,
            'transport_sites': self.scrape_transport_sites
        }

    async def process_aom(self, siren):
        """Traitement complet pour une AOM"""
        # 1. Données structurées de base
        base_data = self.process_structured_sources(siren)

        # 2. Enrichissement Bannatic
        bannatic_data = await self.scrape_bannatic(siren)

        # 3. Identification sociétés de transport
        transport_companies = await self.scrape_france_mobilite(siren)

        # 4. Extraction tarifs et critères
        for company in transport_companies:
            tarifs_data = await self.scrape_transport_site(
                company['site_url'],
                self.llm_agent
            )

        # 5. Consolidation et validation
        consolidated = self.consolidate_data(
            base_data,
            bannatic_data,
            transport_companies,
            tarifs_data
        )
        return consolidated
```

### 3. Agent LLM pour le Scraping

```
python
class LLMScrapingAgent:
    def analyze_page(self, html_content):
        """Analyse une page web pour extraire les informations pertinentes"""
        prompt = f"""
        Analyse cette page web et extrait :
        Les tarifs des abonnements
        Les critères d'éligibilité pour chaque tarif
        Les conditions spéciales (étudiants, RSA, etc.)
        Page : {html_content}
        """
        return self.llm.analyze(prompt)

    def validate_extraction(self, data):
        """Vérifie la cohérence des données extraites"""
        pass
```

### 4. Système de Mise à Jour

```
python
class UpdateManager:
    def schedule_updates(self):
        """Planifie les mises à jour périodiques"""
        for source_type in ["bannatic", "france_mobilite", "transport_sites"]:
            self.schedule_source_update(source_type)

    def detect_changes(self, old_data, new_data):
        """Détecte les changements significatifs"""
        pass
```

## Utilisation du RAG

Le RAG (Retrieval Augmented Generation) est utilisé pour :

1. La recherche sémantique dans les pages web
2. L'extraction intelligente d'informations
3. La détection de changements dans les sources
4. La validation des données extraites

Cette approche permet de combiner :

- La précision d'une base de données relationnelle pour les données structurées
- La flexibilité d'une base vectorielle pour les données non structurées
- L'intelligence du LLM pour l'extraction et la validation des données

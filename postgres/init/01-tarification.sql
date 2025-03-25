-- Table pour stocker les informations brutes scrapées
CREATE TABLE IF NOT EXISTS tarification_raw (
    id SERIAL PRIMARY KEY,
    n_siren_aom VARCHAR(20),
    url_source TEXT,
    contenu_scrape TEXT,
    date_scraping TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding vector(1536)  -- Pour stocker les embeddings pour RAG
);

-- Table pour les tarifs structurés
CREATE TABLE IF NOT EXISTS tarifs (
    id SERIAL PRIMARY KEY,
    n_siren_aom VARCHAR(20),
    nom_tarif TEXT,
    description TEXT,
    prix NUMERIC(10,2),
    periode TEXT, -- mensuel, annuel, etc.
    embedding vector(1536)
);

-- Table pour les critères d'éligibilité
CREATE TABLE IF NOT EXISTS criteres_eligibilite (
    id SERIAL PRIMARY KEY,
    tarif_id INTEGER REFERENCES tarifs(id),
    type_critere TEXT, -- age, revenu, statut, etc.
    condition TEXT,
    embedding vector(1536)
); 

CREATE INDEX IF NOT EXISTS idx_tarification_raw_siren ON tarification_raw(n_siren_aom);
CREATE INDEX IF NOT EXISTS idx_tarifs_siren ON tarifs(n_siren_aom);
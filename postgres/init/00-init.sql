CREATE TABLE IF NOT EXISTS aoms (
    id SERIAL PRIMARY KEY,
    n_siren_aom VARCHAR(20),
    nom_aom TEXT,
    commune_principale_aom TEXT,
    n_siren_groupement VARCHAR(20),
    departement TEXT,
    region TEXT,
    forme_juridique_aom TEXT,
    bassin_mobilite TEXT,
    nombre_membre_aom INTEGER,
    nombre_commune_aom INTEGER,
    population_aom INTEGER,
    surface_km_2 NUMERIC(10, 2),
    id_reseau INTEGER,
    nom_president_aom TEXT,
    adresse_siege_aom TEXT,
    adresse_mail TEXT
);

CREATE TABLE IF NOT EXISTS communes (
    id SERIAL PRIMARY KEY,
    nom_membre TEXT,
    siren_membre VARCHAR(20),
    n_insee INTEGER,
    population_totale_2021_insee INTEGER,
    surface_km_2 NUMERIC(10, 2),
    nom_aom TEXT,
    n_siren_aom VARCHAR(20),
    forme_juridique_aom TEXT,
    plan TEXT,
    comite_partenaire TEXT,
    bassin_mobilite_1 TEXT,
    region_siege TEXT,
    departement_siege TEXT,
    nom_groupement TEXT,
    n_siren_groupement VARCHAR(20),
    id_reseau INTEGER,
    nature_juridique_groupement TEXT,
    nombre_membre INTEGER,
    population_totale_2019_banatic INTEGER
);

CREATE TABLE IF NOT EXISTS passim_aoms (
    id SERIAL PRIMARY KEY,
    metadata_title TEXT,
    last_update TIMESTAMP,
    nom_commercial TEXT,
    autorite TEXT,
    exploitant TEXT,
    site_web_principal TEXT,
    type_de_contrat TEXT,
    identifiant_de_reseau_tcu TEXT,
    fiche_transbus TEXT,
    type_d_usagers_tous BOOLEAN,
    type_d_usagers_pmr BOOLEAN,
    type_d_usagers_faibles_revenus BOOLEAN,
    type_d_usagers_recherche_d_emplois BOOLEAN,
    type_d_usagers_soins_medicaux BOOLEAN,
    type_d_usagers_personnes_agees BOOLEAN,
    type_d_usagers_scolaires BOOLEAN,
    type_d_usagers_touristes BOOLEAN,
    niveau TEXT,
    type_de_transport TEXT,
    sous_type_de_transport TEXT,
    mode_de_transport_autocar BOOLEAN,
    mode_de_transport_bateau BOOLEAN,
    mode_de_transport_bus BOOLEAN,
    mode_de_transport_bus_navette BOOLEAN,
    mode_de_transport_funiculaire BOOLEAN,
    mode_de_transport_metro BOOLEAN,
    mode_de_transport_taxi BOOLEAN,
    mode_de_transport_telepherique BOOLEAN,
    mode_de_transport_train BOOLEAN,
    mode_de_transport_tramway BOOLEAN,
    notes TEXT,
    territoires_concernes TEXT
);

-- Create an index on the columns frequently used for joins
CREATE INDEX IF NOT EXISTS idx_aoms_n_siren ON aoms(n_siren_aom);
CREATE INDEX IF NOT EXISTS idx_communes_n_siren_aom ON communes(siren_membre);
CREATE INDEX IF NOT EXISTS idx_passim_siren_aom ON passim_aoms(_id);

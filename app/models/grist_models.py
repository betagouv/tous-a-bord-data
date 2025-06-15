from typing import Optional

from pydantic import BaseModel


class aom(BaseModel):
    n_siren_aom: int
    nom_aom: str
    commune_principale_aom: str
    n_siren_groupement: Optional[int]
    departement: str
    region: str
    forme_juridique_aom: str
    bassin_mobilite: str
    nombre_membre_aom: int
    nombre_commune_aom: int
    population_aom: Optional[int]
    surface_km_2: str
    id_reseau: Optional[int]
    nom_president_aom: Optional[str]
    adresse_siege_aom: Optional[str]
    adresse_mail: Optional[str]


class commune(BaseModel):
    nom_membre: str
    siren_membre: int
    n_insee: str
    population_totale_2021_insee: int
    surface_km_2: float
    nom_aom: str
    n_siren_aom: Optional[int] = None
    forme_juridique_aom: str
    plan: Optional[str]
    comite_partenaire: Optional[str]
    bassin_mobilite_1: Optional[str]
    region_siege: str
    departement_siege: str
    nom_groupement: Optional[str]
    n_siren_groupement: Optional[int] = None
    id_reseau: Optional[int] = None
    nature_juridique_groupement: Optional[str]
    nombre_membre: Optional[int] = None
    population_totale_2019_banatic: Optional[int] = None

from datetime import datetime
from typing import Any, ClassVar, Dict, Optional, Set

from pydantic import BaseModel, Field, root_validator, validator


class Aom(BaseModel):
    n_siren_aom: int
    nom_aom: str
    commune_principale_aom: str
    n_siren_groupement: int
    departement: str
    region: str
    forme_juridique_aom: str
    bassin_mobilite: Optional[str] = None
    nombre_membre_aom: int
    nombre_commune_aom: int
    population_aom: Optional[int] = None
    surface_km_2: Optional[float] = None

    @validator("surface_km_2", pre=True)
    def parse_surface(cls, value):
        if value is None:
            return None
        if isinstance(value, float):
            return value
        try:
            # Gère les cas comme "123,45" (virgule française)
            if isinstance(value, str) and "," in value:
                value = value.replace(",", ".")
            return float(value)
        except (ValueError, TypeError):
            return None

    id_reseau: int
    nom_president_aom: Optional[str] = None
    adresse_siege_aom: Optional[str] = None
    adresse_mail: Optional[str] = None


class Commune(BaseModel):
    nom_membre: Optional[str] = None
    siren_membre: Optional[int] = None
    n_insee: Optional[int] = None
    population_totale_2021_insee: Optional[int] = None
    surface_km_2: Optional[float] = None

    @validator("surface_km_2", pre=True)
    def parse_surface(cls, value):
        if value is None:
            return None
        if isinstance(value, float):
            return value
        try:
            # Gère les cas comme "123,45" (virgule française)
            if isinstance(value, str) and "," in value:
                value = value.replace(",", ".")
            return float(value)
        except (ValueError, TypeError):
            return None

    nom_aom: Optional[str] = None

    @validator("nom_aom", pre=True)
    def parse_nom_aom(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        try:
            if isinstance(value, str) and "-" in value:
                return None
        except (ValueError, TypeError):
            return None

    n_siren_aom: Optional[int] = None

    @validator("n_siren_aom", pre=True)
    def parse_n_siren_aom(cls, value):
        if value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            if isinstance(value, str) and "-" in value:
                return None
            else:
                return int(value)
        except (ValueError, TypeError):
            return None

    forme_juridique_aom: Optional[str] = None
    plan: Optional[str] = None
    comite_partenaire: Optional[str] = None
    bassin_mobilite_1: Optional[str] = None
    region_siege: Optional[str] = None
    departement_siege: Optional[str] = None
    nom_groupement: Optional[str] = None
    n_siren_groupement: Optional[int] = None

    @validator("n_siren_groupement", pre=True)
    def parse_n_siren_groupement(cls, value):
        if value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            if isinstance(value, str) and "-" in value:
                return None
            else:
                return int(value)
        except (ValueError, TypeError):
            return None

    id_reseau: Optional[int] = None
    nature_juridique_groupement: Optional[str] = None
    nombre_membre: Optional[int] = None
    population_totale_2019_banatic: Optional[int] = None


class TransportOffer(BaseModel):
    metadata_title: str
    _id: str
    last_update: datetime
    nom_commercial: str
    autorite: Optional[str] = None
    exploitant: Optional[str] = None
    site_web_principal: Optional[str] = None
    type_de_contrat: Optional[str] = None
    identifiant_de_reseau_tcu: Optional[int] = None
    fiche_transbus_tc: Optional[str] = None
    fiche_wikipedia: Optional[str] = None

    # Type d'usagers (boolean flags)
    type_d_usagers_tous: bool = Field(default=False)
    type_d_usagers_pmr: bool = Field(default=False)
    type_d_usagers_faibles_revenus: bool = Field(default=False)
    type_d_usagers_recherche_emplois: bool = Field(default=False)
    type_d_usagers_soins_medicaux: bool = Field(default=False)
    type_d_usagers_personnes_agees: bool = Field(default=False)
    type_d_usagers_scolaires: bool = Field(default=False)
    type_d_usagers_touristes: bool = Field(default=False)

    # Classification
    niveau: Optional[str] = None
    type_de_transport: Optional[str] = None
    sous_type_de_transport: Optional[str] = None

    # Modes de transport (boolean flags)
    mode_de_transport_autocar: bool = Field(default=False)
    mode_de_transport_avion: bool = Field(default=False)
    mode_de_transport_bateau: bool = Field(default=False)
    mode_de_transport_bus: bool = Field(default=False)
    mode_de_transport_bus_navette: bool = Field(default=False)
    mode_de_transport_deux_roues_motorises: bool = Field(default=False)
    mode_de_transport_funiculaire: bool = Field(default=False)
    mode_de_transport_marche: bool = Field(default=False)
    mode_de_transport_metro: bool = Field(default=False)
    mode_de_transport_moto: bool = Field(default=False)
    mode_de_transport_poids_lourd: bool = Field(default=False)
    mode_de_transport_taxi: bool = Field(default=False)
    mode_de_transport_telepherique: bool = Field(default=False)
    mode_de_transport_train: bool = Field(default=False)
    mode_de_transport_tramway: bool = Field(default=False)
    mode_de_transport_trottinette: bool = Field(default=False)
    mode_de_transport_velo: bool = Field(default=False)
    mode_de_transport_voiture: bool = Field(default=False)

    # Liste des champs booléens pour le validateur
    _boolean_fields: ClassVar[Set[str]] = {
        "type_d_usagers_tous",
        "type_d_usagers_pmr",
        "type_d_usagers_faibles_revenus",
        "type_d_usagers_recherche_emplois",
        "type_d_usagers_soins_medicaux",
        "type_d_usagers_personnes_agees",
        "type_d_usagers_scolaires",
        "type_d_usagers_touristes",
        "mode_de_transport_autocar",
        "mode_de_transport_avion",
        "mode_de_transport_bateau",
        "mode_de_transport_bus",
        "mode_de_transport_bus_navette",
        "mode_de_transport_deux_roues_motorises",
        "mode_de_transport_funiculaire",
        "mode_de_transport_marche",
        "mode_de_transport_metro",
        "mode_de_transport_moto",
        "mode_de_transport_poids_lourd",
        "mode_de_transport_taxi",
        "mode_de_transport_telepherique",
        "mode_de_transport_train",
        "mode_de_transport_tramway",
        "mode_de_transport_trottinette",
        "mode_de_transport_velo",
        "mode_de_transport_voiture",
    }

    @root_validator(pre=True)
    def map_grist_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map fields from Grist format to model format and convert boolean values
        """
        result = dict(values)  # Create a copy to avoid modifying the original

        # Process boolean fields from Grist format
        for grist_field, model_field in {
            "type_d_usagers___tous": "type_d_usagers_tous",
            "type_d_usagers___PMR": "type_d_usagers_pmr",
            "type_d_usagers___faibles_revenus": "type_d_usagers_faibles_revenus",
            "type_d_usagers___recherche_d_emplois": "type_d_usagers_recherche_emplois",
            "type_d_usagers___soins_medicaux": "type_d_usagers_soins_medicaux",
            "type_d_usagers___personnes_agees": "type_d_usagers_personnes_agees",
            "type_d_usagers___scolaires": "type_d_usagers_scolaires",
            "type_d_usagers___touristes": "type_d_usagers_touristes",
            "mode_de_transport___Autocar": "mode_de_transport_autocar",
            "mode_de_transport___Avion": "mode_de_transport_avion",
            "mode_de_transport___Bateau": "mode_de_transport_bateau",
            "mode_de_transport___Bus": "mode_de_transport_bus",
            "mode_de_transport___Bus_navette": "mode_de_transport_bus_navette",
            "mode_de_transport___Deux_roues_motorises": "mode_de_transport_deux_roues_motorises",
            "mode_de_transport___Funiculaire": "mode_de_transport_funiculaire",
            "mode_de_transport___Marche": "mode_de_transport_marche",
            "mode_de_transport___Metro": "mode_de_transport_metro",
            "mode_de_transport___Moto": "mode_de_transport_moto",
            "mode_de_transport___Poids_lourd": "mode_de_transport_poids_lourd",
            "mode_de_transport___Taxi": "mode_de_transport_taxi",
            "mode_de_transport___Telepherique": "mode_de_transport_telepherique",
            "mode_de_transport___Train": "mode_de_transport_train",
            "mode_de_transport___Tramway": "mode_de_transport_tramway",
            "mode_de_transport___Trottinette": "mode_de_transport_trottinette",
            "mode_de_transport___Velo": "mode_de_transport_velo",
            "mode_de_transport___Voiture": "mode_de_transport_voiture",
        }.items():
            if grist_field in values:
                value = values[grist_field]
                # Convert to boolean
                if value is None:
                    bool_value = False
                elif isinstance(value, bool):
                    bool_value = value
                elif isinstance(value, (int, float)):
                    # 0 -> False, tout autre nombre -> True
                    bool_value = bool(value)
                elif isinstance(value, str):
                    bool_value = value.lower() in (
                        "true",
                        "yes",
                        "y",
                        "1",
                        "oui",
                        "vrai",
                    )
                else:
                    bool_value = bool(value)  # Conversion par défaut
                result[model_field] = bool_value

        return result

    # Additional information
    notes: Optional[str] = None
    territoire_s_concerne_s: str

    class Config:
        # Allow the use of field names with special characters in the original data
        populate_by_name = True

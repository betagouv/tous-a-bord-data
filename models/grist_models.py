from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Set

from pydantic import BaseModel, Field, root_validator, validator


class DownloadedAom(BaseModel):
    n_sirenaom: Optional[int] = None
    nom_aom: Optional[str] = None
    commune_principale_aom: Optional[str] = None
    n_siren_groupement: Optional[int] = None
    departement: Optional[str] = None
    region: Optional[str] = None
    code_insee_region: Optional[int] = None

    @validator("code_insee_region", pre=True)
    def parse_code_insee_region(cls, value):
        if value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            # Gère les cas comme "12345" (code INSEE)
            return int(value)
        except (ValueError, TypeError):
            return None

    forme_juridique_aom: Optional[str] = None
    bassin_mobilite: Optional[str] = None
    nombre_membre_aom: Optional[int] = None
    nombre_commune_aom: Optional[int] = None
    population_aom_banatic: Optional[int] = None
    surface_km_2: Optional[float] = None
    lienbanatic: Optional[str] = None
    id_reseau: Optional[int] = None
    nom_president_aom: Optional[str] = None
    adresse_siege_aom: Optional[str] = None
    adresse_mail: Optional[str] = None
    offre_territoire_aom: Optional[str] = None
    vm_taux_max: Optional[float] = None

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

    def to_aom(self) -> "Aom":
        """
        Convert to Aom model
        """
        return Aom(
            n_siren_aom=self.n_sirenaom,
            nom_aom=self.nom_aom,
            commune_principale_aom=self.commune_principale_aom,
            n_siren_groupement=self.n_siren_groupement,
            departement=self.departement,
            region=self.region,
            forme_juridique_aom=self.forme_juridique_aom,
            bassin_mobilite=self.bassin_mobilite,
            nombre_membre_aom=self.nombre_membre_aom,
            nombre_commune_aom=self.nombre_commune_aom,
            population_aom=self.population_aom_banatic,
            surface_km_2=self.surface_km_2,
            id_reseau=self.id_reseau,
            nom_president_aom=self.nom_president_aom,
            adresse_siege_aom=self.adresse_siege_aom,
            adresse_mail=self.adresse_mail,
        )


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

    @validator("population_aom", pre=True)
    def parse_population(cls, value):
        if value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            if isinstance(value, str):
                return int(value)
        except (ValueError, TypeError):
            return None

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


class ComarquageTransportOffer(BaseModel):
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


class ComarquageAom(BaseModel):
    metadata_title: str
    id2: str
    last_update: datetime
    nom_de_l_aom: str
    commune_principale: str
    departement: str
    region: str
    forme_juridique: str
    nombre_de_membres: Optional[int] = None
    nombre_de_communes_du_rt: Optional[int] = None
    population_de_l_aom: Optional[int] = None
    surface_km2: Optional[float] = None
    site_de_l_aom: Optional[str] = None
    lien_base_banatic: Optional[str] = None
    ndeg_siren: Optional[int] = None
    id_reseau: Optional[int] = None
    nom_du_president: Optional[str] = None
    adresse_du_siege: Optional[str] = None
    adresse_mail: Optional[str] = None
    offres_sur_le_territoire_de_l_aom: Optional[str] = None
    comite_des_partenaires: Optional[str] = None
    lien_vers_la_deliberation_d_installation_du_comite_des_partenaires: Optional[
        str
    ] = None
    plan_de_mobilite_pdu_ou_pdm: Optional[str] = None
    plan_de_mobilite_simplifie_ou_assimile: Optional[str] = None
    annee_d_approbation_du_pdms_ou_assimile: Optional[int] = None
    bassins_de_mobilite_sur_le_rt_de_l_aom: Optional[str] = None
    territoire_s_concerne_s: Optional[str] = None
    taux_de_versement_mobilite_moyen: Optional[str] = None

    @validator(
        "nombre_de_membres",
        "nombre_de_communes_du_rt",
        "population_de_l_aom",
        "ndeg_siren",
        "id_reseau",
        "annee_d_approbation_du_pdms_ou_assimile",
        pre=True,
    )
    def parse_numeric(cls, value):
        if value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @validator("surface_km2", pre=True)
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


class AomTransportOffer(BaseModel):
    # Informations AOM
    n_siren_groupement: int
    n_siren_aom: int
    nom_aom: str
    commune_principale_aom: str
    nombre_commune_aom: int
    population_aom: Optional[int] = None
    surface_km_2: Optional[float] = None
    id_reseau_aom: Optional[int] = None

    # Informations offre de transport
    nom_commercial: Optional[str] = None
    exploitant: Optional[str] = None
    site_web_principal: Optional[str] = None
    territoire_s_concerne_s: Optional[str] = None
    type_de_contrat: Optional[str] = None

    # Validateurs similaires à ceux des autres modèles pour les champs numériques
    @validator(
        "population_aom",
        "nombre_commune_aom",
        "n_siren_aom",
        "n_siren_groupement",
        "id_reseau_aom",
        pre=True,
    )
    def parse_numeric(cls, value):
        if value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

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


class AomTags(BaseModel):
    """
    Extension du modèle AomTransportOffer avec des champs supplémentaires
    pour les tags, fournisseurs et statuts.
    """

    n_siren_groupement: int
    n_siren_aom: int
    nom_aom: str
    commune_principale_aom: str
    nombre_commune_aom: int
    population_aom: Optional[int] = None
    surface_km_2: Optional[float] = None
    id_reseau_aom: Optional[int] = None

    # Informations offre de transport
    nom_commercial: Optional[str] = None
    exploitant: Optional[str] = None
    site_web_principal: Optional[str] = None
    territoire_s_concerne_s: Optional[str] = None
    type_de_contrat: Optional[str] = None

    # Validateurs similaires à ceux des autres modèles pour les champs numériques
    @validator(
        "population_aom",
        "nombre_commune_aom",
        "n_siren_aom",
        "n_siren_groupement",
        "id_reseau_aom",
        pre=True,
    )
    def parse_numeric(cls, value):
        if value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

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

    criteres_eligibilite: Optional[List[str]] = None
    fournisseurs: Optional[List[str]] = None

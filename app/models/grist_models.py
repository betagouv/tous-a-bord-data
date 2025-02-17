from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AOM:
    N_SIREN_AOM: int
    Nom_de_l_AOM: str
    Commune_principale_De_l_AOM: str
    N_SIREN_Groupement: Optional[int]
    Departement: str
    Region: str
    Forme_juridique_De_l_AOM: str
    Bassin_de_mobilite: str
    Nombre_de_membre_De_l_AOM: int
    Nombre_de_commune_De_l_AOM: int
    Population_De_l_AOM: int
    Surface_km2_: float
    Lien_Banatic: Optional[str]
    Id_reseau: Optional[int]
    Nom_de_president_De_l_AOM: Optional[str]
    Adresse_du_siege_De_l_AOM: Optional[str]
    Adresse_mail: Optional[str]
    Offres_sur_le_territoire_de_l_AOM: Optional[str]
    Offres_sur_le_territoire_de_l_AOM2: Optional[str]
    Description_des_tarifications: List[str]
    Tarif_etudiant: bool
    Page_tarification: Optional[str]
    QF_CAF_: bool
    GART: bool


@dataclass
class Commune:
    Nom_membre: str
    Siren_membre: int
    N_INSEE: str
    Population_totale_2019_Banatic_: int
    Surface_km2_: float
    Lien_Page_Wikipedia: Optional[str]
    Nom_de_l_AOM: str
    N_SIREN_AOM: int
    Forme_juridique_De_l_AOM: str
    Plan: Optional[str]
    Comite_des_Partenaires: Optional[str]
    Bassin_de_mobilite_1: Optional[str]
    Bassin_de_mobilite_2: Optional[str]
    Region_siege: str
    Departement_siege: str
    Nom_du_groupement: Optional[str]
    N_SIREN_Groupement: Optional[int]
    Id_reseau: Optional[int]
    Nature_juridique_Du_groupement: Optional[str]
    Lien_Banatic: Optional[str]
    Nombre_De_membres: Optional[int]
    Population_totale_2019_Banatic_2: Optional[int]

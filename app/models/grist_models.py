from typing import List, Optional

from pydantic import BaseModel


class AOM(BaseModel):
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
    Description_des_tarifications: Optional[List[str]] = None
    Tarif_etudiant: bool
    Page_tarification: Optional[str]
    QF_CAF_: bool
    GART: bool

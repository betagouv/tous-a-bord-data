import logging
from typing import List, Optional

import requests
from models.grist_models import AOM, Commune


class GristDataService:
    def __init__(self, api_key: str, doc_id: str):
        if not api_key:
            raise ValueError("La clé API Grist est requise")

        self.base_url = "https://grist.numerique.gouv.fr/o/tous-a-bord"
        self.doc_id = "jn4Z4deNRbM9MyGBpCK5Jk"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def get_aoms(self) -> List[AOM]:
        """Récupère toutes les AOM depuis Grist."""
        try:
            base = f"{self.base_url}/api/docs/{self.doc_id}"
            url = f"{base}/tables/AOM/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            # Debug pour voir la structure
            data = response.json()
            # Conversion en objets AOM
            # La structure est {'id': X, 'fields': {...}}
            # Si data est une liste de records
            if isinstance(data, list):
                return [
                    AOM.model_validate(record["fields"]) for record in data
                ]
            # Si data est un dict avec une clé 'records'
            elif isinstance(data, dict) and "records" in data:
                return [
                    AOM.model_validate(record["fields"])
                    for record in data["records"]
                ]
            else:
                raise ValueError(f"Format de données inattendu: {type(data)}")

        except Exception as e:
            logging.error(f"Erreur lors de la récupération des AOM: {e}")
            # logging.error(f"Response text: {response.text}")
            raise

    async def get_communes(self) -> List[Commune]:
        """Récupère toutes les communes depuis Grist."""
        try:
            base = f"{self.base_url}/api/docs/{self.doc_id}"
            url = f"{base}/tables/Communes/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            response.raise_for_status()
            # Debug pour voir la structure
            print("Response raw:", response.text)
            data = response.json()
            d = data["records"]
            print("Response JSON:", data)
            # Si data est une liste de records
            if isinstance(data, list):
                return [Commune.model_validate(rec["fields"]) for rec in data]
            # Si data est un dict avec une clé 'records'
            elif isinstance(data, dict) and "records" in data:
                return [Commune.model_validate(rec["fields"]) for rec in d]
            else:
                raise ValueError(f"Format inattendu: {type(data)}")

        except Exception as e:
            logging.error(f"Erreur lors de la récupération des communes: {e}")
            logging.error(f"Response text: {response.text}")
            raise

    async def get_aom_by_siren(self, siren: int) -> Optional[AOM]:
        """Récupère une AOM spécifique par son SIREN."""
        aoms = await self.get_aoms()
        return next((aom for aom in aoms if aom.N_SIREN_AOM == siren), None)

    async def get_communes_by_aom(self, siren_aom: int) -> List[Commune]:
        """Récupère toutes les communes d'une AOM."""
        communes = await self.get_communes()
        return [c for c in communes if c.N_SIREN_AOM == siren_aom]

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
            url = f"{self.base_url}/api/docs/{self.doc_id}/tables/AOM/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            # Debug: afficher les données brutes
            # print("Response raw:", response.text)
            data = response.json()
            # print("Response JSON:", data)
            # records = self._process_grist_response(data)
            # print("Processed records:", records)

            # Retourner les records bruts sans typage pour debug
            return data
            # Commenté temporairement pour debug:
            # return [AOM(**record) for record in records]
        except requests.exceptions.JSONDecodeError as e:
            logging.error(f"Erreur de décodage JSON: {e}")
            logging.error(f"URL: {url}")
            # logging.error(f"Réponse: {response.text}")
            raise
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des AOM: {e}")
            raise

    async def get_communes(self) -> List[Commune]:
        """Récupère toutes les communes depuis Grist."""
        try:
            base = f"{self.base_url}/api/docs/{self.doc_id}"
            url = f"{base}/tables/Communes/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            records = response.json()
            return [Commune(**record) for record in records]
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des communes: {e}")
            raise

    async def get_aom_by_siren(self, siren: int) -> Optional[AOM]:
        """Récupère une AOM spécifique par son SIREN."""
        aoms = await self.get_aoms()
        return next((aom for aom in aoms if aom.N_SIREN_AOM == siren), None)

    async def get_communes_by_aom(self, siren_aom: int) -> List[Commune]:
        """Récupère toutes les communes d'une AOM."""
        communes = await self.get_communes()
        return [c for c in communes if c.N_SIREN_AOM == siren_aom]

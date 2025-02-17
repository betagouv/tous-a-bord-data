import logging
from typing import List, Optional

import requests
from grist_api import GristDocAPI
from models.grist_models import AOM, Commune


class GristDataService:
    def __init__(self, api_key: str, doc_id: str):
        if not api_key:
            raise ValueError("La clé API Grist est requise")

        base_url = "https://grist.numerique.gouv.fr/o/tous-a-bord"
        self.api = GristDocAPI(
            "jn4Z4deNRbM9",
            server=base_url,
            api_key=api_key,
        )

    async def get_aoms(self) -> List[AOM]:
        """Récupère toutes les AOM depuis Grist."""
        try:
            records = self.api.fetch_table("AOM")
            return [AOM(**record) for record in records]
        except requests.exceptions.JSONDecodeError as e:
            logging.error(f"Erreur de décodage JSON: {e}")
            logging.error(f"URL: {self.api._server}")
            response_text = (
                e.response.text if hasattr(e, "response") else "Pas de réponse"
            )
            logging.error(f"Réponse: {response_text}")
            raise
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des AOM: {e}")
            raise

    async def get_communes(self) -> List[Commune]:
        """Récupère toutes les communes depuis Grist."""
        records = self.api.fetch_table("Communes")
        return [Commune(**record) for record in records]

    async def get_aom_by_siren(self, siren: int) -> Optional[AOM]:
        """Récupère une AOM spécifique par son SIREN."""
        aoms = await self.get_aoms()
        return next((aom for aom in aoms if aom.N_SIREN_AOM == siren), None)

    async def get_communes_by_aom(self, siren_aom: int) -> List[Commune]:
        """Récupère toutes les communes d'une AOM."""
        communes = await self.get_communes()
        return [c for c in communes if c.N_SIREN_AOM == siren_aom]

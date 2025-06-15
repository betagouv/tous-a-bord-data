import logging
from typing import List, Optional

import requests
from models.grist_models import aom, commune


class GristDataService:
    _instance: Optional["GristDataService"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GristDataService, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_key: str, doc_id: str):
        # Only initialize once
        if not GristDataService._initialized:
            if not api_key:
                raise ValueError("La clé API Grist est requise")

            self.base_url = "https://grist.numerique.gouv.fr/o/droits-data"
            self.doc_id = doc_id
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            GristDataService._initialized = True

    @classmethod
    def get_instance(cls, api_key: str, doc_id: str) -> "GristDataService":
        """
        Get the singleton instance of GristDataService
        """
        if cls._instance is None:
            cls._instance = cls(api_key, doc_id)
        return cls._instance

    async def get_aoms(self) -> List[aom]:
        """
        Retrieve aoms from Grist
        """
        try:
            base = f"{self.base_url}/api/docs/{self.doc_id}"
            url = f"{base}/tables/Aoms/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            # validate data structure
            if isinstance(data, list):
                return [
                    aom.model_validate(record["fields"]) for record in data
                ]
            elif isinstance(data, dict) and "records" in data:
                return [
                    aom.model_validate(record["fields"])
                    for record in data["records"]
                ]
            else:
                raise ValueError(f"Format de données inattendu: {type(data)}")
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des aoms: {e}")
            raise

    async def get_communes(self) -> List[commune]:
        """
        Retrieve communes from Grist
        """
        try:
            base = f"{self.base_url}/api/docs/{self.doc_id}"
            url = f"{base}/tables/Communes/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            # validate data structure
            if isinstance(data, list):
                return [commune.model_validate(rec["fields"]) for rec in data]
            elif isinstance(data, dict) and "records" in data:
                return [
                    commune.model_validate(rec["fields"])
                    for rec in data["records"]
                ]
            else:
                raise ValueError(f"Format inattendu: {type(data)}")

        except Exception as e:
            logging.error(f"Erreur lors de la récupération des communes: {e}")
            raise

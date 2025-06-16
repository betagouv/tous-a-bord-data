import logging
from typing import Dict, List, Optional, Union

import requests
from models.grist_models import Aom, Commune, TransportOffer


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

    async def get_aoms(self) -> List[Aom]:
        """
        Retrieve Aoms from Grist
        """
        try:
            base = f"{self.base_url}/api/docs/{self.doc_id}"
            url = f"{base}/tables/Aoms/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return [
                    Aom.model_validate(record["fields"]) for record in data
                ]
            elif isinstance(data, dict) and "records" in data:
                return [
                    Aom.model_validate(record["fields"])
                    for record in data["records"]
                ]
            else:
                raise ValueError(f"Format de données inattendu: {type(data)}")
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des aoms: {e}")
            raise

    async def get_communes(self) -> List[Commune]:
        """
        Retrieve communes from Grist
        """
        try:
            base = f"{self.base_url}/api/docs/{self.doc_id}"
            url = f"{base}/tables/Communes/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                return [
                    Commune.model_validate(record["fields"]) for record in data
                ]
            elif isinstance(data, dict) and "records" in data:
                return [
                    Commune.model_validate(record["fields"])
                    for record in data["records"]
                ]
            else:
                raise ValueError(f"Format de données inattendu: {type(data)}")

        except Exception as e:
            logging.error(
                f"Erreur lors de la récupération des offres de transport: {e}"
            )
            raise

    async def update_aoms(
        self, aoms: List[Aom]
    ) -> Dict[str, Union[int, List[Dict]]]:
        """
        Update Aoms in Grist using n_siren_aom as identifier

        Args:
            aoms: List of Aom objects to update

        Returns:
            Dict containing the response from Grist API
        """
        try:
            base = f"{self.base_url}/api/docs/{self.doc_id}"
            url = f"{base}/tables/Aoms/records"

            # Format the records for the API request
            records = []
            for aom in aoms:
                aom_dict = aom.model_dump()
                # Use n_siren_aom as the identifier in the require object
                record = {
                    "require": {"n_siren_aom": aom_dict.pop("n_siren_aom")},
                    "fields": aom_dict,
                }
                records.append(record)

            payload = {"records": records}

            # Make the PUT request to update the records
            response = requests.put(url, headers=self.headers, json=payload)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour des aoms: {e}")
            raise

    async def get_transport_offers(self) -> List[TransportOffer]:
        """
        Retrieve transport offers from Grist
        """
        try:
            base = f"{self.base_url}/api/docs/{self.doc_id}"
            url = f"{base}/tables/Comarquage_offretransport/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                return [
                    TransportOffer.model_validate(record["fields"])
                    for record in data
                ]
            elif isinstance(data, dict) and "records" in data:
                return [
                    TransportOffer.model_validate(record["fields"])
                    for record in data["records"]
                ]
            else:
                raise ValueError(f"Format de données inattendu: {type(data)}")

        except Exception as e:
            logging.error(
                f"Erreur lors de la récupération des offres de transport: {e}"
            )
            raise

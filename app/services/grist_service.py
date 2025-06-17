import logging
from typing import Dict, List, Optional, Union

import requests
import streamlit as st
from models.grist_models import (
    Aom,
    AomTransportOffer,
    ComarquageAom,
    ComarquageTransportOffer,
)


class GristDataService:
    _instance: Optional["GristDataService"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GristDataService, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_key: str):
        # Only initialize once
        if not GristDataService._initialized:
            if not api_key:
                raise ValueError("La clé API Grist est requise")

            self.base_url = "https://grist.numerique.gouv.fr/o/droits-data"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            GristDataService._initialized = True

    @classmethod
    def get_instance(cls, api_key: str) -> "GristDataService":
        """
        Get the singleton instance of GristDataService
        """
        if cls._instance is None:
            cls._instance = cls(api_key)
        return cls._instance

    async def get_aoms(self, doc_id: str) -> List[Aom]:
        """
        Retrieve Aoms from Grist

        Args:
            doc_id: The Grist document ID to use for this operation
        """
        try:
            base = f"{self.base_url}/api/docs/{doc_id}"
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

    async def delete_aoms(
        self, aoms: List[Aom], doc_id: str
    ) -> Dict[str, Union[int, List[Dict]]]:
        """
        Delete AOMs in Grist that are not in the provided list

        Args:
            aoms: List of Aom objects to keep (all others will be deleted)
            doc_id: The Grist document ID to use for this operation

        Returns:
            Dict containing the response from Grist API
        """
        try:
            base = f"{self.base_url}/api/docs/{doc_id}"

            # First, get all current records to determine which ones to delete
            get_url = f"{base}/tables/Aoms/records"
            response = requests.get(get_url, headers=self.headers)
            response.raise_for_status()
            all_records = response.json()

            # Create a set of unique identifiers for the AOMs we want to keep
            # Using n_siren_groupement as identifier
            keep_identifiers = {aom.n_siren_groupement for aom in aoms}

            # Find records to delete (those not in our keep list)
            rows_to_delete = []
            if isinstance(all_records, list):
                records = all_records
            elif isinstance(all_records, dict) and "records" in all_records:
                records = all_records["records"]
            else:
                raise ValueError(
                    f"Format de données inattendu: {type(all_records)}"
                )

            for record in records:
                record_id = record.get("id")
                fields = record.get("fields", {})
                n_siren_groupement = fields.get("n_siren_groupement")

                # If this record's identifier is not in our keep list, mark it for deletion
                if n_siren_groupement not in keep_identifiers:
                    rows_to_delete.append(record_id)

            # If there are rows to delete, send the delete request
            if rows_to_delete:
                delete_url = f"{base}/tables/Aoms/data/delete"
                response = requests.post(
                    delete_url, headers=self.headers, json=rows_to_delete
                )
                response.raise_for_status()
                return {
                    "deleted": len(rows_to_delete),
                    "rowIds": rows_to_delete,
                }
            else:
                return {"deleted": 0, "message": "No rows to delete"}

        except Exception as e:
            logging.error(f"Erreur lors de la suppression des AOMs: {e}")
            raise

    async def update_aoms(
        self, aoms: List[Aom], doc_id: str
    ) -> Dict[str, Union[int, List[Dict]]]:
        """
        Update Aoms in Grist using n_siren_aom as identifier

        Args:
            aoms: List of Aom objects to update
            doc_id: The Grist document ID to use for this operation

        Returns:
            Dict containing the response from Grist API
        """
        try:
            base = f"{self.base_url}/api/docs/{doc_id}"
            url = f"{base}/tables/Aoms/records"

            # Format the records for the API request
            records = []
            for aom in aoms:
                aom_dict = aom.model_dump()
                # Use n_siren_aom as the identifier in the require object
                record = {
                    "require": {
                        "n_siren_groupement": aom_dict.pop(
                            "n_siren_groupement"
                        )
                    },
                    "fields": aom_dict,
                }
                records.append(record)

            payload = {"records": records}

            # Make the PUT request to update the records
            response = requests.put(url, headers=self.headers, json=payload)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            st.error(e)
            logging.error(f"Erreur lors de la mise à jour des aoms: {e}")
            raise

    async def get_comarquage_transport_offers(
        self, doc_id: str
    ) -> List[ComarquageTransportOffer]:
        """
        Retrieve transport offers from Grist

        Args:
            doc_id: The Grist document ID to use for this operation
        """
        try:
            base = f"{self.base_url}/api/docs/{doc_id}"
            url = f"{base}/tables/Comarquage_offretransport/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                return [
                    ComarquageTransportOffer.model_validate(record["fields"])
                    for record in data
                ]
            elif isinstance(data, dict) and "records" in data:
                return [
                    ComarquageTransportOffer.model_validate(record["fields"])
                    for record in data["records"]
                ]
            else:
                raise ValueError(f"Format de données inattendu: {type(data)}")

        except Exception as e:
            logging.error(
                f"Erreur lors de la récupération des offres de transport: {e}"
            )
            raise

    async def get_comarquage_aoms(self, doc_id: str) -> List[ComarquageAom]:
        """
        Retrieve passim aoms from Grist

        Args:
            doc_id: The Grist document ID to use for this operation
        """
        try:
            base = f"{self.base_url}/api/docs/{doc_id}"
            url = f"{base}/tables/Comarquage_aom/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                return [
                    ComarquageAom.model_validate(record["fields"])
                    for record in data
                ]
            elif isinstance(data, dict) and "records" in data:
                return [
                    ComarquageAom.model_validate(record["fields"])
                    for record in data["records"]
                ]
            else:
                raise ValueError(f"Format de données inattendu: {type(data)}")

        except Exception as e:
            logging.error(
                f"Erreur lors de la récupération des offres de transport: {e}"
            )
            raise

    async def get_aom_transport_offers(
        self, doc_id: str
    ) -> List[AomTransportOffer]:
        """
        Retrieve AomTransportOffer from Grist

        Args:
            doc_id: The Grist document ID to use for this operation
        """
        try:
            base = f"{self.base_url}/api/docs/{doc_id}"
            url = f"{base}/tables/AomTransportOffers/records"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return [
                    AomTransportOffer.model_validate(record["fields"])
                    for record in data
                ]
            elif isinstance(data, dict) and "records" in data:
                return [
                    AomTransportOffer.model_validate(record["fields"])
                    for record in data["records"]
                ]
            else:
                raise ValueError(f"Format de données inattendu: {type(data)}")
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des aoms: {e}")
            raise

    async def delete_aom_transport_offers(
        self, offers: List[AomTransportOffer], doc_id: str
    ) -> Dict[str, Union[int, List[Dict]]]:
        """
        Delete AOM transport offers in Grist

        Args:
            offers: List of AomTransportOffer objects to keep (all others will be deleted)
            doc_id: The Grist document ID to use for this operation

        Returns:
            Dict containing the response from Grist API
        """
        try:
            base = f"{self.base_url}/api/docs/{doc_id}"

            # First, get all current records to determine which ones to delete
            get_url = f"{base}/tables/AomTransportOffers/records"
            response = requests.get(get_url, headers=self.headers)
            response.raise_for_status()
            all_records = response.json()

            # Create a set of unique identifiers for the offers we want to keep
            # Using a combination of n_siren_groupement and site_web_principal as identifiers
            keep_identifiers = {
                (offer.n_siren_groupement, offer.site_web_principal)
                for offer in offers
            }

            # Find records to delete (those not in our keep list)
            rows_to_delete = []
            if isinstance(all_records, list):
                records = all_records
            elif isinstance(all_records, dict) and "records" in all_records:
                records = all_records["records"]
            else:
                raise ValueError(
                    f"Format de données inattendu: {type(all_records)}"
                )

            for record in records:
                record_id = record.get("id")
                fields = record.get("fields", {})
                n_siren_groupement = fields.get("n_siren_groupement")
                site_web_principal = fields.get("site_web_principal")

                # If this record's identifier is not in our keep list, mark it for deletion
                if (
                    n_siren_groupement,
                    site_web_principal,
                ) not in keep_identifiers:
                    rows_to_delete.append(record_id)

            # If there are rows to delete, send the delete request
            if rows_to_delete:
                delete_url = f"{base}/tables/AomTransportOffers/data/delete"
                response = requests.post(
                    delete_url, headers=self.headers, json=rows_to_delete
                )
                response.raise_for_status()
                return {
                    "deleted": len(rows_to_delete),
                    "rowIds": rows_to_delete,
                }
            else:
                return {"deleted": 0, "message": "No rows to delete"}

        except Exception as e:
            logging.error(
                f"Erreur lors de la suppression des offres de transport des AOMs: {e}"
            )
            raise

    async def update_aom_transport_offers(
        self, offers: List[AomTransportOffer], doc_id: str
    ) -> Dict[str, Union[int, List[Dict]]]:
        """
        Update AOM transport offers in Grist

        Args:
            offers: List of AomTransportOffer objects to update
            doc_id: The Grist document ID to use for this operation
        """
        try:
            base = f"{self.base_url}/api/docs/{doc_id}"
            url = f"{base}/tables/AomTransportOffers/records"

            # Format the records for the API request
            records = []
            for offer in offers:
                offer_dict = offer.model_dump()
                # Use a combination of n_siren_groupement and site_web_principal as identifiers
                record = {
                    "require": {
                        "n_siren_groupement": offer_dict.get(
                            "n_siren_groupement"
                        ),
                        "site_web_principal": offer_dict.get(
                            "site_web_principal"
                        ),
                    },
                    "fields": offer_dict,
                }
                records.append(record)

            payload = {"records": records}

            # Make the PUT request to update the records
            response = requests.put(url, headers=self.headers, json=payload)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            logging.error(
                f"Erreur lors de la mise à jour des offres de transport des AOMs: {e}"
            )
            raise

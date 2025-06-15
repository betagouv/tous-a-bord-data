import logging

import requests
from constants.urls import URL_DATASET_AOM


def get_aom_dataset():
    url = URL_DATASET_AOM
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors (4XX, 5XX)
        data = response.json()
        # Filtrer les ressources au format ODS
        ods_resources = [
            r
            for r in data.get("resources", [])
            if r.get("format", "").lower() == "ods"
        ]
        if not ods_resources:
            return None
        # Trier par date de mise à jour et prendre la plus récente
        latest_resource = max(
            ods_resources, key=lambda x: x.get("updated", "")
        )
        return latest_resource
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return {"error": f"HTTP error: {http_err}"}
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred: {conn_err}")
        return {"error": f"Connection error: {conn_err}"}
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred: {timeout_err}")
        return {"error": f"Timeout error: {timeout_err}"}
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
        return {"error": f"Request error: {req_err}"}
    except ValueError as json_err:
        logging.error(f"JSON parsing error occurred: {json_err}")
        return {"error": f"JSON parsing error: {json_err}"}
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return {"error": f"Unexpected error: {e}"}

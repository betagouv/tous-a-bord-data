import requests
from constants.urls import URL_DATASET_AOM, URL_TRANSPORT_GOUV_DATASETS


def get_aom_dataset():
    url = URL_DATASET_AOM
    response = requests.get(url)
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
    latest_resource = max(ods_resources, key=lambda x: x.get("updated", ""))
    return latest_resource


def filter_datasets_with_fares():
    # Get data from the API
    url = URL_TRANSPORT_GOUV_DATASETS
    response = requests.get(url)
    datasets = response.json()

    # Filter datasets
    filtered_datasets = []
    for dataset in datasets:
        if dataset.get("type") != "public-transit":
            continue
        # Loop through each dataset's resources
        for resource in dataset.get("resources", []):
            # Check if metadata and fares_rules_count exist
            metadata = resource.get("metadata", {})
            has_fares_rules = (
                metadata
                and metadata.get("stats", {}).get("fares_rules_count", 0) != 0
            )
            # Check if "tarifs" is in features
            features = resource.get("features", [])
            has_tarifs_feature = "tarifs" in features
            if has_fares_rules or has_tarifs_feature:
                filtered_datasets.append(dataset)
                # Break once we find a matching resource
                break

    return filtered_datasets

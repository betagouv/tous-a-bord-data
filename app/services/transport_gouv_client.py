import requests
from constants.urls import URL_TRANSPORT_GOUV_DATASETS


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

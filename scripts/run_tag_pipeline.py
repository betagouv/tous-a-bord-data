import asyncio
import logging
import os

from services.batch_tag_extraction import BatchProcessor
from services.grist_service import GristDataService


async def process_siren(siren: str):
    grist_service = GristDataService.get_instance(os.getenv("GRIST_API_KEY"))
    doc_id = os.getenv("GRIST_DOC_INTERMEDIARY_ID")
    aom = await grist_service.get_aom_transport_offer_by_siren(doc_id, siren)

    processor = BatchProcessor()
    result = await processor.process_single_aom(aom)

    # Version 1: Log des résultats
    logging.info(result)

    # Version 2: Sauvegarde dans Grist
    # await grist_service.save_result(aom.id, result)


def main(siren: str):
    asyncio.run(process_siren(siren))


if __name__ == "__main__":
    main(os.getenv("SIREN"))

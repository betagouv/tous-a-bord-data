import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List

from constants.keywords import DEFAULT_KEYWORDS
from services.grist_service import GristDataService
from services.nlp_services import (
    extract_markdown_text,
    extract_tags_and_providers,
    filter_transport_fare,
    load_spacy_model,
    normalize_text,
)
from services.tsst_spacy_llm_task import TSSTClassifier
from utils.crawler_utils import CrawlerManager

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Résultat du traitement d'une AOM"""

    n_siren_aom: str
    nom_aom: str
    status: str  # 'success', 'error', 'no_data', 'processing'
    tags: List[str] = None
    providers: List[str] = None
    error_message: str = None
    processing_time: float = None
    nb_tokens_original: int = 0
    nb_tokens_filtered: int = 0
    current_step: str = ""  # Nouvelle propriété pour l'étape courante


class BatchProcessor:
    """Service de traitement batch pour toutes les AOMs"""

    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers
        self.nlp = None
        self._lock = threading.Lock()  # Pour thread-safety
        self.keywords = DEFAULT_KEYWORDS.copy()  # Mots-clés pour le scraping
        self.model_name = None  # Modèle LLM pour la classification TSST
        # Créer une seule instance de CrawlerManager partagée
        self.crawler_manager = CrawlerManager()

    def initialize_nlp(self):
        """Initialise le modèle NLP une seule fois"""
        if self.nlp is None:
            self.nlp = load_spacy_model()

    def count_tokens_simple(self, text: str) -> int:
        """Compte approximativement les tokens (méthode rapide)"""
        return len(text.split()) if text else 0

    async def process_single_aom(
        self,
        aom,
        step_callback: Callable = None,
    ) -> BatchResult:
        """Traite une seule AOM avec callbacks détaillés"""
        start_time = datetime.now()

        self.initialize_nlp()
        siren = str(aom.n_siren_aom) if hasattr(aom, "n_siren_aom") else ""
        nom_aom = aom.nom_aom if hasattr(aom, "nom_aom") else f"AOM {siren}"
        url = getattr(aom, "site_web_principal", None)

        def update_step(step: str, progress: float):
            if step_callback:
                step_callback(siren, nom_aom, step, progress)

        try:
            logger.info(f"Traitement de {siren} - {nom_aom}")
            update_step("Démarrage du traitement", 0.0)

            # 1. Récupérer le contenu via le crawler
            logger.info("Récupération du contenu via le crawler")
            update_step("Récupération du contenu via le crawler", 0.1)
            raw_content = ""

            if url:
                try:
                    # Utiliser le crawler partagé pour récupérer le contenu
                    pages = await self.crawler_manager.fetch_content(
                        url, self.keywords
                    )

                    # Combiner le contenu de toutes les pages
                    for page in pages:
                        raw_content += (
                            f"--- Page: {page.url} ---\n{page.markdown}\n\n"
                        )
                except Exception as e:
                    logger.error(f"Erreur lors du crawling de {url}: {str(e)}")
                    # Continuer avec un contenu vide
            else:
                logger.warning(f"Aucune URL fournie pour l'AOM {siren}")

            if not raw_content or not raw_content.strip():
                update_step("Terminé - Aucun contenu", 1.0)
                return BatchResult(
                    n_siren_aom=siren,
                    nom_aom=nom_aom,
                    status="no_data",
                    error_message="Aucun contenu trouvé",
                    processing_time=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    current_step="Terminé - Aucun contenu",
                )

            nb_tokens_original = self.count_tokens_simple(raw_content)

            # 2. Filtrage NLP
            logger.info("Extraction du texte markdown")
            update_step("Extraction du texte markdown", 0.3)
            raw_text = extract_markdown_text(raw_content)
            logger.info("Normalisation du texte")
            update_step("Normalisation du texte", 0.4)
            paragraphs = normalize_text(raw_text, self.nlp)

            logger.info("Filtrage des paragraphes pertinents")
            update_step("Filtrage des paragraphes pertinents", 0.6)
            paragraphs_filtered, _ = filter_transport_fare(
                paragraphs, self.nlp
            )
            filtered_content = "\n\n".join(paragraphs_filtered)

            if not filtered_content or not filtered_content.strip():
                logger.info("Terminé - Aucun contenu pertinent")
                update_step("Terminé - Aucun contenu pertinent", 1.0)
                return BatchResult(
                    n_siren_aom=siren,
                    nom_aom=nom_aom,
                    status="no_data",
                    error_message="Aucun contenu pertinent après filtrage",
                    processing_time=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    nb_tokens_original=nb_tokens_original,
                    nb_tokens_filtered=0,
                    current_step="Terminé - Aucun contenu pertinent",
                )

            nb_tokens_filtered = self.count_tokens_simple(filtered_content)

            # 3. Classification TSST avec LLM
            if self.model_name:
                logger.info(
                    f"Classification TSST avec le modèle {self.model_name}"
                )
                update_step(f"Classification TSST avec {self.model_name}", 0.7)
                try:
                    # Initialiser le classifieur TSST avec le modèle spécifié
                    classifier = TSSTClassifier(model_name=self.model_name)

                    # Classifier le contenu
                    is_tsst, details = classifier.classify_paragraph(
                        filtered_content
                    )

                    # Si le contenu ne concerne pas la TSST, arrêter le traitement
                    if not is_tsst:
                        logger.info("Terminé - Contenu non TSST")
                        update_step("Terminé - Contenu non TSST", 1.0)
                        return BatchResult(
                            n_siren_aom=siren,
                            nom_aom=nom_aom,
                            status="no_data",
                            error_message="Le contenu ne concerne pas la tarification sociale et solidaire des transports",
                            processing_time=(
                                datetime.now() - start_time
                            ).total_seconds(),
                            nb_tokens_original=nb_tokens_original,
                            nb_tokens_filtered=nb_tokens_filtered,
                            current_step="Terminé - Contenu non TSST",
                        )
                except Exception as e:
                    logger.error(
                        f"Erreur lors de la classification TSST: {str(e)}"
                    )
                    # Continuer même en cas d'erreur de classification

            # 4. Extraction des tags et fournisseurs
            logger.info("Extraction des tags et fournisseurs")
            update_step("Extraction des tags et fournisseurs", 0.8)
            tags, providers, _, _ = extract_tags_and_providers(
                filtered_content, self.nlp, siren, nom_aom
            )

            logger.info("Update AOM dans Grist")
            update_step("Update AOM dans Grist", 0.9)

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Traitement terminé pour {siren} - {nom_aom} en {processing_time:.2f} secondes"
            )
            update_step("Terminé avec succès", 1.0)
            grist_service = GristDataService.get_instance(
                api_key=os.getenv("GRIST_API_KEY")
            )
            doc_id = os.getenv("GRIST_DOC_OUTPUT_ID")
            await grist_service.update_aom_tags(aom, doc_id)

            return BatchResult(
                n_siren_aom=siren,
                nom_aom=nom_aom,
                status="success",
                tags=tags,
                providers=providers,
                processing_time=processing_time,
                nb_tokens_original=nb_tokens_original,
                nb_tokens_filtered=nb_tokens_filtered,
                current_step="Terminé avec succès",
            )

        except Exception as e:
            logger.error(f"Erreur pour {siren} - {nom_aom}: {str(e)}")
            update_step(f"Erreur: {str(e)}", 1.0)

            return BatchResult(
                n_siren_aom=siren,
                nom_aom=nom_aom,
                status="error",
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds(),
                current_step=f"Erreur: {str(e)}",
            )

    async def process_batch(
        self,
        aom_list,
        progress_callback=None,
        step_callback=None,
    ) -> List[BatchResult]:
        """Traite un batch d'AOMs de manière séquentielle"""

        logger.info(f"Démarrage du traitement batch pour {len(aom_list)} AOMs")

        results = []

        # Traitement séquentiel (une AOM à la fois)
        for i, aom in enumerate(aom_list):
            # Extraire les informations de l'AOM
            siren = ""
            nom_aom = ""
            if hasattr(aom, "n_siren_aom") and hasattr(aom, "nom_aom"):
                # Si c'est un objet AomTransportOffer
                siren = str(aom.n_siren_aom)
                nom_aom = aom.nom_aom

            try:
                # Traiter cette AOM
                result = await self.process_single_aom(aom, step_callback)
                results.append(result)

                # Callback pour mettre à jour la progress bar
                if progress_callback:
                    progress_callback(i + 1, len(aom_list), result)

                # Petit délai entre les traitements pour laisser l'event loop "respirer"
                import time

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Erreur inattendue pour {siren}: {str(e)}")
                results.append(
                    BatchResult(
                        n_siren_aom=siren,
                        nom_aom=nom_aom,
                        status="error",
                        error_message=f"Erreur inattendue: {str(e)}",
                        current_step="Erreur inattendue",
                    )
                )

        logger.info(f"Traitement batch terminé: {len(results)} résultats")
        return results

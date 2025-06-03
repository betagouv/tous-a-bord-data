import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from services.nlp_services import (
    extract_markdown_text,
    extract_tags_and_providers,
    filter_transport_fare,
    load_spacy_model,
    normalize_text,
)
from sqlalchemy import create_engine, text
from utils.db_utils import get_postgres_cs

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
    nb_sources: int = 0
    current_step: str = ""  # Nouvelle propriété pour l'étape courante


@dataclass
class ProcessingStatus:
    """Status de traitement en temps réel"""

    aom_id: str
    nom_aom: str
    status: str
    current_step: str
    progress: float  # 0.0 à 1.0


class BatchProcessor:
    """Service de traitement batch pour toutes les AOMs"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.engine = create_engine(get_postgres_cs())
        self.nlp = None
        self._processing_status = (
            {}
        )  # Dict pour suivre le status de chaque AOM
        self._lock = threading.Lock()  # Pour thread-safety

    def initialize_nlp(self):
        """Initialise le modèle NLP une seule fois"""
        if self.nlp is None:
            self.nlp = load_spacy_model()

    def get_all_aoms(self) -> List[Tuple[str, str, int]]:
        """Récupère toutes les AOMs avec des données"""
        with self.engine.connect() as conn:
            aoms = conn.execute(
                text(
                    """
                    SELECT DISTINCT
                        t.n_siren_aom,
                        a.nom_aom,
                        COUNT(DISTINCT t.url_source) as nb_sources,
                        STRING_AGG(DISTINCT t.url_source, ' | ') as sources
                    FROM tarification_raw t
                    LEFT JOIN aoms a ON t.n_siren_aom = a.n_siren_aom
                    GROUP BY t.n_siren_aom, a.nom_aom
                    ORDER BY COUNT(DISTINCT t.url_source) DESC, a.nom_aom
                    """
                )
            ).fetchall()
        return [(aom[0], aom[1], aom[2]) for aom in aoms]

    def get_aom_content(self, siren: str) -> str:
        """Récupère tout le contenu d'une AOM depuis la DB"""
        with self.engine.connect() as conn:
            # Récupérer toutes les sources pour cette AOM
            sources = conn.execute(
                text(
                    """
                    SELECT DISTINCT url_source
                    FROM tarification_raw
                    WHERE n_siren_aom = :siren
                """
                ),
                {"siren": siren},
            ).fetchall()

            all_content = []
            for source in sources:
                # Récupérer toutes les pages pour cette source
                pages = conn.execute(
                    text(
                        """
                        SELECT url_page, contenu_scrape
                        FROM tarification_raw
                        WHERE n_siren_aom = :siren
                        AND url_source = :url
                        ORDER BY id
                    """
                    ),
                    {"siren": siren, "url": source[0]},
                ).fetchall()

                for page in pages:
                    all_content.append(
                        f"--- Page: {page.url_page} ---\n{page.contenu_scrape}"
                    )

        return "\n\n".join(all_content)

    def count_tokens_simple(self, text: str) -> int:
        """Compte approximativement les tokens (méthode rapide)"""
        return len(text.split()) if text else 0

    def _update_processing_status(
        self,
        siren: str,
        nom_aom: str,
        step: str,
        progress: float,
        status: str = "processing",
    ):
        """Met à jour le status de traitement d'une AOM"""
        with self._lock:
            self._processing_status[siren] = ProcessingStatus(
                aom_id=siren,
                nom_aom=nom_aom,
                status=status,
                current_step=step,
                progress=progress,
            )

    def get_processing_status(self) -> Dict[str, ProcessingStatus]:
        """Récupère le status actuel de tous les traitements"""
        with self._lock:
            return self._processing_status.copy()

    def process_single_aom(
        self,
        siren: str,
        nom_aom: str,
        nb_sources: int,
        step_callback: Callable = None,
    ) -> BatchResult:
        """Traite une seule AOM avec callbacks détaillés"""
        start_time = datetime.now()

        def update_step(step: str, progress: float):
            if step_callback:
                step_callback(siren, nom_aom, step, progress)
            self._update_processing_status(siren, nom_aom, step, progress)

        try:
            logger.info(f"Traitement de {siren} - {nom_aom}")
            update_step("Démarrage du traitement", 0.0)

            # 1. Récupérer le contenu
            update_step("Récupération du contenu depuis la DB", 0.1)
            raw_content = self.get_aom_content(siren)
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
                    nb_sources=nb_sources,
                    current_step="Terminé - Aucun contenu",
                )

            nb_tokens_original = self.count_tokens_simple(raw_content)
            update_step("Extraction du texte markdown", 0.3)

            # 2. Filtrage NLP
            update_step("Normalisation du texte", 0.4)
            raw_text = extract_markdown_text(raw_content)
            paragraphs = normalize_text(raw_text, self.nlp)

            update_step("Filtrage des paragraphes pertinents", 0.6)
            paragraphs_filtered, _ = filter_transport_fare(
                paragraphs, self.nlp
            )
            filtered_content = "\n\n".join(paragraphs_filtered)

            if not filtered_content or not filtered_content.strip():
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
                    nb_sources=nb_sources,
                    current_step="Terminé - Aucun contenu pertinent",
                )

            nb_tokens_filtered = self.count_tokens_simple(filtered_content)

            # 3. Extraction des tags et fournisseurs
            update_step("Extraction des tags et fournisseurs", 0.8)
            tags, providers, _, _ = extract_tags_and_providers(
                filtered_content, self.nlp, siren, nom_aom
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            update_step("Terminé avec succès", 1.0)

            # Marquer comme terminé dans le status
            with self._lock:
                if siren in self._processing_status:
                    self._processing_status[siren].status = "success"

            return BatchResult(
                n_siren_aom=siren,
                nom_aom=nom_aom,
                status="success",
                tags=tags,
                providers=providers,
                processing_time=processing_time,
                nb_tokens_original=nb_tokens_original,
                nb_tokens_filtered=nb_tokens_filtered,
                nb_sources=nb_sources,
                current_step="Terminé avec succès",
            )

        except Exception as e:
            logger.error(f"Erreur pour {siren} - {nom_aom}: {str(e)}")
            update_step(f"Erreur: {str(e)}", 1.0)

            # Marquer comme erreur dans le status
            with self._lock:
                if siren in self._processing_status:
                    self._processing_status[siren].status = "error"

            return BatchResult(
                n_siren_aom=siren,
                nom_aom=nom_aom,
                status="error",
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds(),
                nb_sources=nb_sources,
                current_step=f"Erreur: {str(e)}",
            )

    def process_batch(
        self,
        aom_list: Optional[List[Tuple[str, str, int]]] = None,
        progress_callback=None,
        step_callback=None,
    ) -> List[BatchResult]:
        """Traite un batch d'AOMs en parallèle"""

        # Initialiser le modèle NLP une seule fois
        self.initialize_nlp()

        # Récupérer la liste des AOMs si non fournie
        if aom_list is None:
            aom_list = self.get_all_aoms()

        logger.info(f"Démarrage du traitement batch pour {len(aom_list)} AOMs")

        # Réinitialiser le status
        with self._lock:
            self._processing_status.clear()

        results = []

        # Traitement en parallèle
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Soumettre tous les jobs
            future_to_aom = {
                executor.submit(
                    self.process_single_aom,
                    siren,
                    nom_aom,
                    nb_sources,
                    step_callback,
                ): (siren, nom_aom)
                for siren, nom_aom, nb_sources in aom_list
            }

            # Récupérer les résultats au fur et à mesure
            for i, future in enumerate(as_completed(future_to_aom)):
                try:
                    result = future.result()
                    results.append(result)

                    # Callback pour mettre à jour la progress bar
                    if progress_callback:
                        progress_callback(i + 1, len(aom_list), result)

                except Exception as e:
                    siren, nom_aom = future_to_aom[future]
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

    def save_results_to_db(
        self, results: List[BatchResult], table_name: str = "batch_results"
    ):
        """Sauvegarde les résultats en base de données"""
        # Convertir en DataFrame
        data = []
        for result in results:
            data.append(
                {
                    "n_siren_aom": result.n_siren_aom,
                    "nom_aom": result.nom_aom,
                    "status": result.status,
                    "tags": ",".join(result.tags) if result.tags else None,
                    "providers": ",".join(result.providers)
                    if result.providers
                    else None,
                    "error_message": result.error_message,
                    "processing_time": result.processing_time,
                    "nb_tokens_original": result.nb_tokens_original,
                    "nb_tokens_filtered": result.nb_tokens_filtered,
                    "nb_sources": result.nb_sources,
                    "current_step": result.current_step,
                    "created_at": datetime.now(),
                }
            )

        df = pd.DataFrame(data)

        # Sauvegarder en base
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        logger.info(f"Résultats sauvegardés dans la table {table_name}")

    def get_summary_stats(self, results: List[BatchResult]) -> Dict:
        """Génère des statistiques de synthèse"""
        total = len(results)
        success = len([r for r in results if r.status == "success"])
        errors = len([r for r in results if r.status == "error"])
        no_data = len([r for r in results if r.status == "no_data"])

        # Temps de traitement
        processing_times = [
            r.processing_time for r in results if r.processing_time
        ]
        avg_processing_time = (
            sum(processing_times) / len(processing_times)
            if processing_times
            else 0
        )

        # Tags les plus fréquents
        all_tags = []
        for r in results:
            if r.tags:
                all_tags.extend(r.tags)

        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Fournisseurs les plus fréquents
        all_providers = []
        for r in results:
            if r.providers:
                all_providers.extend(r.providers)

        provider_counts = {}
        for provider in all_providers:
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        return {
            "total_aoms": total,
            "success_count": success,
            "error_count": errors,
            "no_data_count": no_data,
            "success_rate": (success / total * 100) if total > 0 else 0,
            "avg_processing_time": avg_processing_time,
            "total_tags_found": len(all_tags),
            "unique_tags_count": len(tag_counts),
            "most_common_tags": sorted(
                tag_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "total_providers_found": len(all_providers),
            "unique_providers_count": len(provider_counts),
            "most_common_providers": sorted(
                provider_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

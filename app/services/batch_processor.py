import asyncio
from typing import Dict, List

import streamlit as st
from services.llm_services import (
    LLM_MODELS,
    call_anthropic,
    call_ollama,
    call_scaleway,
)
from services.nlp_services import (
    extract_markdown_text,
    filter_text_with_spacy,
    load_spacy_model,
    normalize_text,
)
from sqlalchemy import create_engine
from utils.db_utils import get_postgres_cs


class BatchProcessor:
    def __init__(
        self,
        max_concurrent_aoms: int = 5,
        selected_steps: List[str] = None,
        methods_config: Dict = None,
    ):
        """
        Initialise le processeur batch.

        Args:
            max_concurrent_aoms: Nombre maximum d'AOMs à traiter en parallèle
            selected_steps: Liste des étapes à exécuter
            methods_config: Configuration des méthodes pour chaque étape
        """
        self.max_concurrent_aoms = max_concurrent_aoms
        default_steps = [
            "filtering",
            "pre_formatting",
            "yaml_formatting",
        ]
        self.selected_steps = selected_steps or default_steps
        self.methods_config = methods_config or {}
        self.engine = create_engine(get_postgres_cs())
        self.nlp = load_spacy_model()

    def get_model_name(self, display_name: str) -> str:
        """
        Récupère le nom technique du modèle à partir de son nom d'affichage.
        """
        if not display_name:
            return ""
        return (
            LLM_MODELS[display_name]["name"]
            if display_name in LLM_MODELS
            else display_name
        )

    async def process_single_aom(
        self, siren: str, nom: str, content: str, progress_callback=None
    ) -> Dict:
        """
        Traite une seule AOM de manière asynchrone.

        Args:
            siren: SIREN de l'AOM
            nom: Nom de l'AOM
            content: Contenu scrapé
            progress_callback: Fonction de callback pour la progression

        Returns:
            Dict contenant les résultats du traitement
        """
        try:
            result = {"siren": siren, "nom": nom}

            # Étape 2: Filtrage
            if "filtering" in self.selected_steps:
                filter_config = self.methods_config.get("filtering", {})

                if filter_config.get("method") == "NLP (SpaCy)":
                    raw_text = extract_markdown_text(content)
                    paragraphs = normalize_text(raw_text, self.nlp)
                    paragraphs_filtered, _ = filter_text_with_spacy(
                        paragraphs, self.nlp
                    )
                    result["filtering"] = "\n\n".join(paragraphs_filtered)
                else:
                    model = self.get_model_name(filter_config.get("model"))
                    if "claude" in model.lower():
                        result["filtering"] = call_anthropic(
                            content, model=model
                        )
                    elif "llama" in model.lower():
                        result["filtering"] = call_ollama(content, model=model)
                    else:
                        result["filtering"] = call_scaleway(
                            content, model=model
                        )

                if progress_callback:
                    progress_callback(siren, "filtering", 0.33)

            # Étape 3: Pré-formatage
            has_filtering = "filtering" not in self.selected_steps
            has_filtering_result = result.get("filtering")
            if "pre_formatting" in self.selected_steps and (
                has_filtering or has_filtering_result
            ):
                content_to_format = result.get("filtering", content)
                preformat_config = self.methods_config.get(
                    "pre_formatting", {}
                )
                model = self.get_model_name(preformat_config.get("model"))

                if "claude" in model.lower():
                    result["pre_formatting"] = call_anthropic(
                        content_to_format, model=model
                    )
                elif "llama" in model.lower():
                    result["pre_formatting"] = call_ollama(
                        content_to_format, model=model
                    )
                else:
                    result["pre_formatting"] = call_scaleway(
                        content_to_format, model=model
                    )

                if progress_callback:
                    progress_callback(siren, "pre_formatting", 0.66)

            # Étape 4: Format YAML
            has_preformat = "pre_formatting" not in self.selected_steps
            has_preformat_result = result.get("pre_formatting")
            if "yaml_formatting" in self.selected_steps and (
                has_preformat or has_preformat_result
            ):
                content_to_yaml = result.get(
                    "pre_formatting", result.get("filtering", content)
                )
                yaml_config = self.methods_config.get("yaml_formatting", {})
                model = self.get_model_name(yaml_config.get("model"))

                if "claude" in model.lower():
                    result["yaml_formatting"] = call_anthropic(
                        content_to_yaml, model=model
                    )
                elif "llama" in model.lower():
                    result["yaml_formatting"] = call_ollama(
                        content_to_yaml, model=model
                    )
                else:
                    result["yaml_formatting"] = call_scaleway(
                        content_to_yaml, model=model
                    )

                if progress_callback:
                    progress_callback(siren, "yaml_formatting", 1.0)

            return result

        except Exception as e:
            error_msg = (
                f"Erreur lors du traitement de l'AOM {nom} ({siren}): {str(e)}"
            )
            st.error(error_msg)
            return {"siren": siren, "nom": nom, "error": str(e)}

    async def process_all_aoms(
        self, aoms_data: List[Dict], progress_callback=None
    ) -> List[Dict]:
        """
        Traite toutes les AOMs en parallèle.

        Args:
            aoms_data: Liste des AOMs à traiter
            progress_callback: Fonction de callback pour la progression

        Returns:
            Liste des résultats de traitement
        """
        tasks = []
        semaphore = asyncio.Semaphore(self.max_concurrent_aoms)

        async def process_with_semaphore(aom):
            async with semaphore:
                return await self.process_single_aom(
                    aom["siren"], aom["nom"], aom["content"], progress_callback
                )

        for aom in aoms_data:
            task = asyncio.create_task(process_with_semaphore(aom))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

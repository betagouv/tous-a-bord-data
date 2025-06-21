import json
import logging
import re
from typing import Any, Dict, List, Tuple

# Import du prompt pour la classification TSST
from prompts.analyze_tsst_context import (
    generate_tsst_classification_prompt_with_examples,
)

# Import des services LLM existants
from services.llm_services import LLM_MODELS, call_scaleway


class TSSTClassifier:
    """
    Classifieur pour déterminer si un texte concerne la tarification
    sociale et solidaire des transports (TSST) en utilisant directement
    les services LLM existants.
    """

    def __init__(self, model_name: str = "Llama 3.1 8B (Scaleway)"):
        """
        Initialise le classifieur avec un modèle LLM.

        Args:
            model_name: Nom du modèle LLM à utiliser (dans LLM_MODELS)
        """
        if model_name not in LLM_MODELS:
            raise ValueError(
                f"Modèle {model_name} non disponible. "
                f"Options: {list(LLM_MODELS.keys())}"
            )

        # Stocker le modèle utilisé
        self.model_name = model_name
        self.model_info = LLM_MODELS[model_name]
        self.model_technical_name = self.model_info["name"]

        logging.info(
            f"Classifieur TSST initialisé avec le modèle: {model_name}"
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Appelle le LLM avec le prompt donné.

        Args:
            prompt: Prompt à envoyer au LLM

        Returns:
            str: Réponse du LLM
        """
        try:
            # Sélectionner le service LLM approprié
            if "Scaleway" in self.model_name:
                response = call_scaleway(
                    prompt, model=self.model_technical_name
                )
            else:
                raise ValueError(
                    f"Service LLM non supporté pour le modèle "
                    f"{self.model_name}"
                )

            return response
        except Exception as e:
            error_msg = f"Erreur lors de l'appel au LLM: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse la réponse du LLM pour extraire les scores de classification
        et la justification.

        Args:
            response: Réponse du LLM

        Returns:
            Dict[str, Any]: Scores de classification et justification
        """
        # Rechercher les scores dans la réponse
        try:
            # Nettoyer la réponse
            response = response.strip()

            # Initialiser le résultat
            result = {"TSST": 0.0, "NON_TSST": 1.0, "justification": ""}

            # Essayer d'extraire une justification
            justification_match = re.search(
                r"[Jj]ustification\s*:(.+?)(?:\n\n|\Z)", response, re.DOTALL
            )
            if justification_match:
                result["justification"] = justification_match.group(1).strip()
            else:
                # Chercher d'autres formats de justification
                justification_match = re.search(
                    r"D\'après l\'analyse(.+?)(?:\n\n|\Z)", response, re.DOTALL
                )
                if justification_match:
                    result["justification"] = justification_match.group(
                        0
                    ).strip()

            # Essayer de trouver un format JSON dans la réponse
            json_pattern = (
                r'\{.*"TSST".*"NON_TSST".*\}|\{.*"NON_TSST".*"TSST".*\}'
            )
            json_match = re.search(
                json_pattern,
                response,
                re.DOTALL,
            )
            if json_match:
                json_str = json_match.group(0)
                scores = json.loads(json_str)
                result.update(scores)
                return result

            # Rechercher des patterns comme "TSST: 0.8" ou "NON_TSST: 0.2"
            tsst_match = re.search(r"TSST\s*:\s*(0\.\d+|1\.0|1|0)", response)
            non_tsst_match = re.search(
                r"NON_TSST\s*:\s*(0\.\d+|1\.0|1|0)", response
            )

            if tsst_match and non_tsst_match:
                result["TSST"] = float(tsst_match.group(1))
                result["NON_TSST"] = float(non_tsst_match.group(1))
                return result

            # Si le LLM répond simplement "TSST" ou "NON_TSST"
            has_tsst = re.search(r"\bTSST\b", response)
            has_non_tsst = re.search(r"\bNON_TSST\b", response)

            if has_tsst and not has_non_tsst:
                result["TSST"] = 1.0
                result["NON_TSST"] = 0.0
                return result
            elif has_non_tsst and not has_tsst:
                result["TSST"] = 0.0
                result["NON_TSST"] = 1.0
                return result

            # Rechercher des patterns comme "Classification: TSST"
            if re.search(r"[Cc]lassification\s*:?\s*TSST", response):
                result["TSST"] = 1.0
                result["NON_TSST"] = 0.0
                return result
            elif re.search(r"[Cc]lassification\s*:?\s*NON_TSST", response):
                result["TSST"] = 0.0
                result["NON_TSST"] = 1.0
                return result

            # Rechercher des patterns comme "je le classe en TSST"
            tsst_pattern = r"je\s+(?:le\s+)?classe\s+(?:en|comme)\s+TSST"
            non_tsst_pattern = (
                r"je\s+(?:le\s+)?classe\s+(?:en|comme)\s+NON_TSST"
            )

            if re.search(tsst_pattern, response, re.IGNORECASE):
                result["TSST"] = 1.0
                result["NON_TSST"] = 0.0
                return result
            elif re.search(non_tsst_pattern, response, re.IGNORECASE):
                result["TSST"] = 0.0
                result["NON_TSST"] = 1.0
                return result

            # Rechercher des patterns comme "Le texte concerne la TSST"
            concerne_pattern = r"concerne\s+la\s+TSST"
            ne_concerne_pas_pattern = r"ne\s+concerne\s+pas"
            ne_concerne_pas_tsst = r"ne\s+concerne\s+pas\s+la\s+TSST"

            if re.search(
                concerne_pattern, response, re.IGNORECASE
            ) and not re.search(
                ne_concerne_pas_pattern, response, re.IGNORECASE
            ):
                result["TSST"] = 1.0
                result["NON_TSST"] = 0.0
                return result
            elif re.search(ne_concerne_pas_tsst, response, re.IGNORECASE):
                result["TSST"] = 0.0
                result["NON_TSST"] = 1.0
                return result

            # Par défaut, considérer comme non TSST
            logging.warning(
                f"Impossible de déterminer la classification à partir de "
                f"la réponse: {response}"
            )
            return result
        except Exception as e:
            logging.error(
                f"Erreur lors du parsing de la réponse LLM: " f"{str(e)}"
            )
            return {"TSST": 0.0, "NON_TSST": 1.0}

    def classify_paragraph(self, paragraph: str) -> Tuple[bool, Dict]:
        """
        Classifie un paragraphe pour déterminer s'il concerne la TSST.

        Args:
            paragraph: Paragraphe à analyser

        Returns:
            Tuple[bool, Dict]: (est_tsst, détails de classification)
        """
        # Construire le prompt avec des exemples
        prompt = generate_tsst_classification_prompt_with_examples(paragraph)

        # Appeler le LLM
        response = self._call_llm(prompt)

        # Parser la réponse
        result = self._parse_llm_response(response)

        # Extraire les scores et la justification
        scores = {
            "TSST": result.get("TSST", 0),
            "NON_TSST": result.get("NON_TSST", 1),
        }
        justification = result.get("justification", "")

        # Déterminer si le paragraphe concerne la TSST
        is_tsst = scores.get("TSST", 0) > scores.get("NON_TSST", 0)

        # Stocker les détails de classification
        details = {
            "paragraph": paragraph,
            "is_tsst": is_tsst,
            "scores": scores,
            "justification": justification,
            "prompt": prompt,
            "response": response,
        }

        # Logging
        if is_tsst:
            logging.info(f"Paragraphe validé comme TSST: {paragraph[:50]}...")
        else:
            logging.info(f"Paragraphe rejeté (non TSST): {paragraph[:50]}...")

        return is_tsst, details

    def classify_paragraphs(
        self, paragraphs: List[str]
    ) -> Tuple[List[str], List[Dict]]:
        """
        Classifie une liste de paragraphes pour déterminer s'ils concernent
        la TSST.

        Args:
            paragraphs: Liste de paragraphes à analyser

        Returns:
            Tuple[List[str], List[Dict]]: Paragraphes validés et détails
        """
        validated_paragraphs = []
        classification_details = []

        # Traiter les paragraphes un par un
        for paragraph in paragraphs:
            is_tsst, details = self.classify_paragraph(paragraph)

            # Si le paragraphe concerne la TSST, l'ajouter à la liste validée
            if is_tsst:
                validated_paragraphs.append(paragraph)

            classification_details.append(details)

        return validated_paragraphs, classification_details


# Fonction utilitaire pour intégration facile
def filter_tsst_context(
    paragraphs: List[str], model_name: str = "Llama 3.1 8B (Scaleway)"
) -> List[str]:
    """
    Filtre les paragraphes pour ne garder que ceux concernant la TSST.

    Args:
        paragraphs: Liste de paragraphes à filtrer
        model_name: Nom du modèle LLM à utiliser (dans LLM_MODELS)

    Returns:
        List[str]: Paragraphes filtrés
    """
    try:
        classifier = TSSTClassifier(model_name=model_name)
        validated_paragraphs, _ = classifier.classify_paragraphs(paragraphs)
        return validated_paragraphs
    except Exception as e:
        logging.error(f"Erreur lors de la classification TSST: {str(e)}")
        return paragraphs

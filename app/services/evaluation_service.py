import os
from typing import Optional

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()


class EvaluationService:
    """Service pour gérer les évaluations et le feedback vers LangSmith"""

    def __init__(self):
        self.client = Client(
            api_key=os.getenv("LANGCHAIN_API_KEY"),
            api_url=os.getenv("LANGCHAIN_ENDPOINT"),
        )
        self.project_name = os.getenv("LANGCHAIN_PROJECT")

    def create_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        correction: str,
    ) -> Optional[str]:
        """
        Crée un feedback pour un run LangSmith

        Args:
            run_id: ID du run LangSmith
            key: Type de feedback (ex: "quality")
            score: Score de 0 à 1 (0 = mauvais, 1 = excellent)
            correction: Correction proposée par l'humain

        Returns:
            ID du feedback créé ou None si erreur
        """
        try:
            feedback = self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                value=correction,
            )
            return feedback.id
        except Exception as e:
            print(f"Erreur lors de la création du feedback: {e}")
            return None


# Instance globale du service d'évaluation
evaluation_service = EvaluationService()

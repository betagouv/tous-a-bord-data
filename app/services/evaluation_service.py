import os
import uuid
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()


class EvaluationService:
    """Service pour gérer les évaluations et le feedback vers LangSmith"""

    def __init__(self):
        self.client = Client(
            api_key=os.getenv("LANGCHAIN_API_KEY"),
            api_url=os.getenv(
                "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
            ),
        )
        self.project_name = os.getenv(
            "LANGCHAIN_PROJECT", "transport-tarifs-pipeline"
        )

    def create_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        value: Optional[str] = None,
        comment: Optional[str] = None,
        correction: Optional[Dict] = None,
    ) -> str:
        """
        Crée un feedback pour un run LangSmith

        Args:
            run_id: ID du run LangSmith
            key: Type de feedback (ex: "quality", "accuracy", "relevance")
            score: Score de 0 à 1 (0 = mauvais, 1 = excellent)
            value: Valeur textuelle du feedback
            comment: Commentaire libre
            correction: Correction proposée par l'humain

        Returns:
            ID du feedback créé
        """
        try:
            feedback = self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                value=value,
                comment=comment,
                correction=correction,
            )
            return feedback.id
        except Exception as e:
            print(f"Erreur lors de la création du feedback: {e}")
            return None

    def get_run_by_id(self, run_id: str):
        """Récupère un run par son ID"""
        try:
            return self.client.read_run(run_id)
        except Exception as e:
            print(f"Erreur lors de la récupération du run: {e}")
            return None

    def list_recent_runs(self, limit: int = 50) -> List:
        """Liste les runs récents du projet"""
        try:
            runs = list(
                self.client.list_runs(
                    project_name=self.project_name, limit=limit
                )
            )
            return runs
        except Exception as e:
            print(f"Erreur lors de la récupération des runs: {e}")
            return []

    def get_feedback_for_run(self, run_id: str) -> List:
        """Récupère tous les feedbacks pour un run donné"""
        try:
            feedbacks = list(self.client.list_feedback(run_ids=[run_id]))
            return feedbacks
        except Exception as e:
            print(f"Erreur lors de la récupération des feedbacks: {e}")
            return []

    def create_evaluation_session(
        self, session_name: str, description: str = ""
    ) -> str:
        """Crée une session d'évaluation pour grouper les évaluations"""
        session_id = str(uuid.uuid4())
        # Pour l'instant, on stocke juste l'ID, mais on pourrait étendre
        # avec une DB
        return session_id

    def get_evaluation_stats(self) -> Dict:
        """Récupère les statistiques d'évaluation du projet"""
        try:
            runs = self.list_recent_runs(limit=100)
            total_runs = len(runs)

            # Compter les runs avec feedback
            runs_with_feedback = 0
            total_score = 0
            feedback_count = 0

            for run in runs:
                feedbacks = self.get_feedback_for_run(run.id)
                if feedbacks:
                    runs_with_feedback += 1
                    for feedback in feedbacks:
                        if feedback.score is not None:
                            total_score += feedback.score
                            feedback_count += 1

            avg_score = (
                total_score / feedback_count if feedback_count > 0 else 0
            )

            return {
                "total_runs": total_runs,
                "runs_with_feedback": runs_with_feedback,
                "feedback_coverage": runs_with_feedback / total_runs
                if total_runs > 0
                else 0,
                "average_score": avg_score,
                "total_feedbacks": feedback_count,
            }
        except Exception as e:
            print(f"Erreur lors du calcul des statistiques: {e}")
            return {
                "total_runs": 0,
                "runs_with_feedback": 0,
                "feedback_coverage": 0,
                "average_score": 0,
                "total_feedbacks": 0,
            }


# Instance globale du service d'évaluation
evaluation_service = EvaluationService()

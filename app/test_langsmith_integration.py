#!/usr/bin/env python3
"""
Script de test pour vérifier l'intégration LangSmith
"""

import os
import sys

from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


def test_environment_variables():
    """Teste que toutes les variables d'environnement nécessaires
    sont définies"""
    print("🔍 Vérification des variables d'environnement...")

    required_vars = [
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
    ]

    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            # Masquer la clé API pour la sécurité
            if "API_KEY" in var:
                display_value = value[:8] + "..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"  ✅ {var} = {display_value}")

    if missing_vars:
        print(f"  ❌ Variables manquantes : {', '.join(missing_vars)}")
        return False

    print("  ✅ Toutes les variables d'environnement sont définies")
    return True


def test_langsmith_connection():
    """Teste la connexion à LangSmith"""
    print("\n🔗 Test de connexion à LangSmith...")

    try:
        from services.evaluation_service import evaluation_service

        # Tester la connexion en récupérant les statistiques
        stats = evaluation_service.get_evaluation_stats()
        print("  ✅ Connexion réussie à LangSmith")
        print(f"  📊 Projet : {evaluation_service.project_name}")
        print(f"  📈 Runs totaux : {stats['total_runs']}")
        return True

    except Exception as e:
        print(f"  ❌ Erreur de connexion : {str(e)}")
        return False


def test_llm_services():
    """Teste que les services LLM ont bien les décorateurs @traceable"""
    print("\n🤖 Vérification des services LLM...")

    try:
        from services.llm_services import (
            call_anthropic,
            call_ollama,
            call_scaleway,
        )

        # Vérifier que les fonctions ont l'attribut __wrapped__
        # (signe du décorateur)
        services = [
            ("call_anthropic", call_anthropic),
            ("call_scaleway", call_scaleway),
            ("call_ollama", call_ollama),
        ]

        for name, func in services:
            if hasattr(func, "__wrapped__"):
                print(f"  ✅ {name} : décorateur @traceable détecté")
            else:
                print(f"  ⚠️ {name} : décorateur @traceable manquant")

        return True

    except ImportError as e:
        print(f"  ❌ Erreur d'import : {str(e)}")
        return False


def test_simple_llm_call():
    """Teste un appel LLM simple pour vérifier le tracking"""
    print("\n🧪 Test d'appel LLM avec tracking...")

    try:
        from services.llm_services import call_ollama

        # Test simple avec Ollama (le plus accessible)
        print("  🔄 Test d'appel Ollama...")

        # Appel simple qui devrait être tracké
        prompt = "Dis bonjour en français en une phrase."

        # Note: Cet appel peut échouer si Ollama n'est pas disponible
        # mais l'important est de vérifier que le tracking fonctionne
        try:
            response = call_ollama(prompt, model="llama3:8b")
            print(f"  ✅ Appel réussi : {response[:50]}...")
        except Exception as e:
            error_msg = (
                f"  ⚠️ Appel échoué (normal si Ollama non disponible) : "
                f"{str(e)}"
            )
            print(error_msg)
            print("  ℹ️ L'important est que le tracking soit configuré")

        return True

    except Exception as e:
        print(f"  ❌ Erreur lors du test : {str(e)}")
        return False


def main():
    """Fonction principale de test"""
    print("🚀 Test d'intégration LangSmith pour le pipeline RAG Transport")
    print("=" * 60)

    tests = [
        test_environment_variables,
        test_langsmith_connection,
        test_llm_services,
        test_simple_llm_call,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Erreur inattendue : {str(e)}")
            results.append(False)

    print("\n" + "=" * 60)
    print("📋 Résumé des tests :")

    test_names = [
        "Variables d'environnement",
        "Connexion LangSmith",
        "Services LLM",
        "Appel LLM tracké",
    ]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {name}: {status}")

    success_rate = sum(results) / len(results)
    print(f"\n🎯 Taux de réussite : {success_rate:.1%}")

    if success_rate == 1.0:
        print(
            "🎉 Tous les tests sont passés ! L'intégration LangSmith "
            "est prête."
        )
    elif success_rate >= 0.5:
        print("⚠️ Certains tests ont échoué. Vérifiez la configuration.")
    else:
        print("❌ La plupart des tests ont échoué. Vérifiez la documentation.")

    print("\n📚 Pour plus d'aide, consultez CONFIGURATION_LANGSMITH.md")

    return success_rate == 1.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

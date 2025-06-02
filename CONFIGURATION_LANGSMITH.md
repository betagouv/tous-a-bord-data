# Configuration LangSmith pour l'évaluation HITL

## Variables d'environnement requises

Ajoutez ces variables à votre fichier `.env` :

```bash
# Configuration LangSmith pour l'évaluation HITL
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=transport-tarifs-pipeline
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

## Étapes de configuration

### 1. Créer un compte LangSmith

1. Allez sur [smith.langchain.com](https://smith.langchain.com)
2. Créez un compte ou connectez-vous
3. Créez un nouveau projet nommé `transport-tarifs-pipeline`

### 2. Obtenir la clé API

1. Dans LangSmith, allez dans Settings > API Keys
2. Créez une nouvelle clé API
3. Copiez la clé et ajoutez-la à votre `.env` comme `LANGCHAIN_API_KEY`

### 3. Vérifier la configuration

Une fois configuré, vous devriez voir :
- ✅ Connecté à LangSmith dans la sidebar de la page "📊 Évaluation HITL"
- Les runs LLM apparaître automatiquement dans LangSmith
- La possibilité de créer des feedbacks depuis l'interface Streamlit

## Fonctionnalités activées

### Tracking automatique
- Tous les appels LLM (Anthropic, Scaleway, Ollama) sont automatiquement trackés
- Les prompts, réponses, et métadonnées sont sauvegardés
- Les erreurs et temps d'exécution sont enregistrés

### Interface d'évaluation
- **Page dédiée** : `📊 Évaluation HITL` pour l'évaluation complète
- **Section intégrée** : Étape 6 dans la page principale d'évaluation
- **Métriques** : Qualité, Pertinence, Précision (scores 0-1)
- **Feedback** : Commentaires et corrections proposées

### Tableau de bord
- Statistiques de couverture d'évaluation
- Score moyen des évaluations
- Historique des évaluations avec filtres
- Recommandations d'amélioration

## Utilisation

### 1. Exécuter le pipeline
1. Sélectionnez une AOM
2. Exécutez les étapes 1-5 (scraping, filtrage, nettoyage, YAML)
3. Les appels LLM sont automatiquement trackés dans LangSmith

### 2. Évaluer les résultats
1. Allez dans l'Étape 6 ou la page dédiée "📊 Évaluation HITL"
2. Sélectionnez l'étape à évaluer (filtrage, nettoyage, ou YAML)
3. Donnez des scores et commentaires
4. Sauvegardez l'évaluation

### 3. Analyser les tendances
1. Utilisez la page "📊 Évaluation HITL" pour voir les statistiques
2. Identifiez les runs mal notés pour améliorer les prompts
3. Suivez l'évolution de la qualité dans le temps

## Avantages

### Amélioration continue
- **Détection de régressions** : Alertes si la qualité baisse
- **Optimisation des prompts** : Basée sur les feedbacks humains
- **Comparaison de modèles** : Évaluation objective des performances

### Traçabilité
- **Historique complet** : Tous les runs et évaluations sauvegardés
- **Reproductibilité** : Possibilité de rejouer les runs problématiques
- **Audit** : Traçabilité des décisions et améliorations

### Collaboration
- **Feedback partagé** : Équipe peut évaluer et commenter
- **Standards qualité** : Critères d'évaluation cohérents
- **Formation** : Nouveaux utilisateurs voient les bonnes pratiques

## Prochaines étapes

### Phase 2 : Stabilisation du scraping
- Remplacer crawl4ai par LangChain WebBaseLoader
- Comparer la qualité via le système d'évaluation

### Phase 3 : Interface de production
- API FastAPI pour re-processing automatique
- Dashboard LangSmith comme interface de production
- Validation/correction directe via LangSmith

## Dépannage

### Erreur de connexion LangSmith
- Vérifiez que `LANGCHAIN_API_KEY` est correctement définie
- Vérifiez que `LANGCHAIN_TRACING_V2=true`
- Testez la connexion avec `curl -H "Authorization: Bearer $LANGCHAIN_API_KEY" https://api.smith.langchain.com/info`

### Runs non visibles
- Vérifiez que `LANGCHAIN_PROJECT` correspond au nom du projet LangSmith
- Redémarrez l'application Streamlit après modification des variables d'environnement
- Vérifiez les logs pour les erreurs de tracking 
# tous-a-bord-data

Application d'accès aux données publiques des Autorités Organisatrices de la mobilité (ressort territorial, tarifs, critères d'éligibilité aux tarifs sociaux et solidaires).

## Prérequis

### Python

L'application nécessite Python 3.9.x. Pour vérifier votre version :

```bash
python --version  # Doit afficher Python 3.9.x
```


### Docker

L'application fonctionne avec Docker. Si vous ne l'avez pas déjà installé :

1. Installez Docker en suivant les instructions officielles pour votre système d'exploitation :

   - [Docker pour Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Docker pour Mac](https://docs.docker.com/desktop/install/mac-install/)
   - [Docker pour Linux](https://docs.docker.com/engine/install/)

2. Vérifiez que Docker est bien installé :

```bash
docker --version
```

## Configuration de l'environnement de développement

### Créer et activer un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate
# Sur Windows :
.\venv\Scripts\activate
```

### Copier les variables d'environnement :

```bash
cp .env.example .env
```

### Installer les outils de développement :

```bash
pip install -r requirements-dev.txt
```


## Outils de développement

### pre-commit

[Pre-commit](https://pre-commit.com/) permet de linter et formatter votre code avant chaque commit. Par défaut ici, il exécute :

- [black](https://github.com/psf/black) pour formatter automatiquement vos fichiers .py en conformité avec la PEP 8
- [flake8](https://github.com/pycqa/flake8) pour soulever les "infractions" restantes
- [isort](https://github.com/pycqa/isort) pour ordonner vos imports

Pour l'installer :

```bash
pre-commit install
```

Vous pouvez effectuer un premier passage sur tous les fichiers du repo avec :

```bash
pre-commit run --all-files
```

### commitzen pour formatter les commits

Commitizen est installé avec les autres outils de développement via `requirements-dev.txt`. Pour l'utiliser :

```bash
git add .
cz commit
# suivre les instructions du cli
```

## Lancement de l'application

### Option 1 : Exécution locale (sans Docker)

1. Activez votre environnement virtuel :
   ```bash
   source venv/bin/activate  # ou .\venv\Scripts\activate sur Windows
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   venv/bin/python -m spacy download fr_core_news_sm
   ```

3. Lancez l'application Streamlit :
   ```bash
   streamlit run main.py
   ```

4. Accédez à l'application dans votre navigateur à l'adresse [http://localhost:8501](http://localhost:8501)

### Option 2 : Exécution avec Docker

1. Construisez l'image Docker :
   ```bash
   docker build -t tous-a-bord-streamlit .
   ```

2. Lancez le conteneur :
   ```bash
   docker run -p 8501:8501 --env-file .env tous-a-bord-streamlit
   ```

   **Note importante :** Ne définissez pas la variable `PORT` dans votre fichier `.env` pour le développement local. Cette variable est réservée pour le déploiement sur Scalingo et sera automatiquement définie par la plateforme.

3. Accédez à l'application dans votre navigateur à l'adresse [http://localhost:8501](http://localhost:8501)

4. Pour arrêter le conteneur, utilisez `Ctrl+C` ou trouvez l'ID du conteneur avec `docker ps` puis exécutez `docker stop <container_id>`

## Étapes de configuration Langsmith

### 1. Créer un compte LangSmith

1. Allez sur [smith.langchain.com](https://smith.langchain.com)
2. Créez un compte ou connectez-vous
3. Créez un nouveau projet nommé `social-solidarity-transport-fares`

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
- Tous les appels LLM sont automatiquement trackés
- Les prompts, réponses, et métadonnées sont sauvegardés
- Les erreurs et temps d'exécution sont enregistrés
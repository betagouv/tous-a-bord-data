# tous-a-bord-data

Application d'accès aux données publiques des Autorités Organisatrices de la mobilité (ressort territorial, tarifs, critères d'éligibilité aux tarifs sociaux et solidaires).

## Prérequis

### Python

L'application nécessite Python 3.9.x. Pour vérifier votre version :

```bash
python --version  # Doit afficher Python 3.9.x
```

Si vous n'avez pas la bonne version :

- [Python pour Windows](https://www.python.org/downloads/windows/)
- [Python pour Mac](https://www.python.org/downloads/macos/)
- Pour Linux :
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa  # Pour Ubuntu
  sudo apt update
  sudo apt install python3.9
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
docker compose version
```

## Configuration de l'environnement de développement

1. Créer et activer un environnement virtuel :

```bash
python -m venv tab
source tab/bin/activate
# Sur Windows :
.\tab\Scripts\activate
```

2. Copier les variables d'environnement :

```bash
cp .env.example .env
```

3. Installer les outils de développement :

```bash
python -m pip install --no-user -r requirements-dev.txt
```


## Outils de développement

1. pre-commit

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

2. commitzen pour formatter les commits

Commitizen est installé avec les autres outils de développement via `requirements-dev.txt`. Pour l'utiliser :

```bash
git add .
cz commit
# suivre les instructions du cli
```

## Lancement de l'application

1. Démarrer l'application :

```bash
docker compose up
```

Ou en arrière-plan :

```bash
docker compose up -d
```

2. Accéder à l'application :

   - Ouvrez votre navigateur
   - Accédez à [http://localhost:8501](http://localhost:8501)

3. Arrêter l'application :
   - Si lancée en mode attaché : utilisez `Ctrl+C`
   - Si lancée en arrière-plan : `docker compose down`

## Commandes Docker utiles

1. Commandes de base

```bash
# Reconstruire les images
docker compose up --build

# Voir les logs
docker compose logs -f

# Voir l'état des conteneurs
docker compose ps

# Supprimer les conteneurs et les volumes
docker compose down -v
```

2. Gestion des volumes et des données PostgreSQL

```bash
# Lister tous les volumes Docker
docker volume ls

# Inspecter le volume PostgreSQL
docker volume inspect aom-postgres-data

# Supprimer un volume spécifique (attention: perte de données)
docker volume rm aom-postgres-data

# Supprimer tous les volumes non utilisés
docker volume prune

# Créer une sauvegarde (dump) de la base de données
docker exec -t postgres pg_dump -U ${POSTGRES_USER:-postgres} ${POSTGRES_DB:-postgres} > backup_$(date +%Y%m%d_%H%M%S).sql

# Restaurer une sauvegarde
cat backup_20230101_120000.sql | docker exec -i postgres psql -U ${POSTGRES_USER:-postgres} ${POSTGRES_DB:-postgres}
```

## Tests

1. Lancer les tests une fois

```bash
pytest app/tests/test_parser_utils.py -v
```

2. Ou en mode watch pour le TDD

```bash
pytest-watch app/tests/test_parser_utils.py
```

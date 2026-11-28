# SentinelFlow — DataRescue Banque

Plateforme de data engineering bancaire : prédiction ML des pannes,
diagnostic par règles + agent LLM encadré, réparation contrôlée en sandbox.

## Tableau de supervision local

Depuis la racine du projet, installe et compile l'interface React, puis démarre
FastAPI avec l'environnement Python du projet.

```powershell
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe -m uvicorn src.api.dashboard:app --host 127.0.0.1 --port 8000
```

Ouvre [DataRescue](http://127.0.0.1:8000/). Le tableau utilise les résultats
simulés de `data/prediction/xgb_detection` et la comparaison de
`data/prediction/model_comparison`. Les commandes de génération sont dans le
[guide de prédiction](src/prediction/README.md).

## Setup local (dev)
```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv sync


git clone https://github.com/<ton-user>/DataRescue.git
cd DataRescue
uv sync                    # recrée le venv exact depuis uv.lock
Copy-Item .env.example .env
docker compose up -d --build


## Données Delta
Les tables `data/delta/` ne sont **pas versionnées** dans Git.
Après un `git clone`, régénérez-les avec les scripts Bronze/Silver.

# SentinelFlow — DataRescue Banque

Plateforme de data engineering bancaire : prédiction ML des pannes,
diagnostic par règles + agent LLM encadré, réparation contrôlée en sandbox.

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

daily_account_summary.

Ce qu'elle contient
Pour chaque account_id et chaque jour, on calcule :

total_debits : somme des montants négatifs (en valeur absolue)

total_credits : somme des montants positifs

net_flow : total_credits - total_debits

transaction_count : nombre de transactions

C'est la table métier de base pour le reporting et pour la réconciliation.


gold_reconciliation_check.

Ce qu'elle contient
Pour chaque jour, on calcule :

total_debits : somme de tous les montants négatifs (en valeur absolue)

total_credits : somme de tous les montants positifs

difference : total_credits - total_debits

is_reconciled : booléen (true si difference == 0, sinon false)

C'est cette table qui alimente la règle R7 du cahier des charges : "total débits ≠ total crédits → escalade obligatoire". Elle sera utilisée dans le scénario de démo "reprise refusée".
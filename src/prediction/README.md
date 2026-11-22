# Prédiction — étape 1 : historique synthétique

Ce module prépare les données prévues à la section 6 du cahier des charges.
Il n'entraîne pas encore de modèle. Il utilise uniquement la bibliothèque
standard Python ; aucun service Airflow, Spark ou Docker n'est nécessaire.

Depuis la racine du dépôt, dans PowerShell :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_prediction_history.py -v
.\.venv\Scripts\python.exe -m src.prediction.generate_history --days 365 --seed 42
Import-Csv data/prediction/history.csv | Select-Object -First 5 | Format-Table
Import-Csv data/prediction/history.csv | Group-Object failed | Select-Object Name,Count
```

Attendu : cinq tests réussis et 3 650 lignes (365 jours × 10 tâches).
Le CSV est ignoré par Git et reproductible à paramètres identiques.
Un fichier existant n'est jamais écrasé ; utiliser `--output` pour un autre essai :

```powershell
.\.venv\Scripts\python.exe -m src.prediction.generate_history --days 30 --seed 7 --output data/prediction/essai.csv
```

## Contrat temporel

Une ligne représente une tâche à une date donnée. Les variables d'entrée sont
observées avant son lancement, puis `failed` indique le résultat de cette
exécution imminente : 1 pour panne et 0 pour succès. Il ne faut donc pas décaler
à nouveau cette cible vers le lendemain.

- `FEATURE_COLUMNS` liste les seules colonnes candidates à l'entraînement.
- Les colonnes `previous_*` proviennent de la veille pour la même tâche.
- `volume_mean_7d` et `duration_trend_7d_s` utilisent jusqu'à sept jours antérieurs.
- `volume_delta_pct` compare le volume entrant à cette moyenne passée.
- Les valeurs historiques absentes au démarrage restent vides, pas remplacées
  par des observations futures. L'imputation sera ajustée sur l'entraînement seul.
- `observed_*`, `failed` et `simulated_cause` sont des résultats, jamais des features.
- `date` sert au découpage chronologique ; `run_id` sert à la traçabilité.

Le volume, le taux de nulls et le retard courants supposent un contrôle d'entrée
avant chaque tâche. Ils ne sont pas nécessairement disponibles lors d'un scoring
global avant le démarrage du DAG : il faudra alors les remplacer par des estimations
ou des mesures historiques.

## Hypothèses et limites

Les dix profils correspondent aux fonctions prévues dans le CDC ; seuls les noms
des tâches Bronze et Silver sont déjà alignés sur le DAG actuel. Les autres noms,
capacités mémoire et durées sont des hypothèses de simulation.

Le jour de paie est fixé au 25 et la fin de mois au dernier jour calendaire.
La charge évolue dans le temps. Les pannes sont tirées probabilistiquement selon
la pression mémoire, la qualité et le retard, avec du bruit. `simulated_cause`
désigne le facteur dominant de la simulation, pas une cause racine vérifiée par
une injection réelle. Aucune propagation des pannes ni tâche ignorée en aval
n'est simulée. Cet historique ne remplace pas le benchmark des scénarios injectés.

Le générateur est une implémentation propre à DataRescue. Le projet
[pipeline-failure-predictor](https://github.com/sethneh/pipeline-failure-predictor)
a servi de référence conceptuelle pour la simulation probabiliste ; son CSV
n'est pas importé. Les performances futures sur ces données ne démontreront pas
une efficacité sur un pipeline réel.

## Étapes suivantes, en commits séparés

1. Préparer les ensembles chronologiques entraînement/validation/test et la baseline de seuils.
2. Entraîner et comparer Random Forest, XGBoost et Isolation Forest.
3. Évaluer les alertes (PR-AUC, précision/rappel à budget fixé, F1, faux positifs).
4. Ajouter les explications SHAP et la sauvegarde du modèle retenu.
5. Intégrer le scoring avant les tâches et journaliser les prédictions.

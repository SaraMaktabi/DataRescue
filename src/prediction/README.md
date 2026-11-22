# Prédiction — historique synthétique et baseline

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

## Étape 2 : découpage chronologique et baseline

Après génération de l'historique, depuis la racine du dépôt :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_prediction_*.py" -v
.\.venv\Scripts\python.exe -m src.prediction.baseline --input data/prediction/history.csv --output-dir data/prediction/baseline
Get-Content data/prediction/baseline/report.json
Import-Csv data/prediction/baseline/validation_predictions.csv | Where-Object alert -eq '1' | Select-Object -First 5
```

Le dossier de sortie doit être nouveau (utiliser `--output-dir` pour refaire un
essai). Les résultats sont ignorés par Git. Pour l'historique de 365 jours :

| Ensemble | Jours | Lignes | Utilisation |
| --- | ---: | ---: | --- |
| train | 255 | 2 550 | Futur entraînement |
| validation | 54 | 540 | Comparaison et choix des paramètres |
| test | 56 | 560 | Réservé à l'évaluation finale |

Les proportions 70 % et 15 % sont arrondies vers le bas en nombre de jours.
Toutes les tâches d'une date restent ensemble. Les CSV exportés excluent
`observed_*` et `simulated_cause` ; `failed` reste la cible et doit être séparée
des entrées lors de l'entraînement (utiliser `FEATURE_COLUMNS`).

Les caractéristiques historiques peuvent inclure les résultats des jours précédents
de validation ou de test : c'est une évaluation quotidienne où ces résultats sont
déjà connus. Ce n'est pas une prévision de toute la période en une seule fois.
Aucune imputation ni normalisation n'est ajustée à cette étape.

Une alerte est déclenchée si au moins un seuil est atteint : nulls >= 5 %,
retard >= 30 minutes, écart absolu de volume >= 50 %, mémoire de la veille >= 90 %
de la capacité courante. Ce sont des hypothèses initiales fixées dans le code,
pas des seuils optimisés. Une valeur historique absente ne déclenche pas la règle
correspondante ; les contrôles d'entrée restent actifs.

Le rapport évalue uniquement la validation : matrice de confusion, précision,
rappel, F1, taux de fausses alertes `FP / (FP + TN)` et proportion d'alertes.
Une métrique dont le dénominateur est nul vaut `null`. La référence `never_alert`
représente un système qui n'émet jamais d'alerte. Les raisons des alertes sont
exportées pour inspection. La baseline produit une décision, pas une probabilité.

Cette étape ne calcule pas encore la PR-AUC, le coût métier ou les métriques à
budget d'alertes fixé ; ils seront ajoutés pour la comparaison des modèles.
Ne pas utiliser le test pour modifier les seuils.

## Étape 3 : Random Forest

On conserve l'historique existant de 3 650 lignes. Le modèle est entraîné sur
les 2 550 premières observations et évalué uniquement sur les 540 observations
de validation. Le test reste réservé.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_prediction_*.py" -v
.\.venv\Scripts\python.exe -m src.prediction.train_random_forest --output-dir data/prediction/rf_mon_test
Get-Content data/prediction/rf_mon_test/report.json
Import-Csv data/prediction/rf_mon_test/validation_predictions.csv | Select-Object -First 5
```

Le dossier doit être nouveau. Les dépendances scikit-learn, pandas et joblib
(dépendance de scikit-learn) doivent être disponibles dans l'environnement.
Les tests comprennent désormais cinq vérifications supplémentaires pour le ML.

La médiane pour les valeurs manquantes et l'encodage des tâches sont appris
exclusivement sur l'entraînement. Le modèle sauvegardé dans `model.joblib`
inclut ce prétraitement. Ne charger que des fichiers joblib de confiance.
L'empreinte du CSV, la version scikit-learn, les variables, les paramètres et les
périodes sont enregistrés dans le rapport pour reproduire l'expérience.

Le rapport compare les seuils fixes, Random Forest au seuil 0,5 et Random Forest
en sélectionnant autant d'alertes que la baseline sur la validation. Ce dernier
classement top-k est rétrospectif, pas une politique de déploiement. Les égalités
sont résolues par l'ordre chronologique puis l'identifiant de tâche.

`average_precision` résume la courbe précision-rappel sans interpolation ; ce
n'est pas une aire trapézoïdale. La proportion de pannes est aussi fournie comme
repère. Les sorties du modèle ne sont pas calibrées et ne doivent pas être
interprétées comme des probabilités fiables en production. Aucun seuil n'est
optimisé sur le test. SHAP, XGBoost et Isolation Forest restent à ajouter.

## Étapes suivantes, en commits séparés

1. Historique, découpage chronologique et baseline : implémentés.
2. Random Forest : implémenté ; ajouter XGBoost et Isolation Forest.
3. Évaluer les alertes (PR-AUC, précision/rappel à budget fixé, F1, faux positifs).
4. Ajouter les explications SHAP et la sauvegarde du modèle retenu.
5. Intégrer le scoring avant les tâches et journaliser les prédictions.

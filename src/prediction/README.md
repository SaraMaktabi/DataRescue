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

## Étape 4 : XGBoost

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_prediction_*.py" -v
.\.venv\Scripts\python.exe -m src.prediction.train_xgboost --output-dir data/prediction/xgb_mon_test
Get-Content -Encoding utf8 data/prediction/xgb_mon_test/report.json
```

Attendu : 19 tests réussis. Le dossier de sortie doit être nouveau. Le module
réutilise le découpage, les variables, le prétraitement et les métriques de
Random Forest. La pondération `scale_pos_weight` est le rapport succès/pannes
sur l'entraînement uniquement. Les paramètres initiaux sont fixes : 200 arbres,
profondeur 3, learning rate 0,05. Aucune recherche de paramètres ni early stopping
n'utilise la validation ou le test à cette étape.

Le rapport conserve l'empreinte des données, les périodes et les versions de
scikit-learn et XGBoost. Comparer seulement des expériences ayant la même empreinte
et les mêmes périodes. Le modèle sauvegardé inclut son prétraitement.

Les métriques `xgboost_threshold_0_5` et `xgboost_same_alert_count` ont les mêmes
définitions que celles de Random Forest. Le budget commun reste celui des alertes
de la baseline sur la validation, pour une comparaison rétrospective. Il ne
constitue pas encore un budget opérationnel fixé à l'avance. Le test reste réservé.

## Réglage du seuil sur la validation

Le modèle reste inchangé. Cette commande utilise ses prédictions sauvegardées :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_prediction_*.py" -v
.\.venv\Scripts\python.exe -m src.prediction.tune_threshold --model-dir data/prediction/xgb_mon_test --output-dir data/prediction/xgb_seuil_mon_test --max-alert-rate 0.15
Get-Content -Encoding utf8 data/prediction/xgb_seuil_mon_test/threshold.json
```

Les 23 tests couvrent notamment les scores ex aequo, le budget nul et le refus
d'un rapport de test. Le dossier de sortie doit être nouveau. La commande fonctionne
aussi avec un dossier d'expérience Random Forest.

Le budget de 15 % est une hypothèse expérimentale modifiable, pas une exigence du
CDC. Sur 540 observations, il autorise au maximum 81 alertes. Tous les seuils
correspondant aux scores distincts sont comparés, ainsi que 0,5 et un seuil sans
alerte. Sélection : rappel maximal sous budget, puis moins de faux positifs,
puis seuil le plus élevé. Les scores égaux sont toujours traités ensemble.

`threshold_comparison.csv` contient les métriques par seuil ; `threshold.json`
enregistre le seuil retenu, le budget, les empreintes du modèle et des données.
La règle sauvegardée est `score >= threshold`. Un seuil supérieur à 1 signifie
aucune alerte. Le modèle n'est ni modifié ni réentraîné.

Ces résultats sont calculés sur la validation qui sert à choisir le seuil : ils
ne constituent pas une évaluation indépendante. Le taux d'alertes futur n'est pas
garanti et l'Average Precision ne change pas. L'évaluation finale devra appliquer
le seuil figé au test sans le réajuster.

## Recherche des paramètres XGBoost

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_prediction_*.py" -v
.\.venv\Scripts\python.exe -m src.prediction.search_xgboost --output-dir data/prediction/xgb_search_mon_test
Get-Content -Encoding utf8 data/prediction/xgb_search_mon_test/search_report.json
Get-Content -Encoding utf8 data/prediction/xgb_search_mon_test/report.json
```

Attendu : 27 tests réussis. Le dossier de sortie doit être nouveau. La recherche
compare 12 configurations sur trois fenêtres croissantes au sein des 255 jours
d'entraînement uniquement. Toutes les tâches d'un jour restent ensemble.
Chaque prétraitement et chaque pondération sont ajustés à nouveau sur le passé
de la fenêtre. Les observations de validation interne suivent ce passé.

Douze candidats combinent quatre couples (nombre d'arbres, profondeur) et trois
pondérations : ratio succès/pannes, racine du ratio, aucune pondération. Le modèle
initial fait partie des candidats. L'option expérimentale `--include-task-load`
ajoute trois candidats qui utilisent le
volume courant rapporté au volume médian de la tâche sur la fenêtre d'entraînement,
la variation récente de volume et la pression mémoire précédente normalisée.
Leurs scores sont contraints à augmenter ou rester stables quand le retard,
le taux de valeurs manquantes ou le volume relatif augmente, à entrées égales.
La moyenne non pondérée des trois Average
Precision choisit le candidat ; une égalité conserve le premier. Aucun early
stopping ni choix de paramètres ne consulte la validation externe ou le test.

Le candidat retenu est réentraîné sur tout l'entraînement, puis évalué sur la
validation externe. `search_report.json` décrit les fenêtres, tous les candidats
et les scores internes. `report.json`, `model.joblib` et les prédictions suivent
le format habituel. Une recherche peut ne pas améliorer la validation externe.
Les historiques utilisés supposent un scoring quotidien avec les résultats
des jours précédents disponibles, comme pour les autres expériences.

Pour régler ensuite le seuil du nouveau modèle (les scores ont changé) :

```powershell
.\.venv\Scripts\python.exe -m src.prediction.tune_threshold --model-dir data/prediction/xgb_search_mon_test --output-dir data/prediction/xgb_search_seuil_mon_test --max-alert-rate 0.15
```

Le test final reste réservé. Ne pas multiplier les recherches en fonction des
résultats de validation au risque de surajuster cette période.

## Détection des pannes avec seuil validé

```powershell
.\.venv\Scripts\python.exe -m src.prediction.search_xgboost --input data/prediction/mon_test.csv --output-dir data/prediction/xgb_detection --evaluate-test
```

Le seuil maximise le rappel sur la validation dans le budget d'alertes
(`--max-alert-rate`, 15 % par défaut). Le test chronologique optionnel applique
ce seuil figé et exporte les décisions dans `test_predictions.csv`.
Pour lire les alertes, utiliser `selected_alert`. Le seuil correspondant se
trouve dans `operating_point.threshold` dans `report.json`.
Comparer le rappel, le F1 et les faux positifs avec l'accuracy, qui peut être
élevée simplement parce que les succès sont majoritaires.

## Expliquer les alertes XGBoost avec SHAP

```powershell
.\.venv\Scripts\python.exe -m src.prediction.explain_xgboost --input data/prediction/mon_test.csv --model-dir data/prediction/xgb_detection --output data/prediction/xgb_detection/shap_alerts.csv
```

Le CSV présente la panne détectée et la fausse alerte de score le plus élevé,
puis l'alerte restante la mieux classée. Il donne jusqu'à trois facteurs qui
augmentent le score, classés par contribution. Les valeurs SHAP sont exprimées
en log-odds, sur l'échelle interne de XGBoost. Elles décrivent le calcul du
modèle. Le seuil et les prédictions de l'expérience restent ceux du rapport.

## Comparer Random Forest, XGBoost et Isolation Forest

```powershell
.\.venv\Scripts\python.exe -m src.prediction.compare_models --input data/prediction/mon_test.csv --output-dir data/prediction/model_comparison
```

Le dossier de sortie doit être nouveau. Les trois modèles utilisent les mêmes
périodes chronologiques. Isolation Forest apprend sur les variables d'entrée
sans labels de panne. Son score d'anomalie augmente quand l'exécution paraît
plus inhabituelle. Les labels servent au choix du seuil sur la validation et
au calcul des résultats.

Chaque seuil respecte le budget `--max-alert-rate` sur la validation
(15 % par défaut). Il est ensuite appliqué au test. Le taux d'alertes observé
sur le test peut différer d'un modèle à l'autre. Le fichier `comparison.csv`
regroupe les résultats et chaque sous-dossier contient le modèle, son rapport
et ses prédictions.

## Étapes suivantes, en commits séparés

1. Historique, découpage chronologique et baseline : implémentés.
2. Random Forest, XGBoost et Isolation Forest : implémentés.
3. Évaluer les alertes (PR-AUC, précision/rappel à budget fixé, F1, faux positifs).
4. Étendre les explications SHAP au scoring intégré.
5. Intégrer le scoring avant les tâches et journaliser les prédictions.

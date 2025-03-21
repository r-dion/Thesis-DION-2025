Ce Git contient les scripts Python nécessaires pour la reproduction des évaluations expérimentales de la thèse.

## Chapitre 3
Les fichiers associés se trouvent au sein du dossier "Chapter3". Les fichiers les plus importants sont :
- thresholders.py : Contient le script Python des différentes méthodes de seuillage explorées
- demo_thresholds.ipynb : Présente un exemple d'utilisation de ces seuils

Les sorties d'algorithmes utilisées pour l'évaluation expérimentale des méthodes de seuillages se trouvent aux adresses suivantes :
- Attracteur de Lorenz : https://drive.google.com/file/d/1aD5RPfjdcMEkdbFaEjJQpDUqdanoksrA/view?usp=sharing
- Boîtier papillon : https://drive.google.com/file/d/1lyhbmqLnwW_zenCG0M8ReyKEKLJAhiN8/view?usp=sharing

Les fichiers "lorenz_allmethods_relative_f1score.csv" et "etc_allmethods_absolute_f1score.csv" contient les résultats.

## Chapitres 4 & 5
Les fichiers associés se trouvent au sein du dossier "Chapters4-5". Les fichiers les plus importants sont :
- mti.py : Contient le script Python de la métrique MTI
- auto_param.py : Contient le script Python du module d'auto-paramétrisation
- rbo.py : Contient une implémentation de la méta-métrique RBO
- demo_metrics.ipynb : Présente un exemple d'utilisation de ces métriques, ainsi que le protocole suivi pour les expérimentations.

Les prédictions synthétiques utilisées pour les évaluations expérimentales sont disponibles dans les dossiers "Chap4_SyntheticExamples" et "Chap5_SyntheticCollections". 
Les résultats de l'évaluation du module d'auto-paramétrisation sont disponibles au sein du dossier "Results_Autoparam".
Le dossier "Chap4_other_metrics" contient les implémentations associées aux métriques de la littérature évaluées, c'est-à-dire l'Affiliation, l'eTa, le VUS et le PATE.

## Chapitre 6 & 7

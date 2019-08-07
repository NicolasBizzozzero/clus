# clustering
Clustering and fuzzy-clustering library with integrated CLI.

## Installation
```shell
$ python3 setup.py install
```
Running the `setup.py` script will download all required packages and install the following commands either in your `~/.local/bin/` for local installation or in the `/usr/lib/pythonX.X` directories : 
* clus : Main command for partition-based clustering.
* dclus : density-based clustering.
* hclus : Hierarchical clustering.
* eclus : Evaluate clustering. Implementing diferents way to evaluate clustering results by comparing two partitions.


## Usage
```shell
# Simple clustering with 500-components kmeans algorithm
$ clus dataset.csv kmeans -k 500 --max-iter 100  --eps 0.001 --header --save-clus

# Clustering with a weighted euclidean distance + matplotlib visualisation
$ clus dataset.npy kmeans --pairwise-distance weighted_euclidean --weights 1 0 1 1 --visualise

# Clustering from a .npz file with a 3D matplotlib visualisation
$ clus dataset.npz fcm --array-name clusters_center --visualise-3d

# Clustering with automatic dataset filetype guessing and with a normalization beforehand 
$ clus dataset kmeans --file-type guess --normalization rescaling --save-clus

# DBSCAN clustering with a custom clustering results saving path
$ dclus dataset.csv dbscan --save-clus --seed 1 --format-filename-dest-results dbased_{clustering_algorithm}_{dataset}_{seed}
```

## TODO
* Pour les fcm, plusieurs copies de données sont faites plusieurs fois. Optimisable, voir : https://pythonhosted.org/scikit-fuzzy/_modules/skfuzzy/cluster/_cmeans.html#cmeans
* Rendre les opérations de clustering inplace (ne pas dupliquer les données)
* Méthodes de Clustering à implémenter :
  * https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/cluster/k_means_.py#L1318
  * OPTICS
  * Mean-Shift
  * Vérifier le hard_c_medoids
* La loss n'est pas une mesure efficace pour comparer des clusterings à differentes tailles de composantes. Implémenter des méthodes d'évaluations :
  * Implementer silhouette (https://en.wikipedia.org/wiki/Silhouette_(clustering)) dans eclus
* Implementer une detection automatique de header si le flag --header n'est pas passé. Voir : https://stackoverflow.com/questions/32312309/check-if-header-exists-with-python-pandas
* Permettre au HC de prendre plusieurs paramètres pour fcluster values. Si plusieurs, alors on applique ces params. Ajouter option fcluster_value dans le filename.
* Les opérations sur medoids sont lentes. S'inspirer d'autres codes pour les ameliorer :
  * https://github.com/vitordeatorreao/amproj/blob/844842a532524d5f514e7534da65c7a64ab7ef57/amproj/distance/kmedoids.py
  * https://github.com/agartland/utils/blob/e200d7e41039ca0053bd817c1d1857aab33bd503/kmedoids.py#L172
  * https://github.com/Brain-Mapper/BrainMapperV2/blob/63075bdca2428197fc18a1cf6c7403c2764e0664/ourLib/clustering.py#L293
* Application du decorateur "memoryerror_fallback".
* Au lieu de fixer un nom de fichier par défaut, mettre tous les paramètres par défaut à None (et fixer en dur leurs valeus par défaut dans la doc, ou formater selon un fichier), et ajouter les parametres NON None passés par l'utilisateur au nom de fichier selon un ordre prédéfini. L'utilisateur peut aussi très bien passer son nom de fichier prédéfini.
* Implémenter un algo d'appariemment (de comparaison) d'affectations. Peut être utiliser https://fr.wikipedia.org/wiki/Algorithme_hongrois
* Revoir les séparations en sous-commandes. Peut être faire : "clus evaluate" ou "clus visualise".
* Utiliser une mesure de distance chunked : sklearn.metrics.pairwise_distances_chunked (peut être dans le decorateur ?)
* S'inspirer de l'API d'autres clustering softwares :
** https://elki-project.github.io
** https://www.cs.waikato.ac.nz/ml/weka
** https://www.knime.com/nodeguide/analytics/clustering


## References
* [1] R. Krishnapuram ; A. Joshi ; O. Nasraoui ; L. Yi, Low-complexity fuzzy relational clustering algorithms for Web mining,  IEEE Transactions on Fuzzy Systems (Volume: 9, Issue: 4, Aug 2001), p595-607, DOI: 10.1109/91.940971
* [2] Tolga Can, K-Means Empty Cluster Example, http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html (archive: https://web.archive.org/web/20180626092955/http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html)
* [3] Ross, Timothy J., Fuzzy Logic With Engineering Applications, 3rd ed. Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353
* [4] Joe Marino, statistical whitening, http://joelouismarino.github.io/blog_posts/blog_whitening.html (archive: https://web.archive.org/web/20180813034201/http://joelouismarino.github.io/blog_posts/blog_whitening.html)
* [5] TODO: Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into data weighted by the square root of the weights
* [6] James C.Bezdek, Robert Ehrlich, William Full, FCM: The fuzzy c-means clustering algorithm, https://doi.org/10.1016/0098-3004(84)90020-7, 1983

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
* Implementer une detection automatique de header si le flag --header n'est pas passé. Voir : https://stackoverflow.com/questions/32312309/check-if-header-exists-with-python-pandas
* Permettre au HC de prendre plusieurs paramètres pour fcluster values. Si plusieurs, alors on applique ces params. Ajouter option fcluster_value dans le filename.
* Les opérations sur medoids sont lentes. S'inspirer d'autres codes pour les ameliorer :
  * https://github.com/vitordeatorreao/amproj/blob/844842a532524d5f514e7534da65c7a64ab7ef57/amproj/distance/kmedoids.py
  * https://github.com/agartland/utils/blob/e200d7e41039ca0053bd817c1d1857aab33bd503/kmedoids.py#L172
  * https://github.com/Brain-Mapper/BrainMapperV2/blob/63075bdca2428197fc18a1cf6c7403c2764e0664/ourLib/clustering.py#L293
* Application du decorateur "memoryerror_fallback", d'un parametre de taille de chunk, puis utiliser sklearn.metrics.pairwise_distances_chunked
* Au lieu de fixer un nom de fichier par défaut, mettre tous les paramètres par défaut à None (et fixer en dur leurs valeus par défaut dans la doc, ou formater selon un fichier), et ajouter les parametres NON None passés par l'utilisateur au nom de fichier selon un ordre prédéfini. L'utilisateur peut aussi très bien passer son nom de fichier prédéfini.
* Implémenter un algo d'appariemment (de comparaison) d'affectations. Peut être utiliser https://fr.wikipedia.org/wiki/Algorithme_hongrois
* Revoir les séparations en sous-commandes. Peut être faire : "clus evaluate" ou "clus visualise".
* S'inspirer de l'API d'autres clustering softwares :
** https://elki-project.github.io
** https://www.cs.waikato.ac.nz/ml/weka
** https://www.knime.com/nodeguide/analytics/clustering
* Implémenter la lecture de fichiers :
** memmap : https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
* S'inspirer des codes suivants :
** https://github.com/overshiki/kmeans_pytorch/blob/master/kmeans.py
* Etudier mesures :
** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html 
** https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivariate_normal.html
** https://davetang.org/muse/2017/09/21/adjusted-rand-index/
* Trouver meilleure coupe HC :
** https://www.sciencedirect.com/science/article/pii/S0031320310001974
* Etudier d'autres méthodes de normalisation, leurs effets et leurs buts
** https://en.wikipedia.org/wiki/Normalization_(statistics)
** https://en.wikipedia.org/wiki/Feature_scaling
** https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
* Refactoring de l'architecture du code:
** clus.methods, qui offre toutes les méthodes depuis ce module.
** clus.preprocessing, qui offre les méthodes de normalisation, et de pondération
** clus.visualisation
** clus.evaluation
** clus.loading, toutes les méthodes de chargement des données
* Changer la façon dont les paramètres du clustering sont stockés et dont les résultats sont sauvegardés.
** Virer tous les params liés au nom de fichier. Ne garder que "file-name-prefix", et ajouter un UUID comme nom de fichier.
** Stocker tous les paramètres de clustering dans le dictionnaire résultats. clef "params".
** Maintenant que j'écris ça, je me rends compte que sacred serait peut etre plus approprié pour la gestion des résultats.
* Virer silhouette, ne garder que silhouette-samples, puis retourner la moyenne (qui correspond à silhouette-score), on lapelle silhouette_mean, et l'écart type silhouette_std.
* Etudier l'utilisation de : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_chunked.html
* Remettre les références dans la doc de chaque module. Les retirer des références ci-dessous.
* Essayer de proposer une implémentation de k-means utilisant CUDA : http://alexminnaar.com/2019/03/05/cuda-kmeans.html
* Ajouter param UUID : ajouter un param possible, {uuid4}, {uuid6}, etc dans le futur nom de fichier peut etre ? Regarder si on peut l'initialiser avec le temps, sinon faudra faire gaffe à l'initialisation avec graine aléatoire, sinon c'est possiblkz que je retourne les mêmes uuid à chaque fois.
* Ajouter des mesures d'evaluation : Papier pour mesures d'eval clustering : https://amstat.tandfonline.com/doi/abs/10.1080/01621459.1971.10482356#.XXZYe6WxXRY
* Etudier d'autres normalisations et leurs effets : https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#results


## References
* [1] R. Krishnapuram ; A. Joshi ; O. Nasraoui ; L. Yi, Low-complexity fuzzy relational clustering algorithms for Web mining,  IEEE Transactions on Fuzzy Systems (Volume: 9, Issue: 4, Aug 2001), p595-607, DOI: 10.1109/91.940971
* [2] Tolga Can, K-Means Empty Cluster Example, http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html (archive: https://web.archive.org/web/20180626092955/http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html)
* [3] Ross, Timothy J., Fuzzy Logic With Engineering Applications, 3rd ed. Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353
* [4] Joe Marino, statistical whitening, http://joelouismarino.github.io/blog_posts/blog_whitening.html (archive: https://web.archive.org/web/20180813034201/http://joelouismarino.github.io/blog_posts/blog_whitening.html)
* [5] TODO: Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into data weighted by the square root of the weights
* [6] James C.Bezdek, Robert Ehrlich, William Full, FCM: The fuzzy c-means clustering algorithm, https://doi.org/10.1016/0098-3004(84)90020-7, 1983

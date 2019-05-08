# clustering
## Useful links
* http://cs.joensuu.fi/sipu/datasets


## TODO
* Pour les fcm, plusieurs copies de données sont faites plusieurs fois. Optimisable, voir : https://pythonhosted.org/scikit-fuzzy/_modules/skfuzzy/cluster/_cmeans.html#cmeans
* Rendre les opérations de clustering inplace (ne pas dupliquer les données)
* Méthodes de Clustering à implémenter :
  * https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/cluster/k_means_.py#L1318
  * DBSCAN
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


## References
* [1] R. Krishnapuram ; A. Joshi ; O. Nasraoui ; L. Yi, Low-complexity fuzzy relational clustering algorithms for Web mining,  IEEE Transactions on Fuzzy Systems (Volume: 9, Issue: 4, Aug 2001), p595-607, DOI: 10.1109/91.940971
* [2] Tolga Can, K-Means Empty Cluster Example, http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html (archive: https://web.archive.org/web/20180626092955/http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html)
* [3] Ross, Timothy J., Fuzzy Logic With Engineering Applications, 3rd ed. Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353
* [4] Joe Marino, statistical whitening, http://joelouismarino.github.io/blog_posts/blog_whitening.html (archive: https://web.archive.org/web/20180813034201/http://joelouismarino.github.io/blog_posts/blog_whitening.html)
* [5] TODO: Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into data weighted by the square root of the weights
* [6] James C.Bezdek, Robert Ehrlich, William Full, FCM: The fuzzy c-means clustering algorithm, https://doi.org/10.1016/0098-3004(84)90020-7, 1983

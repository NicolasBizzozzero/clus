# clustering
## Useful links
* http://cs.joensuu.fi/sipu/datasets


## TODO
* Rendre les opérations de clustering inplace (ne pas dupliquer les données)
* Méthodes de Clustering à implémenter :
  * https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/cluster/k_means_.py#L1318
  * Vérifier le hard_c_medoids
* La loss n'est pas une mesure efficace pour comparer des clusterings à differentes tailles de composantes. Implémenter des méthodes d'évaluations :
  * silhouette (https://en.wikipedia.org/wiki/Silhouette_(clustering))
  * ARI (https://en.wikipedia.org/wiki/Rand_index, https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=4&ved=2ahUKEwiPyoOkvuvgAhUHuRoKHf_aAggQFjADegQIBxAB&url=https%3A%2F%2Fdavetang.org%2Fmuse%2F2017%2F09%2F21%2Fadjusted-rand-index%2F&usg=AOvVaw2O53qsxo75pv4xn8HfHBin)
* Permettre au HC de prendre plusieurs paramètres pour fcluster values. Si plusieurs, alors on applique ces params. Ajouter option fcluster_value dans le filename.


## References
* [1] R. Krishnapuram ; A. Joshi ; O. Nasraoui ; L. Yi, Low-complexity fuzzy relational clustering algorithms for Web mining,  IEEE Transactions on Fuzzy Systems (Volume: 9, Issue: 4, Aug 2001), p595-607, DOI: 10.1109/91.940971
* [2] Tolga Can, K-Means Empty Cluster Example, http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html (archive: https://web.archive.org/web/20180626092955/http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html)
* [3] Ross, Timothy J., Fuzzy Logic With Engineering Applications, 3rd ed. Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353
* [4] Joe Marino, statistical whitening, http://joelouismarino.github.io/blog_posts/blog_whitening.html (archive: https://web.archive.org/web/20180813034201/http://joelouismarino.github.io/blog_posts/blog_whitening.html)
* [5] TODO: Applying weighted euclidean distance is equivalent to applying traditional euclidean distance into data weighted by the square root of the weights
* [6] James C.Bezdek, Robert Ehrlich, William Full, FCM: The fuzzy c-means clustering algorithm, https://doi.org/10.1016/0098-3004(84)90020-7, 1983

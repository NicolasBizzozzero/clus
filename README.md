# clustering
## Cool dataset
* http://cs.joensuu.fi/sipu/datasets


## TODO
* https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc
* https://github.com/holtskinner/PossibilisticCMeans
* Ne pas faire la PCA sur les données utilisant des matrices des distance, revoir le code
* Detecter probleme du fuzzifier (range ? 2 ou plus ?)
* Voir pourquoi la fonction des lfcmdd est lente (tps à chaque étape, peut etre améliorer en sauvegardant la matrice là)
* Etudier silhouette (https://en.wikipedia.org/wiki/Silhouette_(clustering)) et ARI (https://en.wikipedia.org/wiki/Rand_index, https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=4&ved=2ahUKEwiPyoOkvuvgAhUHuRoKHf_aAggQFjADegQIBxAB&url=https%3A%2F%2Fdavetang.org%2Fmuse%2F2017%2F09%2F21%2Fadjusted-rand-index%2F&usg=AOvVaw2O53qsxo75pv4xn8HfHBin)
* Mettre les valeurs par défaut dans la doc.
* Comment gérer les 0 (petit epsilon partout ou juste sur les 0) ou même supprimer les doublons ? MJ a dit de tester voir si j'obtiens les memes resultats (a epsilon pres) en ajoutant epsilon ou alors en mettant à 0 les valeurs De u_ij pour des exemples qui sont égaux. Ne surtout pas supprimer les doublons. Un cluster avec plein d'exemples au même endroit aura plus de force qu'un cluster avec un seul exemple.
* Est-ce que je normalise mes données ? Ne pas faire de normalisation centrée-réduite sur un même attribut à cause des outliers. On peut cependant faire une normalisation entre les attributs de manière à ce qu'ils soient dans le même intervalle de valeurs, et qu'un attribut ne soit pas plus fort qu'un autre. Adam me parlait aussi de normalisation avec l'écart type ??
* La matrice de dissimilarité, on la fournie ou je dois la calculer ? La dissimilarité est symétrique ? La dissimilarité n'est pas forcement symétrique, mais on peut suppose qu'elle l'est pour pouvoir simplifier des calculs.
* Après discussion avec MJ et Adrien, la loss peut remonter. Il arrive qu'elle "rate" le minimum local et remonte un peu, mais bien souvent cette différence est négligeable. En effet, la loss va converger et chaque mise à jour ne la fera varier que très peu. Il arrive aussi que la loss se remette à décroite après être remontée un peu. MJ Explique ce phénomène comme un "trop de choix" pour l'algorithme dans un grand nombre de clusters, et il arrive que la loss fasse des variations comme ça
* Comme expliqué ci-dessus, la loss peut remonter. Pour palier à ça, sci-kit learn et numpy comparent et stockent les meilleurs clusters à chaque iteration. On pourra surement faire ça et comparer les performances qu'on obtient. Normalement la difference de loss entre le clustering obtenu et le clustering max devrait etre negligeable.
* Lister tous les algos disponibles et leurs acronymes. Dans chaque option, lister pour quels acronymes elles sont disponibles (ou s'ils le sont pour tous sauf ...)
* Arthur : les algos sont chiants à utiliser, trop de paramètres.
* Peut etre passer le lien "cool dataset" en reference ?
* S'appuyer sur ce post pour le tsne : https://stats.stackexchange.com/a/352138 (et voir cette reference : https://arxiv.org/abs/1712.09005)
* Transformer les 4 appels à visualise en seulement 2, s'inspirer de cette réponse : https://stackoverflow.com/a/21884187


## References
* [1] R. Krishnapuram ; A. Joshi ; O. Nasraoui ; L. Yi, Low-complexity fuzzy relational clustering algorithms for Web mining,  IEEE Transactions on Fuzzy Systems (Volume: 9, Issue: 4, Aug 2001), p595-607, DOI: 10.1109/91.940971
* [2] Tolga Can, K-Means Empty Cluster Example, http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html (archive: https://web.archive.org/web/20180626092955/http://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html)
* [3] Ross, Timothy J., Fuzzy Logic With Engineering Applications, 3rd ed. Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353

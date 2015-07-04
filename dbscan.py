import numpy

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric=lambda x, y: numpy.linalg.norm(x - y)):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = numpy.ndarray(0, dtype=int)
    
    # returns indicies of neighbouring to center points
    def _get_neighbours(self, centre, samples):
        res = []
        for i in range(len(samples)):
            if 0 < self.metric(centre, samples[i]) <= self.eps:
                res.append(i)
        return res

    # mark all points density-connected to 'cluster' as belonging to cluster 'n'
    def _expand(self, samples, cluster, visited, n):
        i = 0
        while i < len(cluster):
            if visited[cluster[i]]:
                i += 1
                continue
            
            visited[cluster[i]] = True
            self.labels_[cluster[i]] = n 
            
            # add all the neighbourhood
            nbh = self._get_neighbours(samples[cluster[i]], samples)
            if len(nbh) >= self.min_samples:
                for j in nbh:
                    if visited[j]:
                        continue
                    cluster.append(j)
            i += 1

    def fit(samples):
        n = len(samples)
        self.labels_ = numpy.ndarray(n, dtype=int)
        cur_clusters = 0
        vis = [False for i in range(n)]

        for i in range(n):
            if vis[i]:
                continue
            neighbours = self._get_neighbours(samples[i], samples) 
            vis[i] = True

            if len(neighbours) < min_samples:
                # noise 
                self.labels_[i] = 0
                continue
            
            # found new cluster
            cur_clusters += 1
            self.labels_[i] = cur_clusters
            self._expand(samples, neighbours, vis, cur_clusters)

        return self

    def fit_predict(samples):
        self.fit(samples)
        return self.labels_

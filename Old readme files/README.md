# 1. Clustering

## a. K-means algorithm

- random initialization $K$ cluster centroids
- repeat
  - assign points to above centroids: for each $i = 1$ to $m$, denote $c_i$ the index of centroids closest to $x_i$
  - move centroids: for $k=1$ to $K$, update $\mu_k$ to be mean of all points assigned to cluster $k$
- all steps above is to minimize is distortion function
$$J(c_1,...,c_m,\mu_1,...,\mu_K)=\frac{1}{m} \sum ||x_i - \mu_{c_i}||^2$$

For each initialization, there might be local minima. Therefore, we have improvements on random initialization of $K$ centroid:
- run around 50 - 1000 times to choose best initialization
- for $i=1$ to $100$:
  - pick $K$ random examples and assign to $\mu_i$'s
  - run algorithm to convergence to get $c_1,...,c_m,\mu_1,...,\mu_K$.
  - compute cost function for each case. Compare to choose the best. 

# 2. Anomaly detection
## a. Density estimation:
Suppose there are $m$ examples $x^1, ..., x^m$ where each example $x^i$ has $n$ feature $x^i_1,..., x^i_n$. We may assume further that these features are independent.
Because of CLT, we may estimate for each feature $\mu_j$ and $\sigma_j, j = 1,..., n$ as mean and variance of the corresponding set.

Now for new example $x$, we compute $f(x)$ where $f$ is like joint pdf. Note in the case of independence, $f(x) = \prod_{j=1,...,n} f_j(x_j)$ where each $f_j$ is pdf of normal distribution.

## b. Anomaly detection algorithm
- choose $n$ features that are indicate of anomalous examples
- find $\mu_j$ and $\sigma_j, j = 1,..., n$
- for new $x$, compute $$f(x) = \prod_{j=1,...,n} f_j(x_j)$$ Check if $f(x)<\epsilon$, if yes then confirm anomalous.

## c. Evaluating
- it is helpful to have some anomalies in the cv set and test set. We may assume that we have labels and training set has only $y=0$. The number of $y=1$ is quite slow compared to $y=0$, put some of them in cv and some in test. If it is too small then we can put all in cv, and therfore no test set
- use cv set to tune $\epsilon$ and features (add or subtract)
- use metrics like precision, recall, f1-score to evaluate

## d. Features engineering
- if not Gaussian, test transformations with `plt.hist()` and functions like $log(x+c)$, $x^a$...
- error analysis: add some weird features as long as for training set, it remains $f(x)$ high for normal and low for anomaly.

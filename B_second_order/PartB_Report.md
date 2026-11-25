# Part B · Second-order SMO and paired timing

## Second-order working-set selection

Let $D(\alpha)=\mathbf 1^\top\alpha-\tfrac12\alpha^\top Q\alpha$ be the C-SVC dual objective to maximize, and let $G=Q\alpha-\mathbf 1$ be the gradient of $-D$. Second-order SMO first chooses $i\in I_{\mathrm{up}}$ that maximizes $-y_iG_i$. Holding $i$ fixed, it scans $j\in I_{\mathrm{low}}$ with positive projected gap $b_{ij}$.

$$
d=y_i e_i-y_j e_j,\qquad
b_{ij}=-G^\top d=-y_iG_i+y_jG_j,\qquad
\eta_{ij}=d^\top Qd=K_{ii}+K_{jj}-2K_{ij}.
$$

A step of length $\rho$ changes $D$ by $b_{ij}\rho-\frac12\eta_{ij}\rho^2$. The unconstrained gain is $b_{ij}^2/(2\eta_{ij})$, so the selector ranks partners by $b_{ij}^2/\eta_{ij}$. For example, candidates with $(b,\eta)=(0.90,1.80)$ and $(0.75,0.50)$ score 0.450 and 1.125 respectively. The latter ranks first because its lower curvature offsets its smaller gap. We use $10^{-12}$ in place of a nonpositive curvature. [Fan, Chen, and Lin (2005)](https://www.jmlr.org/papers/v6/fan05a.html)

Because $y^\top d=0$, the update preserves the equality constraint. We clip the line optimum $b_{ij}/\eta_{ij}$ to $\rho_{\max}$, the first box boundary along $d$. Only two coefficients change, so we maintain $G$ using two columns of $Q$. Selection and gradient maintenance each take $O(n)$ work given the kernel columns. The two Python implementations share the same selector, including its tie-breaking rule.

## Matched endpoint and convergence checks

All runs used the distributed feature values and row order, RBF kernel, $C=1$, $\gamma=1/22$, zero initialization, full-set KKT tolerance $\varepsilon=0.001$ and shrinking disabled. The ijcnn1 subset contains 1,500 stratified training rows selected with seed 19. The Python histories record the start, every tenth update and the exact endpoint. We independently recomputed all endpoint metrics from each saved $\alpha$.

| Data | Python method | Updates | Maximized dual | Final KKT | Equality residual | Box violation |
|---|---|---:|---:|---:|---:|---:|
| heart_scale (270) | Second-order SMO | 120 | 106.519167343 | 0.00076362 | 1.33e-15 | 0 |
| heart_scale (270) | CSMO | 119 | 106.519168848 | 0.00093154 | 4.44e-16 | 0 |
| ijcnn1 subset (1,500) | Second-order SMO | 467 | 288.601035292 | 0.00095130 | 5.55e-16 | 0 |
| ijcnn1 subset (1,500) | CSMO | 424 | 288.601069803 | 0.00099134 | 5.55e-16 | 0 |

On the fixed ijcnn1 subset, CSMO used 43 fewer updates (9.21%). Both endpoints met the KKT target, with a dual objective difference of about $3.45\times10^{-5}$. The convergence figure plots the measured KKT violation against update count on a logarithmic axis, with the shared tolerance marked at 0.001.

The two author modes also passed the shared KKT criterion on both datasets. The native author cache stores kernel entries in single precision. Recomputing with a float64 kernel and the shared boundary tolerance gives the following fixed-subset endpoints. The four ijcnn1 subset solutions had zero box violation and equality residuals below $3\times10^{-15}$.

| Author mode on ijcnn1 subset | Updates | Native KKT | Recomputed KKT | Recomputed maximized dual |
|---|---:|---:|---:|---:|
| Second-order SMO | 476 | 0.00084776 | 0.00084763 | 288.601067633 |
| CSMO | 431 | 0.00075606 | 0.00075578 | 288.601105070 |

## Five paired timings on the full ijcnn1 split

We ran second-order SMO (mode 0) and CSMO (mode 1) in one build of libsvm_cd at commit 432631da95845152ce24404fbecb93a5e546e586. We compiled with GCC 13.3.0 and -O3, -std=c++14, and -DNDEBUG. The benchmark used 49,990 training and 91,701 test examples, the shared RBF settings and a 100 MB cache on one logical CPU of an AMD EPYC 9A75 host. The training bytes match the original SHA-256 recorded by C and reproduce the fixed subset in order. Two public mirrors yielded byte-identical test files. [Author implementation](https://github.com/albertotb/libsvm_cd) · [LIBSVM data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1)

Each round ran both modes, alternating which mode went first. A monotonic timer encloses only the call to svm_train(). File loading, saving, prediction and extra endpoint recomputation were outside the timer. We calculated $T/N$ for each run and then took the median by mode. This quantity amortizes training initialization over the updates. A paired ratio $T_{\mathrm{SMO}}/T_{\mathrm{CSMO}}$ above one favors CSMO.

| Author mode | Updates $N$ | Native maximized dual | Native final KKT | Median $T$ (s) | Median $T/N$ (ms) | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Second-order SMO | 6,730 | 8,148.188173 | 0.00098717 | 8.8798 | 1.3194 | 92.787% |
| CSMO | 6,653 | 8,148.188386 | 0.00094379 | 9.0878 | 1.3660 | 92.787% |

| Round | First mode | SMO time (s) | CSMO time (s) | SMO/CSMO time ratio |
|---:|---|---:|---:|---:|
| 1 | Second-order SMO | 8.8798 | 9.0134 | 0.9852× |
| 2 | CSMO | 8.8444 | 9.0839 | 0.9736× |
| 3 | Second-order SMO | 8.9006 | 9.0878 | 0.9794× |
| 4 | CSMO | 8.8404 | 9.1261 | 0.9687× |
| 5 | Second-order SMO | 8.9612 | 9.1417 | 0.9803× |
| **Median** | | | | **0.9794×** |

All ten runs converged. Within each mode, both the saved solution and final metrics were identical across the five runs. With the shared float64 recomputation, the final KKT violations were 0.00098708 for second-order SMO and 0.00094294 for CSMO. Both modes had zero box violation, equality residuals around $1.07\times10^{-14}$ and 85,087 correct test predictions.

On the full split, CSMO saved 77 updates (1.14%) and its median time per update was 3.53% higher. Second-order SMO finished sooner in all five rounds. The paired time ratio ranged from 0.9687× to 0.9852×, with a median of 0.9794×. The reduced number of updates was too small to offset the higher cost per update on this build and host.

The author logs reported 6,343 clipped steps out of 6,653 CSMO updates (95.34%) and zero fallbacks per run. Frequent clipping limits how often history carries forward. The timing measures aggregate training cost under the stated cache and tolerance settings. The cost of each operation would require separate profiling.

# Naïve Bayes on the Flu Symptoms Dataset — Detailed Explainer

This document accompanies `lab-9.ipynb`. It covers the theory behind every question in Assignment 09, the exact numbers the notebook produces, a line-by-line walkthrough of the implementation, and the points one is expected to defend in the viva. No built-in classifier is used for any prediction; scikit-learn's `CategoricalNB` appears only as an independent correctness check.

## 1. What the assignment asks

The lab implements a Naïve Bayes classifier from scratch on a tiny categorical dataset of flu symptoms.

| Question | Task |
| --- | --- |
| Q1 | Load the data; estimate class priors $P(\text{Flu}=v)$ and every conditional probability $P(a_i \mid v)$; display them in a table. |
| Q2 | Implement the score $\text{Score}(v_j) = P(v_j)\prod_i P(a_i \mid v_j)$; classify one patient; print every intermediate probability. |
| Q3 | Classify at least five new symptom combinations not present in the dataset. |
| Q4 | Pick at least three symptom pairs and test the conditional-independence identity $P(A,B\mid C) = P(A\mid C)P(B\mid C)$ empirically. |
| Q5 | Perturb one symptom at a time from several base patients and explain which symptoms dominate the decision. |

## 2. The dataset

`assignment09_flu_symptoms.csv` contains 8 fully labelled patient records and 4 categorical symptoms.

| Chills | RunnyNose | Headache | Fever | Flu |
| --- | --- | --- | --- | --- |
| Y | N | Mild | Y | N |
| Y | Y | No | N | Y |
| Y | N | Strong | Y | Y |
| N | Y | Mild | Y | Y |
| N | N | No | N | N |
| N | Y | Strong | Y | Y |
| N | Y | Strong | N | N |
| Y | Y | Mild | Y | Y |

Domains:

- `Chills` ∈ {N, Y}
- `RunnyNose` ∈ {N, Y}
- `Headache` ∈ {Mild, No, Strong} (ternary)
- `Fever` ∈ {N, Y}
- `Flu` ∈ {N, Y} (the class)

Class balance: 5 positive (`Flu=Y`) and 3 negative (`Flu=N`). The dataset is deliberately tiny. With 8 rows and $2\cdot2\cdot2\cdot3 = 24$ possible symptom combinations, the data is extremely sparse; this sparsity is the main reason the conditional-independence test in Q4 fails and the reason the sensitivity analysis in Q5 behaves the way it does.

## 3. Bayesian decision theory

Naïve Bayes is a **generative** classifier: it models how data is produced, $P(x \mid v)$, rather than the class boundary directly.

Bayes' theorem for a class $v$ and a feature vector $x = (a_1, \dots, a_n)$:

$$P(v \mid x) = \frac{P(x \mid v)\,P(v)}{P(x)}.$$

- $P(v)$ is the **class prior**: how common the class is before seeing any symptom.
- $P(x \mid v)$ is the **likelihood**: how likely this symptom pattern is among patients of class $v$.
- $P(x)$ is the **evidence**: the same denominator for every class, so it does not affect the ranking.

The **MAP decision rule** picks the class with the largest posterior:

$$\hat{v} = \arg\max_v P(v \mid x) = \arg\max_v P(v)\,P(x \mid v).$$

Because $P(x)$ is constant in $v$, classification needs only the numerator. This is exactly the quantity the assignment calls $\text{Score}(v_j)$.

## 4. The Naïve Bayes assumption

Estimating the full joint $P(x \mid v)$ directly is impossible here: each of the 24 patterns would need its own probability per class, and the sample has 8 rows. Naïve Bayes works around this with one strong assumption — the features are **conditionally independent given the class**:

$$P(a_1, \dots, a_n \mid v) = \prod_{i=1}^{n} P(a_i \mid v).$$

This turns one hard joint estimation into $n$ easy one-dimensional estimations, at the cost of an assumption that is rarely exactly true in real data (hence "naïve"). Substituting into the MAP rule gives the assignment's score:

$$\text{Score}(v_j) = P(v_j) \prod_{i=1}^{n} P(a_i \mid v_j).$$

Crucially, the product form uses $P(a_i \mid v_j)$ — one-symptom marginals — not the true pairwise joints. Q4 measures how far apart those two things are in this data.

## 5. Q1 — Estimating priors and conditional probabilities

Every parameter is a **maximum likelihood estimate (MLE)**, i.e. a relative frequency:

$$P(\text{Flu}=v) = \frac{\text{count}(\text{Flu}=v)}{N}, \qquad P(a_i = w \mid v) = \frac{\text{count}(a_i = w,\ \text{Flu}=v)}{\text{count}(\text{Flu}=v)}.$$

### Class priors

| Class | Count | $P(\text{Flu}=v)$ |
| --- | --- | --- |
| N | 3 | 0.3750 |
| Y | 5 | 0.6250 |

### Conditional probabilities

| Symptom | Value | count(Y) | count(N) | $P(a\mid \text{Flu}=Y)$ | $P(a\mid \text{Flu}=N)$ |
| --- | --- | --- | --- | --- | --- |
| Chills | N | 2 | 2 | 0.4000 | 0.6667 |
| Chills | Y | 3 | 1 | 0.6000 | 0.3333 |
| RunnyNose | N | 1 | 2 | 0.2000 | 0.6667 |
| RunnyNose | Y | 4 | 1 | 0.8000 | 0.3333 |
| Headache | Mild | 2 | 1 | 0.4000 | 0.3333 |
| Headache | No | 1 | 1 | 0.2000 | 0.3333 |
| Headache | Strong | 2 | 1 | 0.4000 | 0.3333 |
| Fever | N | 1 | 2 | 0.2000 | 0.6667 |
| Fever | Y | 4 | 1 | 0.8000 | 0.3333 |

Consistency check: within each class the value probabilities of a symptom sum to 1, e.g. `Chills`: $0.4 + 0.6 = 1$ for `Flu=Y` and $0.6667 + 0.3333 = 1$ for `Flu=N`. The notebook asserts this for all four symptoms.

A few observations that drive everything later:

- `Fever` and `RunnyNose` are the most **discriminative** symptoms: their marginals differ sharply between classes ($0.8$ vs $0.2$ for `Y`; $0.3333$ vs $0.6667$ when `N`).
- `Headache` is nearly **uninformative**: its three values are almost uniform within each class, so any of its values contributes roughly the same likelihood under both classes.
- Nothing is zero here for these specific value/class cells, but several **pairs** have zero joint counts (Q4), which is where sparsity bites.

## 6. Q2 — The classifier and one worked prediction

The implementation is a literal transcription of the score formula:

```python
def score(x, v):
    s = priors[v]
    for f in FEATURES:
        s *= conditional[f][x[f]][v]
    return s
```

`predict` computes the score for both classes and returns the argmax. Because products of many small numbers underflow, an equivalent log-space version is also implemented:

$$\log \text{Score}(v_j) = \log P(v_j) + \sum_{i=1}^{n} \log P(a_i \mid v_j).$$

Since $\log$ is monotone, the argmax is identical; the notebook verifies this. Posterior probabilities are recovered by normalising the scores:

$$P(v \mid x) = \frac{\text{Score}(v)}{\sum_{v'} \text{Score}(v')}.$$

### Worked example

Patient: `Chills=Y, RunnyNose=N, Headache=Mild, Fever=Y`.

**Class `Flu = Y`**

$$P(Y)\cdot P(\text{Chills}=Y \mid Y)\cdot P(\text{RunnyNose}=N \mid Y)\cdot P(\text{Headache=Mild}\mid Y)\cdot P(\text{Fever}=Y\mid Y)$$

$$= 0.6250 \times 0.6000 \times 0.2000 \times 0.4000 \times 0.8000 = 0.02400000.$$

**Class `Flu = N`**

$$= 0.3750 \times 0.3333 \times 0.6667 \times 0.3333 \times 0.3333 = 0.00925926.$$

Normalised posteriors:

$$P(\text{Flu}=Y \mid x) = \frac{0.024}{0.024 + 0.009259} = 0.7216, \qquad P(\text{Flu}=N \mid x) = 0.2784.$$

Prediction: **Flu = Y**. Log scores are $-3.7297$ (`Y`) and $-4.6821$ (`N`); their difference $0.9524$ is exactly $\log\frac{P(Y\mid x)}{P(N\mid x)}$, confirming the log form.

## 7. Q3 — Unseen patients

Six combinations absent from the training table were classified. Scores are shown with the normalised $P(Y\mid x)$.

| Chills | RunnyNose | Headache | Fever | Score(Y) | Score(N) | $P(Y\mid x)$ | Pred |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N | N | Mild | Y | 0.016000 | 0.018519 | 0.4635 | N |
| Y | Y | No | Y | 0.048000 | 0.004630 | 0.9120 | Y |
| N | N | No | Y | 0.008000 | 0.018519 | 0.3017 | N |
| Y | N | Strong | N | 0.006000 | 0.018519 | 0.2447 | N |
| N | Y | Mild | N | 0.016000 | 0.018519 | 0.4635 | N |
| Y | N | No | Y | 0.012000 | 0.009259 | 0.5645 | Y |

Notes:

- The same `Score(N) = 0.018519` recurs. That number is $0.375 \times (2/3)^2 \times (1/3)^2$: several of these patients have the two "majority-negative" symptom values (`Chills=N`, `RunnyNose=N` in some rows, or combinations that hit $2/3$ twice and $1/3$ twice), so the negative score collapses to the same product. Whether `Flu=Y` wins then depends only on how many strong "Y-leaning" values the patient has.
- Two patients are decided only weakly: `N,N,Mild,Y` has $P(Y\mid x)=0.4635$, a near tie. With so few samples the posterior is dominated by whichever cells happened to receive counts, so a single flip of a strong symptom can reverse it (see Q5).

## 8. Q4 — Is the conditional-independence assumption true here?

Naïve Bayes multiplies marginals. That equals the true joint iff

$$P(A,B \mid C) = P(A \mid C)\,P(B \mid C).$$

The notebook computes both sides directly from counts for five pairs, in both classes:

| Pair | Class | $n(A,B,C)$ | $P(A,B\mid C)$ | $P(A\mid C)P(B\mid C)$ | Difference |
| --- | --- | --- | --- | --- | --- |
| (Chills=Y, RunnyNose=Y) | N | 0 | 0.0000 | 0.1111 | −0.1111 |
| (Chills=Y, RunnyNose=Y) | Y | 2 | 0.4000 | 0.4800 | −0.0800 |
| (Chills=Y, Fever=Y) | N | 1 | 0.3333 | 0.1111 | +0.2222 |
| (Chills=Y, Fever=Y) | Y | 2 | 0.4000 | 0.4800 | −0.0800 |
| (Headache=Strong, Fever=Y) | N | 0 | 0.0000 | 0.1111 | −0.1111 |
| (Headache=Strong, Fever=Y) | Y | 2 | 0.4000 | 0.3200 | +0.0800 |
| (RunnyNose=Y, Fever=Y) | N | 0 | 0.0000 | 0.1111 | −0.1111 |
| (RunnyNose=Y, Fever=Y) | Y | 3 | 0.6000 | 0.6400 | −0.0400 |
| (Headache=Mild, Fever=Y) | N | 1 | 0.3333 | 0.1111 | +0.2222 |
| (Headache=Mild, Fever=Y) | Y | 2 | 0.4000 | 0.3200 | +0.0800 |

**No pair satisfies the identity exactly.** Reasons, in order of importance:

1. **Sampling noise.** The `Flu=N` class has only 3 rows. A probability such as $1/3$ is estimated from three observations; the joint count is 0 or 1. Empirical marginals from so few rows cannot reproduce the identity even if the features were truly independent.
2. **Genuine dependence.** For `Flu=Y`, `Chills=Y` co-occurs with `Fever=Y` in 2 of 5 cases while the marginals would predict 2.4; the symptoms carry overlapping information (a patient with chills tends to have fever). Naïve Bayes cannot represent this and treats the two as separate evidence.
3. **Sparsity / zero cells.** Joints like `(Chills=Y, RunnyNose=Y)` among `Flu=N` patients are 0, while the marginal product is $1/9$. The product of marginals cannot produce a structural zero.

The practical consequence: Naïve Bayes systematically **over-counts** correlated evidence, pushing posteriors toward 0 or 1. In this dataset the scores for the winning class are often several times the loser's — stronger confidence than 8 samples justify.

## 9. Q5 — Probability sensitivity

Three base patients are perturbed one symptom at a time. Because raw scores are tiny and scale-dependent, each change is summarised by the change in the **log-likelihood ratio** (the amount added to the log-odds of flu):

$$\Delta \text{LLR} = \log\frac{P(a_i^{\text{new}}\mid Y)\,P(a_i^{\text{old}}\mid N)}{P(a_i^{\text{old}}\mid Y)\,P(a_i^{\text{new}}\mid N)}.$$

Because $\log\frac{\text{Score}(Y)}{\text{Score}(N)} = \log\frac{P(Y)}{P(N)} + \sum_i \log\frac{P(a_i\mid Y)}{P(a_i\mid N)}$, changing one symptom adds exactly $\Delta\text{LLR}$ to the log-odds. The sign says which class the change supports; the magnitude ranks its influence independently of the patient.

Selected results (three base patients, base log-odds in parentheses):

| Base patient | Change | Score(Y) / Score(N) | ΔLLR | Pred |
| --- | --- | --- | --- | --- |
| `Chills=Y, RN=N, Headache=Mild, Fever=Y` (+0.9524) | Fever Y→N | 0.006000 / 0.018519 | −2.0794 | **N (flip)** |
| | Chills Y→N | 0.016000 / 0.018519 | −1.0986 | **N (flip)** |
| | RunnyNose N→Y | 0.096000 / 0.004630 | +2.0794 | Y |
| | Headache Mild→No | 0.012000 / 0.009259 | −0.6931 | Y |
| | Headache Mild→Strong | 0.024000 / 0.009259 | 0.0000 | Y |
| `Chills=N, RN=Y, Headache=Strong, Fever=Y` (+1.9333) | RunnyNose Y→N | 0.016000 / 0.018519 | −2.0794 | **N (flip)** |
| | Fever Y→N | 0.016000 / 0.018519 | −2.0794 | **N (flip)** |
| `Chills=Y, RN=Y, Headache=No, Fever=N` (+0.2593) | RunnyNose Y→N | 0.003000 / 0.018519 | −2.0794 | **N (flip)** |
| | Chills Y→N | 0.008000 / 0.018519 | −1.0986 | **N (flip)** |
| | Headache No→Mild | 0.024000 / 0.009259 | +0.6931 | Y |

Aggregate influence:

| Symptom | mean $|\Delta\text{LLR}|$ | max $|\Delta\text{LLR}|$ | predictions flipped |
| --- | --- | --- | --- |
| Chills | 1.0986 | 1.0986 | 2 / 3 |
| RunnyNose | 2.0794 | 2.0794 | 2 / 3 |
| Headache | 0.4621 | 0.6931 | 0 / 6 |
| Fever | 2.0794 | 2.0794 | 2 / 3 |

Interpretation:

- **`Fever` and `RunnyNose` are the strongest signals.** Their $\Delta\text{LLR}$ is $\pm\log(0.8/0.2 \div 0.3333/0.6667) = \pm 2.0794$, the largest possible in this data, because their class-conditional marginals are the furthest apart. Either of them alone flips the decision for two of the three base patients.
- **`Chills` is intermediate** ($\pm 1.0986$): flipping it reverses the prediction whenever the base log-odds margin is smaller than about 1.1, which is why it flips two bases but does not dominate.
- **`Headache` is nearly inert.** `Mild` and `Strong` give the same likelihood ratio (ΔLLR = 0 for that swap), and `Mild`↔`No` changes it by only $0.6931$. It never flips a prediction. The reason is visible in Q1: the three `Headache` probabilities are close to uniform within each class, so swapping values barely changes the product.
- The **strongest single effect** in absolute posterior terms is `RunnyNose`/`Fever` on the base with the smallest prior margin (`+1.9333` etc.): a symptom worth 2.08 in log-odds will always overcome a margin below that.

This is the practical face of the over-confidence noted in Q4: because each symptom contributes multiplicatively, a single high-discrimination symptom can dominate the product irrespective of the rest.

## 10. Implementation details

The notebook is organised to mirror the questions, with a small amount of reusable machinery.

### Data loading

`csv.DictReader` streams the file into a list of dictionaries keyed by column name. `FEATURES` is everything except `Flu`; `CLASSES` and the per-feature value domains `VALUES` are derived from the data rather than hard-coded, so the code would still work if a symptom gained a level.

```python
with open(DATA_PATH, newline="") as handle:
    reader = csv.DictReader(handle)
    FEATURES = [c for c in reader.fieldnames if c != CLASS]
    ROWS = [dict(row) for row in reader]
```

### Counting structures

Two nested structures hold all the statistics used everywhere else:

- `class_counts`: a `Counter` of `Flu` values (the prior numerator).
- `counts[f][value][v]`: number of patients with symptom `f = value` and class `v`. This is a three-level dictionary of `Counter`s, built in one pass over the rows.

Positional counts are then converted to probabilities with a dict comprehension:

```python
conditional = {
    f: {val: {v: counts[f][val][v] / class_counts[v] for v in CLASSES} for val in VALUES[f]}
    for f in FEATURES
}
```

### Model functions

- `score(x, v)` — the product-form score.
- `predict(x)` — returns `(argmax, all scores)`.
- `log_score(x, v)` / `predict_log(x)` — numerically stable form for large $n$.
- `posterior(scores)` — normalises scores into probabilities.
- `joint_prob(a, aval, b, bval, v)` — empirical $P(A,B\mid C)$ for Q4.
- `llr_delta(f, old, new)` — the Q5 influence metric, implemented directly from the conditional table.
- `brute_force_posterior(x)` — recomputes the posterior from raw counts with an independent code path (verify against the probability tables).
- `print_table(...)` — a dependency-free pretty-printer so the notebook needs no pandas.

### Verification

1. Every symptom's conditional probabilities sum to 1 within each class.
2. The product-score ranking and the log-score ranking agree for every query.
3. The table-based posterior and the raw-count brute-force posterior agree to $10^{-12}$.
4. `sklearn.naive_bayes.CategoricalNB(alpha=0)` — built-in Naïve Bayes with **no smoothing**, so its MLE matches ours — produces identical predictions on all seven queries (`N, Y, N, N, N, Y, Y`). This is the only use of a built-in classifier and it is used purely for verification, as the assignment permits.

### Figures

- Left: heatmap of $P(a_i \mid v_j)$ for every symptom/value block, annotated with the numeric values. Reading it column-wise shows that `RunnyNose` and `Fever` rows swing the most between the two classes.
- Right: bar chart of the signed $\Delta\text{LLR}$ for every single-symptom change across the base patients, making the dominance of `RunnyNose` and `Fever` and the inertness of `Headache` visually obvious.

## 11. The zero-frequency problem and Laplace smoothing

The notebook deliberately uses MLE. MLE assigns probability 0 to any symptom value that never co-occurs with a class. If a test patient had such a value, the entire product for that class would become 0 regardless of the other symptoms, producing a hard and unjustified veto. Q4 already exposes zero joint cells.

**Laplace (add-α) smoothing** fixes this by adding a pseudo-count $\alpha$ (usually 1) to every value/class combination:

$$P(a_i = w \mid v) = \frac{\text{count}(a_i = w, v) + \alpha}{\text{count}(v) + \alpha\,|a_i|}, \qquad P(v) = \frac{\text{count}(v) + \alpha}{N + \alpha\,|C|},$$

where $|a_i|$ is the number of levels of symptom $i$. Smoothing never gives probability 0 and decreases as the sample grows.

Effect on this dataset: `CategoricalNB(alpha=1)` changes 2 of the 7 query predictions versus `alpha=0` (e.g. `N,N,Mild,Y` and `N,Y,Mild,N` both flip to `Flu=Y`), because with 8 rows a pseudo-count of 1 is a large correction. The assignment asks for the unsmoothed estimator, so the notebook reports MLE and notes smoothing as the standard remedy.

## 12. Other limitations worth stating

- **Sample size.** Eight rows and a three-level `Headache` give several cells with a single observation; all probabilities have large variance.
- **Discretisation.** All features are naturally categorical here. Continuous features would require Gaussian NB or binning.
- **Feature dependence.** The Q4 mismatches are real; Naïve Bayes cannot model them and over-commits.
- **Calibration.** Posterior scores are not trustworthy as confidences on this data even when the ranking is correct.
- **Binary relevance.** The model assumes every symptom is relevant; `Headache` mostly adds noise, and feature selection would help.

## 13. Viva quick reference

- **Class prior** $P(v)$: base rate of the class before observing symptoms; here $0.625$ (`Y`) and $0.375$ (`N`), computed as class counts over $N=8$.
- **Conditional probability** $P(a_i \mid v)$: frequency of a symptom value within a class; the multiplicative factors of the likelihood under the independence assumption.
- **Naïve Bayes assumption**: features are conditionally independent given the class, so $P(x\mid v) = \prod_i P(a_i\mid v)$. It reduces parameter count from exponential to linear in the number of features, at the cost of ignoring interactions.
- **Probability estimation**: maximum likelihood from relative frequencies; add-α smoothing is the standard fix for zero counts.
- **Classification score**: $P(v)\prod_i P(a_i\mid v)$, compared across classes; the evidence term $P(x)$ is dropped because it is class-independent. Log-space is used for stability.
- **Conditional independence check**: compare $P(A,B\mid C)$ with $P(A\mid C)P(B\mid C)$; they disagree here because of tiny samples, zero cells, and genuine correlation, so the assumption holds only approximately.
- **Prediction behaviour**: discriminative symptoms with large marginal gaps (`Fever`, `RunnyNose`, ΔLLR ≈ 2.08) flip predictions; near-uniform ones (`Headache`, ΔLLR ≤ 0.69) do not. A single strong symptom can outweigh the prior when the margin is small, which is the over-confidence the independence assumption produces.

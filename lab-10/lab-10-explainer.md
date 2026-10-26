# Information Retrieval with the Vector Space Model and TF–IDF — Detailed Explainer

This document accompanies `lab-10.ipynb`. It explains the retrieval model behind Assignment 10, derives the formulas used in Q1–Q5, walks through the NumPy implementation, and lists the related weighting and similarity schemes one is expected to know for the viva. Every number quoted here is produced by the notebook. scikit-learn is used only as an independent check; no built-in vectorizer, TF–IDF transformer, or similarity function supplies a result.

## 1. What the assignment asks

The lab turns a four-sentence corpus into vectors, compares those vectors with two geometric measures, and then reweights them with TF–IDF.

| Question | Task | Core idea |
| --- | --- | --- |
| Q1 | Build the vocabulary, the bag-of-words representation, and the full term–document matrix. | A document becomes a vector of term counts. |
| Q2 | Implement Euclidean distance, compute all pairwise distances, find the nearest pair, and study documents of different lengths. | Distance depends on both content and magnitude. |
| Q3 | Implement cosine similarity, compute all pairwise values, rank `d1`, `d2`, `d3` against `d4`, and compare with Q2. | Normalizing by length isolates direction. |
| Q4 | Implement $w_{ij}=(1+\log \text{tf}_{ij})\log(N/\text{df}_i)$, display the matrix, and interpret high and low weights. | Rare terms are informative; common terms are not. |
| Q5 | Recompute the three similarities with TF–IDF and compare the ranking with raw term frequency. | Weighting can change which document is retrieved. |

The corpus is deliberately tiny so every matrix can be printed and checked by hand. That makes the arithmetic transparent, but it also means the statistical estimates are unstable; Section 12 returns to this point.

## 2. Corpus and preprocessing

| ID | Text | Tokens after preprocessing | Length |
| --- | --- | --- | --- |
| `d1` | The cow jumped over the moon | `the, cow, jumped, over, the, moon` | 6 |
| `d2` | O'Leary's cow kicked the lamp | `o'leary's, cow, kicked, the, lamp` | 5 |
| `d3` | The kicked lamp started a fire | `the, kicked, lamp, started, a, fire` | 6 |
| `d4` | The cow on fire | `the, cow, on, fire` | 4 |

Preprocessing is deliberately minimal:

1. The Unicode right single quotation mark in `O’Leary’s` is replaced by the ASCII apostrophe, so the tokenizer sees one consistent character.
2. The text is lowercased, which merges `The` at the start of `d1` and `d3` with `the` elsewhere.
3. A regular expression extracts alphabetic runs that may contain internal apostrophes. `O'Leary's` therefore stays a single token rather than becoming `o`, `leary`, and `s`.

There is no stemming and no stop-word removal. `jumped`, `kicked`, and `started` remain distinct from one another, and `the`, `a`, `on`, and `over` remain in the vocabulary. That is useful for this lab: Q4 shows numerically what happens to a term that occurs in every document.

The vocabulary is the sorted set of all tokens:

```
a, cow, fire, jumped, kicked, lamp, moon, o'leary's, on, over, started, the
```

Twelve terms over four documents give a $12 \times 4$ matrix with 48 cells. Only 20 are non-zero, so the matrix is already 58% empty; real document collections are far sparser.

## 3. Bag of Words and the vector space model

The bag-of-words assumption is that a document is characterized by which terms it contains and how often, and not by their order. `the cow on fire` and `fire on the cow` receive identical vectors. This discards grammar, negation, and local context, but it makes documents comparable with simple linear algebra.

Formally, let $V$ be the vocabulary and $N$ the number of documents. Document $j$ is a vector

$$
x_j = (x_{1j}, x_{2j}, \dots, x_{|V|j}) \in \mathbb{R}^{|V|},
$$

where $x_{ij}=\text{tf}_{ij}$ is the number of occurrences of term $i$ in document $j$. Stacking the document vectors as columns gives the term–document matrix

$$
M \in \mathbb{N}^{|V| \times N}, \qquad M_{ij} = \text{tf}_{ij}.
$$

In the vector space model, each term is a coordinate axis and each document is a point in $|V|$-dimensional space. The axes are treated as independent, which is the geometric counterpart of the bag-of-words independence assumption. Retrieval becomes a comparison of vectors: a query is also represented in the same space, and documents are ranked by a distance or similarity score.

Two orientations of the same matrix appear in practice. This lab uses terms as rows and documents as columns, because that makes the term weights easy to read down a column. scikit-learn uses documents as rows and terms as columns, so the verification code transposes with `.T`. Both conventions are correct; only the indexing changes.

Raw counts have two weaknesses that Q2 and Q4 make visible. A longer document accumulates larger counts, so it is farther from everything in Euclidean distance. A repeated common word also accumulates count even though it says little about the document's topic. TF–IDF addresses the second problem directly; cosine addresses the first.

## 4. Q1 — Building the term–document matrix

The entry $M_{ij}$ is the term frequency:

$$
\text{tf}_{ij} = \#\{\text{occurrences of term } i \text{ in document } j\}.
$$

The construction is a four-step algorithm:

1. Tokenize every document and collect the unique tokens into a vocabulary.
2. Assign each vocabulary term a row index.
3. For every document, count every vocabulary term.
4. Write the counts into a matrix whose rows are terms and whose columns are documents.

The notebook's matrix is:

| term | d1 | d2 | d3 | d4 | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| a | 0 | 0 | 1 | 0 | 1 |
| cow | 1 | 1 | 0 | 1 | 3 |
| fire | 0 | 0 | 1 | 1 | 2 |
| jumped | 1 | 0 | 0 | 0 | 1 |
| kicked | 0 | 1 | 1 | 0 | 2 |
| lamp | 0 | 1 | 1 | 0 | 2 |
| moon | 1 | 0 | 0 | 0 | 1 |
| o'leary's | 0 | 1 | 0 | 0 | 1 |
| on | 0 | 0 | 0 | 1 | 1 |
| over | 1 | 0 | 0 | 0 | 1 |
| started | 0 | 0 | 1 | 0 | 1 |
| the | 2 | 1 | 1 | 1 | 5 |
| **doc length** | **6** | **5** | **6** | **4** | **21** |

Three details matter for the later questions:

- The document lengths are 6, 5, 6, and 4. The column sums give 21 total tokens.
- `the` occurs in all four documents, but its total count is 5 because `d1` contains it twice. Document frequency and collection frequency are different quantities. Q4 uses $\text{df}(\text{the})=4$, not 5.
- Every count is 0 or 1 except `the` in `d1`, which is 2. The matrix is therefore almost binary, and the repeated `the` is the only place where raw term frequency can reward repetition.

## 5. Q2 — Euclidean distance and document length

Euclidean distance is the $L_2$ norm of the difference vector:

$$
\text{dist}(X,Y)=\sqrt{\sum_{i=1}^{|V|}(x_i-y_i)^2}=\lVert X-Y\rVert_2.
$$

Expanding the square gives the useful identity

$$
\lVert X-Y\rVert_2^2=\lVert X\rVert_2^2+\lVert Y\rVert_2^2-2\,X\cdot Y.
$$

The distance is zero only when the vectors are identical, it is symmetric, and it satisfies the triangle inequality. Those properties make it a genuine metric, unlike cosine similarity (Section 6).

The pairwise matrix is:

| from \ to | d1 | d2 | d3 | d4 |
| --- | ---: | ---: | ---: | ---: |
| d1 | 0.0000 | 2.6458 | 3.1623 | 2.4495 |
| d2 | 2.6458 | 0.0000 | 2.2361 | 2.2361 |
| d3 | 3.1623 | 2.2361 | 0.0000 | 2.4495 |
| d4 | 2.4495 | 2.2361 | 2.4495 | 0.0000 |

The nearest pairs are `(d2, d3)` and `(d2, d4)`, both at $\sqrt5 \approx 2.2361$. The next closest pairs are `(d1, d4)` and `(d3, d4)` at $\sqrt6 \approx 2.4495$. The farthest pair is `(d1, d3)` at $\sqrt{10}\approx 3.1623$.

The reason is the number of differing counts, not the number of tokens. `d2` and `d3` share `the`, `kicked`, and `lamp`, leaving five unit differences; `d2` and `d4` also differ in five places. `d1` and `d3` share only `the`, so ten counts differ, even though both documents have six tokens. Length alone does not determine distance; shared content does.

The vector norms already show the problem:

$$
\lVert d1\rVert=2.8284,\quad \lVert d2\rVert=2.2361,\quad \lVert d3\rVert=2.4495,\quad \lVert d4\rVert=2.0.
$$

### The length experiment

To isolate length from content, the notebook repeats `d4` $k$ times for $k=1,2,3,4$ and compares each copy with the fixed `d3`:

| copies | tokens | $\lVert x\rVert$ | dist$(x,d3)$ | cosine | dist(unit vectors) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 2.00 | 2.4495 | 0.4082 | 1.0879 |
| 2 | 8 | 4.00 | 3.7417 | 0.4082 | 1.0879 |
| 3 | 12 | 6.00 | 5.4772 | 0.4082 | 1.0879 |
| 4 | 16 | 8.00 | 7.3485 | 0.4082 | 1.0879 |

If $x=k\,y$, then

$$
\lVert x-z\rVert_2^2=k^2\lVert y\rVert_2^2+\lVert z\rVert_2^2-2k\,y\cdot z.
$$

For large $k$ the first term dominates, so the distance grows roughly linearly in $k$ (here from 2.4495 to 7.3485). The cosine, however, is unchanged:

$$
\frac{(k y)\cdot z}{\lVert k y\rVert\,\lVert z\rVert}
=\frac{k\,y\cdot z}{k\,\lVert y\rVert\,\lVert z\rVert}
=\frac{y\cdot z}{\lVert y\rVert\,\lVert z\rVert}.
$$

Normalizing both vectors removes the scale exactly. For unit vectors $\hat X$ and $\hat Y$,

$$
\lVert \hat X-\hat Y\rVert_2^2=2-2\cos(X,Y),
$$

which is why the last column of the table is constant at $\sqrt{2-2(0.4082)}=1.0879$. Euclidean distance therefore mixes two things: a content difference and a magnitude difference. It is appropriate when the vectors are already on a comparable scale, for example fixed-length feature vectors or normalized embeddings. On raw term counts of variable-length documents, it penalizes length whether or not the extra words are relevant.

## 6. Q3 — Cosine similarity

Cosine similarity compares directions:

$$
\text{sim}(X,Y)=\frac{X\cdot Y}{\lVert X\rVert\,\lVert Y\rVert}=\cos\theta.
$$

By the Cauchy–Schwarz inequality the value lies in $[-1,1]$. With non-negative term counts it lies in $[0,1]$: 0 means no shared terms (orthogonal vectors), and 1 means the vectors are positive multiples of one another. The angle is $\theta=\arccos(\text{sim}(X,Y))$.

With `d4` as the query, the raw-count components are:

| doc | $X\cdot d4$ | $\lVert X\rVert$ | $\lVert d4\rVert$ | cosine |
| --- | ---: | ---: | ---: | ---: |
| d1 | 3 | 2.8284 | 2.0000 | 0.5303 |
| d2 | 2 | 2.2361 | 2.0000 | 0.4472 |
| d3 | 2 | 2.4495 | 2.0000 | 0.4082 |

For example, `cow` contributes $1\cdot1$ and `the` contributes $2\cdot1$, so $d1\cdot d4=3$ and

$$
\cos(d1,d4)=\frac{3}{2.8284\times2}=0.5303.
$$

The complete cosine matrix is:

| from \ to | d1 | d2 | d3 | d4 |
| --- | ---: | ---: | ---: | ---: |
| d1 | 1.0000 | 0.4743 | 0.2887 | 0.5303 |
| d2 | 0.4743 | 1.0000 | 0.5477 | 0.4472 |
| d3 | 0.2887 | 0.5477 | 1.0000 | 0.4082 |
| d4 | 0.5303 | 0.4472 | 0.4082 | 1.0000 |

### Comparison with Euclidean distance

| doc | cosine to d4 | cosine rank | Euclidean to d4 | Euclidean rank |
| --- | ---: | ---: | ---: | ---: |
| d1 | 0.5303 | 1 | 2.4495 | 2 (tie) |
| d2 | 0.4472 | 2 | 2.2361 | 1 |
| d3 | 0.4082 | 3 | 2.4495 | 2 (tie) |

The measures disagree at the top: Euclidean picks `d2`, cosine picks `d1`. The disagreement comes from how each measure treats magnitude. `d1` overlaps `d4` in three counts (`the` twice and `cow` once), so its dot product is larger, but its norm is also larger; the larger dot product more than compensates, so after normalization its direction is closest to `d4`'s. `d2` is closer to `d4` in raw magnitude because it is shorter, which is exactly the advantage Euclidean distance rewards and cosine discards.

Cosine is useful for information retrieval because document length varies for reasons unrelated to topic (verbose authors, longer abstracts, more metadata). Dividing by the norm removes that nuisance variable and compares the relative mix of terms instead. It is also cheap: once the norms are precomputed, each comparison is one dot product. The price is that magnitude is ignored completely. A document that says `fire` once and one that says `fire` fifty times can have the same cosine if the rest of their vocabulary is proportional, even though the repetition may matter.

One more distinction is worth stating precisely. Cosine similarity is not a metric: it is maximal rather than zero for identical vectors, and it does not satisfy the triangle inequality. The angular distance $\arccos(\cos\theta)$ is a metric on the unit sphere, and the chord distance $\sqrt{2-2\cos\theta}$ is the Euclidean distance between unit vectors. The quantity $1-\cos\theta$ is monotone in the angle but is not itself a metric.

## 7. Q4 — TF–IDF weighting

Raw term frequency has two failure modes: it grows with document length and it rewards repetition of uninformative words. Document frequency supplies the missing corpus-level signal. A term that appears in every document cannot distinguish them; a term that appears in one document is highly specific to it.

The assignment's weighting is

$$
w_{ij}=\bigl(1+\log \text{tf}_{ij}\bigr)\log\frac{N}{\text{df}_i},
$$

with $w_{ij}=0$ when $\text{tf}_{ij}=0$. Here $N=4$ and $\text{df}_i$ is the number of documents containing term $i$ (not the total count).

### The term-frequency factor

The factor $1+\log \text{tf}$ is sublinear. Its derivative $1/\text{tf}$ decreases as the count grows, so the tenth occurrence adds much less than the second:

| tf | 1 | 2 | 3 | 5 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| $1+\log \text{tf}$ | 1.0000 | 1.6931 | 2.0986 | 2.6094 | 3.3026 |

A linear factor would let one repeated word dominate the vector. The logarithm compresses that effect while still ranking "three occurrences" above "one occurrence".

### The inverse-document-frequency factor

$$
\text{idf}_i=\log\frac{N}{\text{df}_i}.
$$

| df | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: |
| idf (N = 4) | 1.3863 | 0.6931 | 0.2877 | 0.0000 |

The factor is zero for a term in every document and maximal for a term in one document. It is a corpus-level measure: the same word can have different weights in two different collections. With natural logarithms, a term in one of four documents gets $\log 4=1.3863$; a term in half the corpus gets $\log 2=0.6931$.

The notebook's df and idf values are:

| term | df | $N/\text{df}$ | idf |
| --- | ---: | ---: | ---: |
| a | 1 | 4.0000 | 1.3863 |
| cow | 3 | 1.3333 | 0.2877 |
| fire | 2 | 2.0000 | 0.6931 |
| jumped | 1 | 4.0000 | 1.3863 |
| kicked | 2 | 2.0000 | 0.6931 |
| lamp | 2 | 2.0000 | 0.6931 |
| moon | 1 | 4.0000 | 1.3863 |
| o'leary's | 1 | 4.0000 | 1.3863 |
| on | 1 | 4.0000 | 1.3863 |
| over | 1 | 4.0000 | 1.3863 |
| started | 1 | 4.0000 | 1.3863 |
| the | 4 | 1.0000 | 0.0000 |

### The weighted matrix

| term | d1 | d2 | d3 | d4 |
| --- | ---: | ---: | ---: | ---: |
| a | 0.0000 | 0.0000 | 1.3863 | 0.0000 |
| cow | 0.2877 | 0.2877 | 0.0000 | 0.2877 |
| fire | 0.0000 | 0.0000 | 0.6931 | 0.6931 |
| jumped | 1.3863 | 0.0000 | 0.0000 | 0.0000 |
| kicked | 0.0000 | 0.6931 | 0.6931 | 0.0000 |
| lamp | 0.0000 | 0.6931 | 0.6931 | 0.0000 |
| moon | 1.3863 | 0.0000 | 0.0000 | 0.0000 |
| o'leary's | 0.0000 | 1.3863 | 0.0000 | 0.0000 |
| on | 0.0000 | 0.0000 | 0.0000 | 1.3863 |
| over | 1.3863 | 0.0000 | 0.0000 | 0.0000 |
| started | 0.0000 | 0.0000 | 1.3863 | 0.0000 |
| the | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Three non-zero weight levels appear, plus a zero level:

- Maximum $1.3863$ for terms in exactly one document: `a`, `jumped`, `moon`, `o'leary's`, `on`, `over`, `started`.
- $0.6931$ for terms in two documents: `fire`, `kicked`, `lamp`.
- $0.2877$ for `cow`, which is in three documents.
- Zero for `the`, which is in all four.

The sublinear factor is 1 almost everywhere because every tf is 1 except `the` in `d1`, where $\text{tf}=2$ gives $1+\log 2=1.6931$. That larger tf weight is multiplied by $\text{idf}(\text{the})=0$, so the cell still becomes zero. This is a useful check that the two factors are genuinely multiplicative: a high tf cannot rescue a term with no discriminative power.

Common terms get small weights because they are poor evidence for ranking. If every document contains `the`, observing `the` in a query tells us nothing about which document to return. Rare terms are more informative about topic. The idf is not a formal entropy, but it captures the same intuition and has a probabilistic justification in the Robertson–Spärck Jones weighting tradition.

Three properties of this particular TF–IDF variant deserve note. It uses natural logarithms; changing the base rescales the idf column by a constant and therefore leaves cosine rankings unchanged. It does not normalize for document length; cosine does that later. And it defines zero for absent terms, so the matrix stays sparse.

## 8. Q5 — Why the ranking reverses

The same cosine measure is now applied to the raw count vectors and to the TF–IDF vectors, with `d4` as the query. Only terms shared with `d4` contribute to the dot product:

| doc | shared terms with d4 | raw dot | raw norm | raw cosine | TF–IDF dot | TF–IDF norm | TF–IDF cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d1 | cow (0.2877), the (0) | 3 | 2.8284 | 0.5303 | 0.0828 | 2.4183 | 0.0217 |
| d2 | cow (0.2877), the (0) | 2 | 2.2361 | 0.4472 | 0.0828 | 1.7221 | 0.0305 |
| d3 | fire (0.6931), the (0) | 2 | 2.4495 | 0.4082 | 0.4805 | 2.2989 | 0.1326 |

The TF–IDF norm of `d4` itself is $1.5764$. The rankings are:

| Representation | Ranking with respect to d4 |
| --- | --- |
| raw tf | d1 (0.5303) > d2 (0.4472) > d3 (0.4082) |
| TF–IDF | d3 (0.1326) > d2 (0.0305) > d1 (0.0217) |

The reversal has a clean explanation. Under raw counts, `d1` wins because it overlaps `d4` in three counts: `the` twice and `cow` once. But `the` is in every document and `cow` is in three of four, so both are weak discriminators. TF–IDF sets `the` to zero and reduces `cow` to 0.2877. `d1` and `d2` are then left with exactly the same weighted overlap (`cow` alone), so their dot products are equal at 0.0828. `d2` wins that tie because its weighted vector is shorter: a smaller denominator gives a larger cosine. `d3` shares `fire`, which occurs in only two documents and therefore carries weight 0.6931, the strongest informative overlap of the three. That moves `d3` from last to first. The absolute scores also shrink, because each document's norm is dominated by high-idf terms that `d4` does not contain.

It is worth seeing how much of the raw result comes from the duplicated article. If `d1` contained `the` only once, its dot product with `d4` would be 2 and its norm $\sqrt5=2.2361$, giving

$$
\cos(d1,d4)=\frac{2}{2.2361\times2}=0.4472,
$$

which would tie `d2` and drop `d1` from first place. The repetition of a stop word is precisely the pattern raw term frequency rewards and TF–IDF is designed to suppress.

The broader lesson is that a retrieval score is only as good as its weighting assumptions. TF–IDF encodes the prior that rare terms are more informative. That prior is reasonable for topical search, but it is not universal: on a corpus where a common word is the actual query target, zeroing its idf can hurt. Ranking changes are a hypothesis, not proof of better retrieval; a labelled relevance set is needed to decide.

## 9. Algorithms and complexity

The notebook uses straightforward dense algorithms because $|V|=12$ and $N=4$. The same operations have standard sparse and inverted-index implementations at scale.

| Stage | Method in the notebook | Time | Space |
| --- | --- | --- | --- |
| Tokenization | regex scan of each document | $O(T)$ | $O(T)$ |
| Vocabulary | set union, then sort | $O(T + V\log V)$ | $O(V)$ |
| Counting | `Counter` hash map (rebuilt per term in the notebook) | $O(VT)$ as written; $O(T)$ if cached | $O(V)$ per document |
| Term–document matrix | dense $V\times N$ array | $O(VN)$ | $O(VN)$ |
| Document frequency | boolean matrix, sum over documents | $O(VN)$ | $O(VN)$ or $O(V)$ |
| TF–IDF | elementwise multiply with broadcast idf | $O(VN)$ | $O(VN)$ |
| Pairwise distances | $N(N-1)/2$ dot products | $O(N^2V)$ | $O(N^2)$ |
| Ranking | sort the $N$ scores | $O(N\log N)$ | $O(N)$ |

Here $T$ is the total number of tokens, $V$ the vocabulary size, and $N$ the number of documents. The pairwise loops exploit symmetry: only the upper triangle is computed and each value is copied to the lower triangle, halving the work. Precomputing the norms reduces each cosine to one dot product plus two divisions.

For a real collection, the dense matrix is replaced by a sparse matrix with $O(\text{nnz})$ storage, where nnz is the number of non-zero term–document pairs. A corpus with 100,000 documents and a 50,000-term vocabulary has $5\times10^9$ possible cells but perhaps $10^7$ non-zero ones. Sparse storage makes the representation feasible; the inverted index makes retrieval fast. Each term stores a postings list of the documents containing it, so a query only scores documents that share at least one query term:

$$
\text{candidates}(Q)=\bigcup_{q\in Q}\text{postings}(q).
$$

The cost is proportional to the length of those postings lists, not to $V$ or $N$. This is the data structure behind practical TF–IDF and BM25 search engines.

## 10. Related weighting and similarity schemes

TF–IDF with cosine is one point in a larger design space. The alternatives below are the ones most likely to come up in a viva or in a follow-up project.

### Set similarities: Jaccard, Dice, overlap

If only presence or absence matters, represent a document as a set of terms $A$ and use

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|},
\qquad
D(A,B)=\frac{2|A\cap B|}{|A|+|B|}.
$$

For `d1` and `d4`, the sets are $\{the,cow,jumped,over,moon\}$ and $\{the,cow,on,fire\}$: the intersection has size 2 and the union has size 7, so $J=2/7\approx0.2857$ and $D=4/9\approx0.4444$. Binary cosine would give $2/\sqrt{5\cdot4}=0.4472$. These measures ignore counts, which is an advantage for near-duplicate detection and a disadvantage when repetition is informative. The overlap coefficient $|A\cap B|/\min(|A|,|B|)$ is a variant that does not penalize a small document for being a subset of a large one.

### TF and IDF variants

The tf factor can be binary, raw, logarithmic, or augmented:

$$
\text{augmented tf}=0.5+0.5\frac{\text{tf}_{ij}}{\max_i \text{tf}_{ij}},
$$

which keeps values in $[0.5,1]$ and limits the influence of very frequent terms. The idf factor has several common forms:

$$
\log\frac{N}{\text{df}},
\qquad
\log\left(\frac{N+1}{\text{df}+1}\right)+1,
\qquad
\log\frac{N-\text{df}+0.5}{\text{df}+0.5}.
$$

The second is the smoothed form used by scikit-learn; the third comes from probabilistic retrieval. Smoothing avoids zero weights for ubiquitous terms and undefined values for terms that never occur in the collection. Changing the base or adding a constant rescales the weights, which matters for absolute scores but not for cosine rankings if the rescaling is global.

### BM25

BM25 (Okapi) is the standard probabilistic extension. For a query $Q$ and document $D$,

$$
\text{score}(D,Q)=\sum_{q\in Q}\text{IDF}(q)\,
\frac{f(q,D)(k_1+1)}{f(q,D)+k_1\left(1-b+b\,\frac{|D|}{\text{avgdl}}\right)},
$$

with

$$
\text{IDF}(q)=\log\left(1+\frac{N-n(q)+0.5}{n(q)+0.5}\right).
$$

Here $f(q,D)$ is the term frequency, $|D|$ the document length, and $\text{avgdl}$ the average document length. The parameter $k_1$ controls how quickly term frequency saturates (typical value 1.2), and $b$ controls length normalization (typical value 0.75; $b=0$ disables it, $b=1$ applies it fully). BM25 keeps TF–IDF's rarity signal but adds explicit saturation and length normalization, which is why it is a strong default baseline.

### Latent semantic analysis

LSA (latent semantic indexing) applies a truncated singular value decomposition to the term–document matrix:

$$
M \approx U_k\Sigma_k V_k^\top.
$$

The top $k$ singular vectors define latent dimensions that mix co-occurring terms, so documents with no shared words can still be close if they use related vocabulary. Similarity is computed with cosine in the reduced space, usually between rows of $U_k\Sigma_k$ or columns of $V_k\Sigma_k$. LSA handles synonymy and some polysemy, at the cost of interpretability and an SVD whose cost grows with the matrix dimensions. It is the linear ancestor of modern embedding models.

### Query likelihood language models

A language-model retriever ranks a document by the probability that it generated the query:

$$
P(Q\mid D)=\prod_{q\in Q}P(q\mid D).
$$

Maximum-likelihood estimates assign zero probability to query terms absent from the document, so smoothing is essential. Dirichlet smoothing is common:

$$
P(q\mid D)=\frac{\text{tf}(q,D)+\mu\,P(q\mid C)}{|D|+\mu},
$$

where $P(q\mid C)$ is the collection language model and $\mu$ controls how much the document is smoothed toward the collection. This approach is probabilistic rather than geometric, but it often performs comparably to BM25.

### Dense embeddings and hybrid retrieval

Modern retrieval represents a document as a dense vector $e(D)\in\mathbb{R}^d$ from a neural encoder and ranks by cosine or dot product. Bi-encoders can embed documents offline, and approximate nearest-neighbour indexes such as HNSW or IVF make billion-scale search feasible. Dense retrieval captures synonymy and paraphrase that bag-of-words misses, but it requires training data, can be opaque, and may miss exact identifiers or rare tokens. Hybrid systems combine a lexical score (BM25 or TF–IDF) with a dense score, which is a common industrial compromise.

| Method | Representation | Similarity | Main strength | Main weakness |
| --- | --- | --- | --- | --- |
| Jaccard / Dice | sets | overlap ratio | simple, length-robust | ignores counts |
| TF–IDF + cosine | sparse weighted counts | cosine | strong lexical baseline, interpretable | no semantics, no length normalization |
| BM25 | sparse counts | probabilistic score | saturation and length control | still lexical |
| LSA | low-rank dense | cosine | captures co-occurrence | costly, less interpretable |
| Query likelihood | term probabilities | product of probabilities | probabilistic foundation | smoothing-sensitive |
| Embeddings | dense neural vectors | cosine / dot | semantic matching | data and compute heavy |

## 11. Python implementation walkthrough

The notebook mirrors Q1–Q5 in order, then adds a figure and a verification cell. The important implementation choices are below; the full code is in `lab-10.ipynb`.

### Tokenization and vocabulary

```python
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")

def tokenize(text):
    return TOKEN_RE.findall(text.lower().replace("’", "'"))
```

`[a-z]+` matches a run of letters, and `(?:'[a-z]+)*` allows zero or more apostrophe-plus-letters groups. Because the group is non-capturing, `findall` returns the whole match, so `o'leary's` survives as one token. The curly apostrophe is replaced before matching; without that step the regex would split at the Unicode character. Lowercasing merges `The` and `the`. The pattern deliberately ignores digits and punctuation; for a general corpus one would keep numbers and decide how to treat hyphens and URLs.

```python
VOCAB = sorted({term for tokens in TOKENS.values() for term in tokens})
TERM_INDEX = {term: i for i, term in enumerate(VOCAB)}
```

The set removes duplicates, and sorting gives a deterministic row order so printed matrices and comparisons are reproducible. `TERM_INDEX` is convenient for lookups; the matrix code uses the list order directly.

### Counting into a matrix

```python
COUNT = np.array(
    [[Counter(TOKENS[d])[term] for d in DOC_IDS] for term in VOCAB],
    dtype=int,
)
```

The outer comprehension walks terms (rows), the inner walks documents (columns). This is correct but rebuilds a `Counter` for every term, which is $O(VT)$ rather than $O(T)$. With 12 terms the cost is irrelevant and the one-line expression is clear. A production version would precompute

```python
COUNTERS = {d: Counter(TOKENS[d]) for d in DOC_IDS}
```

once and then index `COUNTERS[d][term]`. That distinction between clarity and asymptotic efficiency is worth mentioning in a code review.

### Euclidean distance and the pairwise matrix

```python
def euclidean(x, y):
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    return float(np.sqrt(np.dot(diff, diff)))
```

`np.dot(diff, diff)` is $\sum_i \text{diff}_i^2$; it avoids allocating a separate squared array. Casting to `float` keeps the arithmetic in floating point even when the inputs are integer arrays.

```python
for i in range(n):
    for j in range(i + 1, n):
        D[i, j] = D[j, i] = euclidean(COUNT[:, i], COUNT[:, j])
```

Only the upper triangle is evaluated and each result is mirrored, which uses the symmetry of the distance. The diagonal stays zero. For tie detection the notebook uses `math.isclose(dist, best, rel_tol=0, abs_tol=1e-12)` so both nearest pairs are reported even if floating-point rounding differs in the last bits.

The length experiment rebuilds a count vector from a repeated token list:

```python
doc = " ".join(base * k)
vec = np.array([Counter(tokenize(doc))[term] for term in VOCAB], dtype=float)
unit = vec / np.linalg.norm(vec)
```

Repeating the token list is equivalent to multiplying the count vector by $k$, but it exercises the same preprocessing path as a real document. `unit` is the length-normalized vector used for the constant-distance column.

### Cosine similarity and ranking

```python
def cosine(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
```

The denominator is the product of the two $L_2$ norms. There is no zero-vector guard because every document in this corpus is non-empty; a library function should return 0 (or raise) when a norm is zero, since the angle is undefined. The pairwise construction mirrors the Euclidean one.

Ranking uses `sorted(..., key=lambda d: S[DOC_IDS.index(d), 3], reverse=True)` in Q3 and `np.argsort(-scores)` in Q5. Negating the scores turns NumPy's ascending sort into a descending one. If ties matter, `kind="stable"` preserves the input order; the default sort does not promise stability.

### TF–IDF with broadcasting

```python
DF = (COUNT > 0).sum(axis=1)
IDF = np.log(N / DF)
TF = COUNT.astype(float)
TF_WEIGHT = np.where(TF > 0, 1.0 + np.log(np.maximum(TF, 1.0)), 0.0)
TFIDF = TF_WEIGHT * IDF[:, None]
```

`COUNT > 0` is a boolean matrix; summing along `axis=1` counts, for each term, the documents in which it appears. This is the document frequency, not the collection frequency. `N / DF` uses true division and produces floats.

`np.where` selects the sublinear weight for non-zero counts and zero elsewhere. Both branches are evaluated before the selection, so `np.log(0)` would still emit a warning if it appeared. `np.maximum(TF, 1.0)` prevents that: for zero cells the log argument is 1, the branch value is 1, and the result is discarded by the mask. An alternative is to compute only on the non-zero mask, but the guard keeps the expression vectorized.

`IDF[:, None]` changes the shape of the idf vector from $(V,)$ to $(V,1)$, so broadcasting multiplies row $i$ of `TF_WEIGHT` by `IDF[i]`. This is the concise NumPy idiom for weighting every term row by its corpus-wide factor. `IDF.reshape(-1, 1)` is equivalent.

### Verification with scikit-learn

```python
vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    lowercase=False,
    token_pattern=None,
    smooth_idf=False,
    norm=None,
)
```

The custom tokenizer is passed explicitly, so `lowercase=False` avoids a second lowercasing and `token_pattern=None` disables the default token-pattern warning. `smooth_idf=False` requests an unsmoothed IDF and `norm=None` disables vector normalization. scikit-learn's unsmoothed formula is $\log(N/\text{df})+1$, so the code subtracts 1 before comparing with the notebook's `IDF`.

The full scikit-learn TF–IDF matrix is deliberately not compared: scikit-learn uses raw term frequency by default, while Q4 uses the sublinear $1+\log\text{tf}$ factor. Setting `sublinear_tf=True` would align them, but the notebook keeps the check focused on the vocabulary, the IDF values, and the similarity functions. The other checks do not rely on scikit-learn: a recount of the BoW matrix, an explicit per-document TF–IDF loop, the identity $\cos=1-\lVert\hat X-\hat Y\rVert^2/2$, and a comparison of `cosine_similarity(TFIDF.T)` and `euclidean_distances(COUNT.T)` with the hand-written matrices.

The transposes are necessary because scikit-learn expects documents as rows, while the notebook stores them as columns. `TFIDF.T` is $N\times V$, and `cosine_similarity` then returns an $N\times N$ document–document matrix.

### Figure

The figure has three panels: a heatmap of the count matrix, a heatmap of the TF–IDF matrix, and grouped bars comparing raw and weighted cosine to `d4`. The heatmaps make the sparsity and the zeroed `the` row visible; the bars make the ranking reversal visible at a glance. Annotations are drawn only for non-zero cells, which keeps the count plot readable.

## 12. Limitations and what to watch

- The corpus has four documents. IDF estimates from such a small sample are unstable, and the zero weight of `the` is a consequence of this particular collection, not a universal fact.
- Preprocessing choices drive the results. Stemming would merge `fire` and `fires`; stop-word removal would delete `the` before TF–IDF; keeping case would split `The` and `the`. None of these is wrong, but the explainer should state which one was used.
- Bag of words ignores order, negation, and syntax. `the cow on fire` and `fire on the cow` are identical, and `not hot` is indistinguishable from `hot` apart from the token `not`.
- TF–IDF is not length-normalized by itself. The notebook applies cosine afterward; a raw dot product of TF–IDF vectors would still favor long documents. BM25 makes length normalization explicit.
- Cosine ignores magnitude. If the number of occurrences matters, cosine can under-rank a document that discusses the query term intensively.
- The idf is corpus-dependent. Adding documents changes every weight, and a term absent from the collection has no defined idf without smoothing.
- The dense implementation does not scale. The notebook stores $V\times N$ arrays; real systems use sparse matrices and inverted indexes, and pairwise comparison is replaced by top-$k$ retrieval.
- There are no relevance judgments here. The Q5 reversal is an illustration of how weighting changes rankings, not evidence that TF–IDF retrieves the "correct" document. A proper evaluation uses precision, recall, mean average precision, or NDCG on labelled queries.
- Logarithms are natural in the notebook. A different base rescales the idf column and changes absolute scores; cosine rankings are unchanged only when the rescaling is global.

## 13. Viva quick reference

- **Bag of Words**: represents a document by term counts, ignoring order and grammar.
- **Vector Space Model**: treats terms as dimensions and documents as vectors; retrieval is a geometric comparison.
- **Term frequency ($\text{tf}_{ij}$)**: number of times term $i$ occurs in document $j$.
- **Document frequency ($\text{df}_i$)**: number of documents containing term $i$. It is not the total count; `the` has df 4 but count 5 here.
- **Term–document matrix**: rows terms, columns documents, entries term frequencies. scikit-learn uses the transpose.
- **Euclidean distance**: $\lVert X-Y\rVert_2$; zero only for identical vectors; sensitive to magnitude and therefore to document length.
- **Cosine similarity**: $X\cdot Y/(\lVert X\rVert\lVert Y\rVert)$; scale-invariant; in $[0,1]$ for non-negative counts; standard for variable-length documents.
- **Why cosine is preferred in IR**: it compares term composition rather than document length, and it is cheap for sparse vectors.
- **Length effect**: repeating a document scales its norm and its Euclidean distance but leaves its cosine unchanged. Unit-vector distance is constant at $\sqrt{2-2\cos}$.
- **TF–IDF**: $w_{ij}=(1+\log\text{tf}_{ij})\log(N/\text{df}_i)$; sublinear tf times inverse document frequency.
- **Why log tf**: diminishing returns for repeated terms; the tenth occurrence should not count ten times as much as the first.
- **Why idf**: rare terms discriminate between documents; a term in every document has idf zero and cannot affect ranking.
- **High vs low weights here**: singleton terms 1.3863, two-document terms 0.6931, `cow` 0.2877, `the` 0.
- **Q5 result**: raw ranking `d1 > d2 > d3`; TF–IDF ranking `d3 > d2 > d1`. `the` is zeroed, `cow` is downweighted, and the rarer `fire` makes `d3` the best match.
- **Cosine vs Euclidean**: cosine is not a metric (no triangle inequality, maximum at identity); Euclidean is. Use angular distance or chord distance if a metric is required.
- **Complexity**: building the matrix is $O(T + VN)$; pairwise comparison is $O(N^2V)$ dense or proportional to non-zeros sparse; ranking is $O(N\log N)$.
- **Scaling**: sparse matrices plus an inverted index; a query scores only documents in the postings lists of its terms.
- **Related methods**: Jaccard/Dice for sets, BM25 for probabilistic retrieval with saturation and length normalization, LSA for latent semantics, query-likelihood language models, and dense embeddings for semantic search.

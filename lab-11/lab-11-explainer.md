# Lab 11 Explainer: Part-of-Speech Tagging with a Hidden Markov Model

This explainer walks through the mathematics and the code behind `lab-11/lab-11.ipynb`. The notebook implements a first-order hidden Markov model (HMM) part-of-speech tagger from scratch. It does not call a POS tagger, an HMM library, or a pretrained model. The only numerical dependencies are NumPy for display matrices and Matplotlib for the summary figure.

The target sentence is

$$
\text{सीता बगीचे से सुंदर लाल फूल तोड़ती है ।}
$$

and the tagset is

$$
S=\{\text{NNP},\text{NN},\text{PSP},\text{JJ},\text{VM},\text{VAUX}\}.
$$

The danda `।` is punctuation, not a lexical word in the tagset, so it is dropped from the observation sequence. The tagger therefore sees eight words and must choose among $6^8=1{,}679{,}616$ possible tag sequences.

---

## 1. What the assignment asks

The five questions build on one another:

1. **Q1** represents the sentence as an HMM and estimates the transition and emission tables.
2. **Q2** computes the joint probability of the words and one specified tag sequence, including the start and end transitions.
3. **Q3** implements Viterbi decoding from scratch and prints the score table and backpointers.
4. **Q4** makes सुंदर ambiguous between `JJ` and `NN`, reruns Viterbi, changes a transition probability, and explains why the chosen tag changes.
5. **Q5** tags at least three additional short sentences using the same vocabulary and compares the results.

The notebook answers all five and then verifies the decoder against exhaustive enumeration. The point is not just to get a tag sequence. The point is to show that the sequence is the solution to a well-defined optimization problem, and that every number in the result can be traced back to a count in a small corpus.

---

## 2. The sentence, the tagset, and the linguistic reading

The words and their intended tags are:

| Word | Tag | Gloss | Linguistic reason |
| --- | --- | --- | --- |
| सीता | NNP | Sita | personal name, proper noun |
| बगीचे | NN | garden (oblique) | common noun; oblique form before a postposition |
| से | PSP | from | postposition |
| सुंदर | JJ | beautiful | adjective modifying फूल |
| लाल | JJ | red | adjective modifying फूल |
| फूल | NN | flower(s) | common noun |
| तोड़ती | VM | plucks | main verb, feminine singular agreeing with सीता |
| है | VAUX | is | auxiliary verb |

Hindi is verb-final. The main verb `तोड़ती` precedes the auxiliary `है`, adjectives precede the noun they modify, and a postposition follows its noun phrase. Those regularities are not hard-coded as rules. They are learned as transition probabilities from the training corpus.

The tagset distinguishes:

- **NNP** — proper noun.
- **NN** — common noun.
- **PSP** — postposition.
- **JJ** — adjective.
- **VM** — main verb.
- **VAUX** — auxiliary verb.

The tagset is deliberately small. A real Hindi tagger would need more distinctions (case, number, gender, person, tense, aspect, mood, and so on), but a small tagset keeps every count table readable and every probability checkable by hand.

---

## 3. The HMM formalism

### 3.1 Hidden states and observations

An HMM has two sequences:

- The **hidden state sequence** $T=(t_1,\dots,t_n)$ is the sequence of POS tags.
- The **observation sequence** $W=(w_1,\dots,w_n)$ is the sequence of words.

The words are observed. The tags are hidden. A tagger is a procedure that infers the most likely hidden sequence from the observed sequence.

A first-order HMM makes two conditional-independence assumptions:

$$
P(t_i\mid t_1,\dots,t_{i-1})=P(t_i\mid t_{i-1}),
$$

$$
P(w_i\mid T,w_1,\dots,w_{i-1})=P(w_i\mid t_i).
$$

The first says that the current tag depends only on the previous tag. The second says that the current word depends only on its own tag. These are strong assumptions, and they are the main limitation of the model, but they make the inference tractable.

### 3.2 The joint probability

Under those assumptions, the joint probability of a word sequence and a tag sequence factorizes as

$$
P(W,T)
=
P(t_1\mid\text{START})
\left(\prod_{i=2}^{n}P(t_i\mid t_{i-1})\right)
\left(\prod_{i=1}^{n}P(w_i\mid t_i)\right)
P(\text{END}\mid t_n).
$$

There are three kinds of factors:

- **Initial transition** $P(t_1\mid\text{START})$: how a sentence may begin.
- **Internal transitions** $P(t_i\mid t_{i-1})$: how tags follow one another.
- **Emissions** $P(w_i\mid t_i)$: how likely a word is under a tag.
- **Final transition** $P(\text{END}\mid t_n)$: how a sentence may end.

The `START` and `END` states are not words and are not printed as tags. They are bookkeeping states that let the same transition table describe the beginning and the end of a sentence. Without them, the model would have no way to express that a sentence is more likely to begin with a noun phrase than with an auxiliary verb, or that it is more likely to end after `VAUX` than after `PSP`.

### 3.3 Why the factorization matters

The factorization is what makes dynamic programming possible. The joint probability of a full path can be built incrementally:

$$
P(t_1,\dots,t_i,w_1,\dots,w_i)
=
P(t_1,\dots,t_{i-1},w_1,\dots,w_{i-1})
\cdot P(t_i\mid t_{i-1})
\cdot P(w_i\mid t_i).
$$

That is the recurrence behind both the forward algorithm (which sums over paths) and Viterbi (which maximizes over paths). Q2 evaluates one path directly. Q3 maximizes over all paths without enumerating them.

---

## 4. Estimating the parameters

### 4.1 The training corpus

The parameters are estimated from nine short sentences. The target sentence is deliberately **not** in the corpus, so Q2 and Q3 are genuine decoding rather than memorization.

| # | Sentence | Tags |
| --- | --- | --- |
| 1 | सीता फूल तोड़ती है | NNP NN VM VAUX |
| 2 | सुंदर लाल फूल है | JJ JJ NN VAUX |
| 3 | लाल फूल सुंदर है | JJ NN JJ VAUX |
| 4 | सीता बगीचे से फूल तोड़ती है | NNP NN PSP NN VM VAUX |
| 5 | बगीचे से सुंदर फूल है | NN PSP JJ NN VAUX |
| 6 | सीता सुंदर है | NNP JJ VAUX |
| 7 | फूल लाल है | NN JJ VAUX |
| 8 | सीता तोड़ती है | NNP VM VAUX |
| 9 | सुंदर फूल तोड़ती है | JJ NN VM VAUX |

Every word of the target occurs somewhere in the corpus, but the exact target word sequence does not. The corpus is small on purpose: with nine sentences, the count tables fit on one screen, and a reader can verify any probability by hand.

The corpus is also linguistically consistent. It contains:

- sentences that begin with `NNP`, `NN`, or `JJ`;
- postpositional phrases `NN PSP`;
- adjective–noun pairs `JJ NN`;
- verb–auxiliary pairs `VM VAUX`;
- a predicative adjective pattern `NN JJ VAUX`.

Those patterns produce the non-zero transitions that the tagger later uses.

### 4.2 Transition estimation with add-$\alpha$ smoothing

The maximum-likelihood estimate of a transition is

$$
\hat{P}(j\mid i)=\frac{C(i,j)}{C(i)},
$$

where $C(i,j)$ is the number of times tag $j$ follows tag $i$, and $C(i)=\sum_j C(i,j)$ is the number of times tag $i$ occurs in a position with a successor.

With add-$\alpha$ smoothing, the estimate becomes

$$
P(j\mid i)=\frac{C(i,j)+\alpha}{C(i)+\alpha|S|},
\qquad |S|=|\text{TAGS}|+1=7.
$$

The extra state in $|S|$ is `END`. The denominator must distribute probability over all possible successors, including `END`, so it uses seven states, not six.

The notebook uses $\alpha=0.1$. This is a small pseudocount. It keeps every transition non-zero, which means no transition is impossible for a reason that is merely an accident of the small corpus. It does not dominate the observed counts: a transition seen four times still receives most of its row's mass.

Examples:

$$
P(\text{NNP}\mid\text{START})=\frac{4+0.1}{9+0.7}=\frac{4.1}{9.7}=0.422680,
$$

$$
P(\text{NN}\mid\text{NNP})=\frac{2+0.1}{4+0.7}=\frac{2.1}{4.7}=0.446809,
$$

$$
P(\text{JJ}\mid\text{JJ})=\frac{1+0.1}{8+0.7}=\frac{1.1}{8.7}=0.126437.
$$

The denominator for the `START` row is 9 because there are nine sentences and therefore nine initial transitions. The denominator for `NNP` is 4 because `NNP` occurs four times in the corpus. The denominator for `JJ` is 8 because `JJ` occurs eight times.

### 4.3 Emission estimation

The emission probabilities are estimated by maximum likelihood:

$$
P(w\mid t)=\frac{C(w,t)}{C(t)}.
$$

The notebook does **not** smooth the emissions. The vocabulary is closed: the tagger only ever sees the eight words in the corpus. If a word has never been seen with a tag, its emission probability is zero, and any path that uses that word–tag pair receives probability zero.

The base emission table is:

| Word | NNP | NN | PSP | JJ | VM | VAUX |
| --- | --- | --- | --- | --- | --- | --- |
| सीता | 1.0000 | 0 | 0 | 0 | 0 | 0 |
| बगीचे | 0 | 0.2222 | 0 | 0 | 0 | 0 |
| से | 0 | 0 | 1.0000 | 0 | 0 | 0 |
| सुंदर | 0 | 0 | 0 | 0.6250 | 0 | 0 |
| लाल | 0 | 0 | 0 | 0.3750 | 0 | 0 |
| फूल | 0 | 0.7778 | 0 | 0 | 0 | 0 |
| तोड़ती | 0 | 0 | 0 | 0 | 1.0000 | 0 |
| है | 0 | 0 | 0 | 0 | 0 | 1.0000 |

The values follow directly from the corpus:

- `सीता` occurs four times, always as `NNP`: $4/4=1$.
- `बगीचे` occurs twice, always as `NN`: $2/9=0.2222$.
- `फूल` occurs seven times, always as `NN`: $7/9=0.7778$.
- `सुंदर` occurs five times as `JJ`: $5/8=0.6250$.
- `लाल` occurs three times as `JJ`: $3/8=0.3750$.
- `से` occurs twice as `PSP`, `तोड़ती` four times as `VM`, and `है` nine times as `VAUX`.

Because most words have exactly one possible tag in this corpus, the base emission table is deterministic for six of the eight words. That is why the base Viterbi table has only one finite entry per column. Q4 deliberately breaks that determinism for सुंदर.

### 4.4 Why the target must be excluded

If the target sentence were in the training corpus, then every transition and emission in the target path would have been counted directly. Viterbi would still run, but the experiment would not show that the model can generalize to an unseen word sequence. Excluding the target makes the exercise an actual decoding problem.

The notebook asserts this explicitly:

```python
assert TARGET not in [words for words, _ in TAGGED_CORPUS]
```

---

## 5. Q1 in code: tables and data structures

### 5.1 Counts

The notebook stores transition counts as a dictionary of `Counter` objects:

```python
transition_counts = {p: Counter() for p in [START] + TAGS}
emission_counts = {t: Counter() for t in TAGS}
tag_counts = Counter()
```

For each training sentence, it walks through the words and tags, increments the transition count from the previous tag to the current tag, increments the emission count for the current word and tag, and finally increments the transition from the last tag to `END`:

```python
for words, tags in TAGGED_CORPUS:
    prev = START
    for word, tag in zip(words, tags):
        transition_counts[prev][tag] += 1
        emission_counts[tag][word] += 1
        tag_counts[tag] += 1
        prev = tag
    transition_counts[prev][END] += 1
```

The resulting transition count table is:

| from \ to | NNP | NN | PSP | JJ | VM | VAUX | END | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| START | 4 | 2 | 0 | 3 | 0 | 0 | 0 | 9 |
| NNP | 0 | 2 | 0 | 1 | 1 | 0 | 0 | 4 |
| NN | 0 | 0 | 2 | 2 | 3 | 2 | 0 | 9 |
| PSP | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 2 |
| JJ | 0 | 4 | 0 | 1 | 0 | 3 | 0 | 8 |
| VM | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 4 |
| VAUX | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 9 |

Some rows are worth reading carefully:

- `START` starts four sentences with `NNP`, two with `NN`, and three with `JJ`. Those are the three ways a noun phrase can begin in this corpus.
- `NNP` is followed by `NN` twice, `JJ` once, and `VM` once. A proper name can be the subject of a verb (`सीता तोड़ती है`) or the possessor-like start of a noun phrase (`सीता बगीचे से ...`).
- `NN` is followed by `PSP` twice, `JJ` twice, `VM` three times, and `VAUX` twice. A common noun can be followed by a postposition, by a predicative adjective, by a main verb, or by an auxiliary.
- `PSP` is followed by `NN` once and `JJ` once. In this corpus, a postpositional phrase is followed by a noun or an adjective.
- `JJ` is followed by `NN` four times, `JJ` once, and `VAUX` three times. Adjectives usually modify a following noun, but they can also be predicative before `है`.
- `VM` is followed by `VAUX` four times and by nothing else. The corpus contains no subordinate-clause patterns.
- `VAUX` is followed by `END` nine times. Every sentence ends after the auxiliary.

### 5.2 Transition probabilities

The probability table is built with one comprehension per row:

```python
ALPHA = 0.1
transition = {}
for prev in [START] + TAGS:
    total = sum(transition_counts[prev].values())
    transition[prev] = {
        nxt: (transition_counts[prev][nxt] + ALPHA) / (total + ALPHA * len(NEXT_STATES))
        for nxt in NEXT_STATES
    }
```

`NEXT_STATES` is `TAGS + [END]`, so `len(NEXT_STATES)` is 7. The denominator is `total + 0.7`, not `total + 0.6`, because `END` is a possible successor.

The result is:

| from \ to | NNP | NN | PSP | JJ | VM | VAUX | END |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| START | 0.4227 | 0.2165 | 0.0103 | 0.3196 | 0.0103 | 0.0103 | 0.0103 |
| NNP | 0.0213 | 0.4468 | 0.0213 | 0.2340 | 0.2340 | 0.0213 | 0.0213 |
| NN | 0.0103 | 0.0103 | 0.2165 | 0.2165 | 0.3196 | 0.2165 | 0.0103 |
| PSP | 0.0370 | 0.4074 | 0.0370 | 0.4074 | 0.0370 | 0.0370 | 0.0370 |
| JJ | 0.0115 | 0.4713 | 0.0115 | 0.1264 | 0.0115 | 0.3563 | 0.0115 |
| VM | 0.0213 | 0.0213 | 0.0213 | 0.0213 | 0.0213 | 0.8723 | 0.0213 |
| VAUX | 0.0103 | 0.0103 | 0.0103 | 0.0103 | 0.0103 | 0.0103 | 0.9381 |

Every row sums to 1. The notebook asserts this for all seven rows:

```python
for prev in [START] + TAGS:
    assert abs(sum(transition[prev].values()) - 1.0) < 1e-12
```

### 5.3 Emission probabilities and normalization

The emission table is built only for words that actually occur with a tag:

```python
emission = {
    tag: {word: emission_counts[tag][word] / tag_counts[tag]
          for word in WORDS if emission_counts[tag][word] > 0}
    for tag in TAGS
}
```

Zeros are left implicit. `emission["NN"]["सीता"]` would raise a `KeyError`, so the scoring code uses `.get(word, 0.0)` and converts the missing value to $-\infty$ in log space.

The normalization check is:

```python
for tag in TAGS:
    assert abs(sum(emission[tag].get(word, 0.0) for word in WORDS) - 1.0) < 1e-12
```

This checks that each tag's emissions over the closed vocabulary sum to 1, even though the dictionary stores only the non-zero entries.

The code uses a dictionary of dictionaries rather than a dense matrix because the table is sparse and because lookup by tag or word is direct. A dense NumPy matrix is constructed only for the heatmap in the summary figure.

---

## 6. Q2: the joint probability of one path

### 6.1 The candidate path

The candidate sequence is

$$
T=(\text{NNP},\text{NN},\text{PSP},\text{JJ},\text{JJ},\text{NN},\text{VM},\text{VAUX}).
$$

The joint probability is the product of **nine transition factors** and **eight emission factors**:

$$
P(W,T)=
P(\text{NNP}\mid\text{START})
P(\text{सीता}\mid\text{NNP})
P(\text{NN}\mid\text{NNP})
P(\text{बगीचे}\mid\text{NN})
\cdots
P(\text{VAUX}\mid\text{VM})
P(\text{है}\mid\text{VAUX})
P(\text{END}\mid\text{VAUX}).
$$

The full factor list is:

| Step | Factor | Probability | Log |
| ---: | --- | ---: | ---: |
| 1 | START → NNP | 0.422680 | -0.8611 |
| 2 | सीता \| NNP | 1.000000 | 0.0000 |
| 3 | NNP → NN | 0.446809 | -0.8056 |
| 4 | बगीचे \| NN | 0.222222 | -1.5041 |
| 5 | NN → PSP | 0.216495 | -1.5302 |
| 6 | से \| PSP | 1.000000 | 0.0000 |
| 7 | PSP → JJ | 0.407407 | -0.8979 |
| 8 | सुंदर \| JJ | 0.625000 | -0.4700 |
| 9 | JJ → JJ | 0.126437 | -2.0680 |
| 10 | लाल \| JJ | 0.375000 | -0.9808 |
| 11 | JJ → NN | 0.471264 | -0.7523 |
| 12 | फूल \| NN | 0.777778 | -0.2513 |
| 13 | NN → VM | 0.319588 | -1.1407 |
| 14 | तोड़ती \| VM | 1.000000 | 0.0000 |
| 15 | VM → VAUX | 0.872340 | -0.1366 |
| 16 | है \| VAUX | 1.000000 | 0.0000 |
| 17 | VAUX → END | 0.938144 | -0.0639 |

The product splits cleanly into a transition part and an emission part:

$$
\prod_{\text{transitions}}=2.595934\times10^{-4},
\qquad
\log=-8.256394,
$$

$$
\prod_{\text{emissions}}=4.050926\times10^{-2},
\qquad
\log=-3.206225.
$$

Therefore

$$
P(W,T)=2.595934\times10^{-4}\times4.050926\times10^{-2}
=1.051594\times10^{-5},
$$

$$
\log P(W,T)=-8.256394-3.206225=-11.462619.
$$

The smallest individual factors are `JJ→JJ` at 0.1264, `NN→PSP` at 0.2165, and `बगीचे|NN` at 0.2222. The first is small because the corpus contains only one `JJ→JJ` transition. The last is small because `फूल` is much more frequent as an `NN` than `बगीचे`. Neither factor is zero. The product is small mainly because it contains seventeen factors, most below 1.

### 6.2 Why log space is necessary

A product of seventeen probabilities is prone to underflow. In double precision, $1.05\times10^{-5}$ is still representable, but longer sentences or smaller probabilities quickly become zero. The notebook therefore computes the same quantity as a sum of logs:

```python
def sequence_log_probability(words, tags, trans=None, emis=None):
    trans = trans or transition
    emis = emis or emission
    if len(words) != len(tags):
        raise ValueError("words and tags must have equal length")
    total = logp(trans[START][tags[0]]) + logp(emis[tags[0]].get(words[0], 0.0))
    for i in range(1, len(words)):
        total += logp(trans[tags[i - 1]][tags[i]])
        total += logp(emis[tags[i]].get(words[i], 0.0))
    total += logp(trans[tags[-1]][END])
    return total
```

The `logp` helper maps zero to $-\infty$:

```python
def logp(p):
    return math.log(p) if p > 0 else -math.inf
```

That convention is important. An impossible factor contributes $-\infty$, and the whole path becomes impossible. It is not treated as a very small number.

### 6.3 Joint versus marginal probability

Q2 computes $P(W,T)$ for one particular $T$. It does not compute $P(W)$, the marginal probability of the words. The marginal is

$$
P(W)=\sum_{T\in S^n}P(W,T).
$$

The notebook's brute-force verification enumerates all $6^8$ tag sequences and evaluates each one, so it effectively sees every term in that sum. But Q2 itself is a single-path calculation. The distinction matters for the viva: $P(W,T)$ is the score of a path, while $P(W)$ is the likelihood of the observation sequence after summing out the tags.

---

## 7. Q3: Viterbi decoding

### 7.1 The idea

Viterbi answers the question

$$
T^*=\arg\max_{T\in S^n}P(W,T).
$$

A brute-force answer would score all $6^n$ paths. Viterbi avoids that by exploiting the fact that the score of a path can be built left to right. At each position and each possible tag, it keeps only the best score for any prefix ending in that tag. Anything worse can never become optimal later, because all future factors depend only on the current tag.

### 7.2 The recursion

Define the Viterbi variable

$$
\delta_i(j)=\max_{t_1,\dots,t_{i-1}}
P(t_1,\dots,t_{i-1},t_i=j,w_1,\dots,w_i).
$$

It is the best joint probability of any tag prefix of length $i$ that ends in tag $j$.

Initialization:

$$
\delta_1(j)=P(j\mid\text{START})P(w_1\mid j).
$$

Recursion for $i=2,\dots,n$:

$$
\delta_i(j)=
\max_{k\in S}\left[\delta_{i-1}(k)P(j\mid k)\right]P(w_i\mid j).
$$

Backpointer:

$$
\psi_i(j)=
\arg\max_{k\in S}\left[\delta_{i-1}(k)P(j\mid k)\right].
$$

Termination:

$$
P^*=\max_{j\in S}\delta_n(j)P(\text{END}\mid j),
\qquad
t_n^*=\arg\max_{j\in S}\delta_n(j)P(\text{END}\mid j).
$$

Backtracking:

$$
t_i^*=\psi_{i+1}(t_{i+1}^*),
\qquad i=n-1,\dots,1.
$$

The result is the globally optimal tag sequence, not a greedy sequence. A greedy tagger would choose the best tag at each position without looking ahead. Viterbi looks ahead implicitly because the recursion keeps the best prefix for every possible current tag, and the final choice is made only after the whole sentence has been processed.

### 7.3 Log-space implementation

The notebook stores $\log\delta$ instead of $\delta$. The recursion becomes

$$
\log\delta_i(j)=
\max_{k\in S}\left[\log\delta_{i-1}(k)+\log P(j\mid k)\right]
+\log P(w_i\mid j).
$$

The code is:

```python
def viterbi(words, trans=None, emis=None):
    trans = trans or transition
    emis = emis or emission
    n = len(words)
    delta = [dict() for _ in range(n)]
    back = [dict() for _ in range(n)]

    for tag in TAGS:
        delta[0][tag] = logp(trans[START][tag]) + logp(emis[tag].get(words[0], 0.0))
        back[0][tag] = START if math.isfinite(delta[0][tag]) else None

    for i in range(1, n):
        for tag in TAGS:
            candidates = {prev: delta[i - 1][prev] + logp(trans[prev][tag])
                          for prev in TAGS}
            best_prev = max(candidates, key=candidates.get)
            delta[i][tag] = candidates[best_prev] + logp(emis[tag].get(words[i], 0.0))
            back[i][tag] = best_prev if math.isfinite(delta[i][tag]) else None

    final = {tag: delta[-1][tag] + logp(trans[tag][END]) for tag in TAGS}
    best_last = max(final, key=final.get)
    path = [best_last]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    path.reverse()
    return path, delta, back, final[best_last]
```

A few details are worth spelling out:

- `delta` and `back` are lists of dictionaries, one per word position. This keeps the table sparse and makes it easy to print by tag.
- `back[0][tag]` is `START` for a finite initial state. It is `None` when the initial state is impossible, so the printed table does not pretend a predecessor exists.
- The recursion computes a candidate for every predecessor `prev`, then takes the maximum. `max(candidates, key=candidates.get)` returns the tag with the largest candidate score.
- `back[i][tag]` records which predecessor achieved that maximum. This is the only information needed for backtracking.
- The final score adds the `END` transition. The best last tag is the one that maximizes the completed path, not necessarily the one with the largest $\delta_n$.
- Backtracking starts from `best_last`, then follows `back[i][path[-1]]` from right to left, and finally reverses the list.

### 7.4 The base Viterbi table

For the target sentence, the finite entries of the log-$\delta$ table are:

| Word | NNP | NN | PSP | JJ | VM | VAUX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| सीता | -0.861 | −∞ | −∞ | −∞ | −∞ | −∞ |
| बगीचे | −∞ | -3.171 | −∞ | −∞ | −∞ | −∞ |
| से | −∞ | −∞ | -4.701 | −∞ | −∞ | −∞ |
| सुंदर | −∞ | −∞ | −∞ | -6.069 | −∞ | −∞ |
| लाल | −∞ | −∞ | −∞ | -9.118 | −∞ | −∞ |
| फूल | −∞ | -10.121 | −∞ | −∞ | −∞ | −∞ |
| तोड़ती | −∞ | −∞ | −∞ | −∞ | -11.262 | −∞ |
| है | −∞ | −∞ | −∞ | −∞ | −∞ | -11.399 |

The table has one finite entry per column because each word is emitted by exactly one tag in the base table. That is a property of this tiny corpus, not a general property of HMMs.

The backpointers are:

| Word | NNP | NN | PSP | JJ | VM | VAUX |
| --- | --- | --- | --- | --- | --- | --- |
| सीता | START | — | — | — | — | — |
| बगीचे | — | NNP | — | — | — | — |
| से | — | — | NN | — | — | — |
| सुंदर | — | — | — | PSP | — | — |
| लाल | — | — | — | JJ | — | — |
| फूल | — | JJ | — | — | — | — |
| तोड़ती | — | — | — | — | NN | — |
| है | — | — | — | — | — | VM |

Backtracking from `VAUX` gives:

$$
\text{VAUX}\leftarrow\text{VM}\leftarrow\text{NN}\leftarrow\text{JJ}
\leftarrow\text{JJ}\leftarrow\text{PSP}\leftarrow\text{NN}\leftarrow\text{NNP}.
$$

The path is therefore

$$
\text{NNP}\to\text{NN}\to\text{PSP}\to\text{JJ}\to\text{JJ}\to\text{NN}\to\text{VM}\to\text{VAUX}.
$$

The final log score is $-11.462619$, which is exactly the Q2 joint log-probability. That is a useful consistency check: Q2 scored this path directly, and Q3 independently found it as the best path.

### 7.5 Complexity

At each of $n$ positions and for each of $|S|$ tags, the recursion compares $|S|$ predecessors. The time complexity is

$$
O(n|S|^2).
$$

The memory complexity is $O(n|S|)$ because the backpointer table must be retained for backtracking. The score table can be reduced to two columns if only the score is needed, but the backpointers still require $n|S|$ entries.

For $n=8$ and $|S|=6$, the recursion performs $8\times6\times6=288$ predecessor comparisons. Brute force would evaluate $6^8=1{,}679{,}616$ paths. The difference is the whole point of dynamic programming.

---

## 8. Q4: ambiguity, context, and the limits of the emission table

### 8.1 Making सुंदर ambiguous

In the base table, सुंदर is emitted only by `JJ`, with probability 0.6250. To model lexical ambiguity, the notebook gives सुंदर a second reading as a common noun. If the new `NN` mass for सुंदर is $m$, then the other `NN` words keep their relative proportions:

$$
P(\text{बगीचे}\mid\text{NN})=(1-m)\frac{0.2222}{0.2222+0.7778},
$$

$$
P(\text{फूल}\mid\text{NN})=(1-m)\frac{0.7778}{0.2222+0.7778}.
$$

Since $0.2222+0.7778=1$, this simplifies to

$$
P(\text{बगीचे}\mid\text{NN})=(1-m)\frac{2}{9},
\qquad
P(\text{फूल}\mid\text{NN})=(1-m)\frac{7}{9}.
$$

The `JJ` column is unchanged:

$$
P(\text{सुंदर}\mid\text{JJ})=0.6250,
\qquad
P(\text{लाल}\mid\text{JJ})=0.3750.
$$

The code is:

```python
def ambiguous_emission(nn_mass):
    emis = {tag: dict(emission[tag]) for tag in TAGS}
    others = [word for word in emission["NN"] if word != "सुंदर"]
    base_mass = sum(emission["NN"][word] for word in others)
    emis["NN"]["सुंदर"] = nn_mass
    for word in others:
        emis["NN"][word] = emission["NN"][word] / base_mass * (1.0 - nn_mass)
    return emis
```

The copy is important: the base table is not mutated. Each experiment starts from the original probabilities and applies one change.

At $m=0.25$:

| Word | NN |
| --- | ---: |
| बगीचे | 0.1667 |
| फूल | 0.5833 |
| सुंदर | 0.2500 |

At $m=0.40$:

| Word | NN |
| --- | ---: |
| बगीचे | 0.1333 |
| फूल | 0.4667 |
| सुंदर | 0.4000 |

The `NN` column still sums to 1 in both cases, and the notebook asserts it.

### 8.2 What happens at $m=0.25$

At $m=0.25$, Viterbi still tags सुंदर as `JJ`. The path is the base path, but the score is lower because the `NN` alternatives have taken mass away from `फूल` and `बगीचे`:

$$
\log P^*=-12.037983,
\qquad
P^*=5.915215\times10^{-6}.
$$

At the सुंदर position:

$$
\log\delta(\text{JJ})=-6.357,
\qquad
\log\delta(\text{NN})=-7.273.
$$

The `JJ` state is ahead by about 0.916 in log space.

### 8.3 What happens at $m=0.40$

At $m=0.40$, Viterbi flips सुंदर to `NN`:

$$
\text{NNP}\to\text{NN}\to\text{PSP}\to\text{NN}\to\text{JJ}\to\text{NN}\to\text{VM}\to\text{VAUX}.
$$

The score is

$$
\log P^*=-12.392733,
\qquad
P^*=4.148629\times10^{-6}.
$$

At the सुंदर position:

$$
\log\delta(\text{JJ})=-6.580,
\qquad
\log\delta(\text{NN})=-7.026.
$$

The `NN` state is now ahead by about 0.446.

Notice that the total path probability is **lower** at $m=0.40$ than at $m=0.25$. That is not a contradiction. The model is choosing the best path under a different parameter setting. Raising $m$ takes probability mass away from `फूल` and `बगीचे` under `NN`, so every path that uses those words becomes less likely. The best path under the new table is still the `NN` reading of सुंदर, even though the absolute score has dropped.

### 8.4 The local arithmetic

The flip can be explained by comparing the two competing subpaths. The prefix before सुंदर and the suffix after लाल are common to both candidates. The decisive factors are the transition into position 4, the transition from position 4 to position 5, and the two emissions at positions 4 and 5.

For the `JJ` reading:

$$
P_{\text{JJ}}
=
P(\text{PSP}\to\text{JJ})
P(\text{JJ}\to\text{JJ})
P(\text{सुंदर}\mid\text{JJ})
P(\text{लाल}\mid\text{JJ}),
$$

$$
P_{\text{JJ}}
=
0.407407\times0.126437\times0.6250\times0.3750
=0.012073.
$$

For the `NN` reading:

$$
P_{\text{NN}}
=
P(\text{PSP}\to\text{NN})
P(\text{NN}\to\text{JJ})
P(\text{सुंदर}\mid\text{NN})
P(\text{लाल}\mid\text{JJ}),
$$

$$
P_{\text{NN}}
=
0.407407\times0.216495\times m\times0.3750.
$$

At $m=0.25$:

$$
P_{\text{NN}}=0.008269<0.012073,
$$

so `JJ` wins.

At $m=0.40$:

$$
P_{\text{NN}}=0.013230>0.012073,
$$

so `NN` wins.

The important comparison is not

$$
P(\text{सुंदर}\mid\text{JJ})=0.6250
\quad\text{versus}\quad
P(\text{सुंदर}\mid\text{NN})=0.40.
$$

If the decision were made by emission alone, `JJ` would win at both settings. The decision is made by the **product** of emissions and transitions. The transition `NN→JJ` is 0.2165, while `JJ→JJ` is only 0.1264. That difference is large enough to overcome the emission gap once $m$ is high enough.

This is the central lesson of Q4: context and word likelihood are not competing explanations. They are multiplied. A tag with a lower emission can win because it leads to a better transition. A tag with a higher emission can lose because its outgoing transition is poor.

### 8.5 Changing a transition

The notebook also changes one transition: $P(\text{PSP}\to\text{JJ})$ is lowered from 0.4074 to 0.10. The released mass is distributed over the other successors of `PSP` in proportion to their old probabilities. The old `PSP` row is

$$
(0.0370,\;0.4074,\;0.0370,\;0.4074,\;0.0370,\;0.0370,\;0.0370)
$$

for `(NNP, NN, PSP, JJ, VM, VAUX, END)`. After the change, the row is

$$
(0.0563,\;0.6188,\;0.0563,\;0.1000,\;0.0563,\;0.0563,\;0.0563).
$$

The new $P(\text{PSP}\to\text{NN})=0.6188$ is much larger than the old 0.4074. With $m=0.25$, Viterbi now chooses the `NN` reading of सुंदर:

$$
\text{NNP}\to\text{NN}\to\text{PSP}\to\text{NN}\to\text{JJ}\to\text{NN}\to\text{VM}\to\text{VAUX},
$$

$$
\log P^*=-11.998562,
\qquad
P^*=6.153057\times10^{-6}.
$$

At the सुंदर position:

$$
\log\delta(\text{JJ})=-7.761,
\qquad
\log\delta(\text{NN})=-6.855.
$$

The local comparison is now

$$
P_{\text{JJ}}=0.10\times0.126437\times0.6250\times0.3750=0.002963,
$$

$$
P_{\text{NN}}=0.6188\times0.216495\times0.25\times0.3750=0.01256,
$$

so `NN` wins by a wide margin. The transition change does not alter the emission table at all. It changes the context in which the emission is evaluated.

### 8.6 Summary of Q4

| Configuration | सुंदर | $\log P^*$ | $P^*$ | $\log\delta$ JJ | $\log\delta$ NN |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | JJ | -11.462619 | 1.051594e-05 | — | — |
| $P(\text{सुंदर}\mid\text{NN})=0.25$ | JJ | -12.037983 | 5.915215e-06 | -6.357 | -7.273 |
| $P(\text{सुंदर}\mid\text{NN})=0.40$ | NN | -12.392733 | 4.148629e-06 | -6.580 | -7.026 |
| $P(\text{PSP}\to\text{JJ})=0.10$, $m=0.25$ | NN | -11.998562 | 6.153057e-06 | -7.761 | -6.855 |

The base row has no ambiguity, so the `NN` column for सुंदर is impossible and the delta entries are not meaningful in the same way. The table shows the two ambiguity experiments and the transition-stress experiment.

---

## 9. Q5: tagging additional sentences

The four test sentences are unseen word sequences. They reuse the eight-word vocabulary but are not identical to any training sentence. The tagger uses the ambiguous emission table with $m=0.25$.

### 9.1 Results

| # | Sentence | Predicted tags | $\log P^*$ | $P^*$ | $P^{*1/n}$ |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | सीता लाल फूल तोड़ती है | NNP JJ NN VM VAUX | -5.926704 | 2.667259e-03 | 0.305642 |
| 2 | बगीचे से सुंदर फूल तोड़ती है | NN PSP JJ NN VM VAUX | -8.852565 | 1.430144e-04 | 0.228681 |
| 3 | सुंदर फूल लाल है | JJ NN JJ VAUX | -6.508850 | 1.490192e-03 | 0.196476 |
| 4 | सीता बगीचे से लाल फूल तोड़ती है | NNP NN PSP JJ NN VM VAUX | -9.499966 | 7.485436e-05 | 0.257396 |

### 9.2 Reading the predictions

1. **सीता लाल फूल तोड़ती है** → `NNP JJ NN VM VAUX`.
   सीता is the subject, लाल modifies फूल, तोड़ती is the main verb, and है is the auxiliary. The adjective `JJ` before the noun `NN` is the same pattern as training sentence 2.

2. **बगीचे से सुंदर फूल तोड़ती है** → `NN PSP JJ NN VM VAUX`.
   The sentence begins with a common noun because there is no proper name. `से` is a postposition, and सुंदर is an adjective before फूल. The pattern `NN PSP JJ NN` is close to training sentence 5.

3. **सुंदर फूल लाल है** → `JJ NN JJ VAUX`.
   Here लाल is predicative: “the flower is red.” The tagger keeps it as `JJ` before the auxiliary. This is the same `NN JJ VAUX` ending as training sentence 7.

4. **सीता बगीचे से लाल फूल तोड़ती है** → `NNP NN PSP JJ NN VM VAUX`.
   The longer postpositional phrase does not change the tag pattern. The sentence combines the `NNP NN PSP` pattern of training sentence 4 with the `JJ NN VM VAUX` pattern of training sentence 9.

### 9.3 The ambiguity table at सुंदर

The per-sentence tables show what happens when सुंदर has two possible tags.

In sentence 3, सुंदर is `JJ` with $\delta=-1.611$ and `NN` with $\delta=-2.916$. The adjective reading is strongly preferred because the sentence starts with सुंदर and the `START→JJ` transition is 0.3196, while `START→NN` is 0.2165. The emission also favors `JJ` (0.6250 versus 0.2500).

In sentence 2, सुंदर is `JJ` with $\delta=-6.220$ and `NN` with $\delta=-7.136$. The preceding `PSP` makes `PSP→JJ` and `PSP→NN` equally likely in the base table, but the `JJ` emission is stronger. The context does not flip the decision here.

In sentence 1, लाल is forced to `JJ` before फूल. In sentence 3, the same word is forced to `JJ` before है. The same lexical item receives the same tag in these positions because the emission and transition evidence agree. A word is not tagged in isolation; its tag is part of a path.

### 9.4 Why per-token geometric mean is useful

Raw $P^*$ is not comparable across sentences of different lengths. A five-word sentence has 11 factors (6 transitions including START/END, plus 5 emissions), while a seven-word sentence has 15 factors. Every extra factor is at most 1, and most are strictly below 1, so the longer sentence is mechanically less probable even if its tag pattern is just as good.

The per-token geometric mean is

$$
P^{*1/n}=\exp\left(\frac{\log P^*}{n}\right).
$$

It is the average factor per token. On this scale:

- Sentence 1: 0.3056.
- Sentence 2: 0.2287.
- Sentence 3: 0.1965.
- Sentence 4: 0.2574.

Sentence 1 is highest, sentence 3 is lowest, and the long sentence 4 is in between. The ranking reflects how well each local tag pattern is supported by the transition table, not merely how long the sentence is. The geometric mean is still not a calibrated probability of correctness; it is a normalized model score.

---

## 10. Verification

The notebook runs four kinds of checks.

### 10.1 Normalization

Every transition row sums to 1:

```python
for prev in [START] + TAGS:
    assert abs(sum(transition[prev].values()) - 1.0) < 1e-12
```

Every emission column sums to 1 over the closed vocabulary:

```python
for tag in TAGS:
    assert abs(sum(emission[tag].get(word, 0.0) for word in WORDS) - 1.0) < 1e-12
```

The ambiguous tables are checked separately after each renormalization. These checks catch a missing successor, a wrong denominator, or a column that no longer forms a distribution.

### 10.2 Score consistency

The Viterbi path is rescored with `sequence_log_probability`:

```python
target_path, _, _, target_log = viterbi(TARGET)
assert abs(sequence_log_probability(TARGET, target_path) - target_log) < 1e-12
```

This catches a bug in the termination step or in the reconstruction of the path. If the decoder returns a path whose direct score differs from the score it reports, something is inconsistent.

### 10.3 Brute-force enumeration

The strongest check enumerates every tag sequence:

```python
def brute_force(words, trans=None, emis=None):
    best_score = -math.inf
    best_paths = []
    for tags in itertools.product(TAGS, repeat=len(words)):
        score = ...
        if score > best_score + 1e-12:
            best_score = score
            best_paths = [tags]
        elif abs(score - best_score) <= 1e-12:
            best_paths.append(tags)
    return best_score, best_paths
```

For the target, the loop visits $6^8=1{,}679{,}616$ paths. For the Q5 sentences, it visits between $6^4=1{,}296$ and $6^7=279{,}936$ paths. The brute-force best score matches Viterbi to within $10^{-12}$, and the Viterbi path is among the optimal paths.

The output reports exactly one optimal path for the target and for each Q5 sentence. That means the optimum is unique in this parameterization; there is no tie to break.

### 10.4 Training/test separation

The notebook asserts that the target is not in the training corpus:

```python
assert TARGET not in [words for words, _ in TAGGED_CORPUS]
```

It also asserts that each Q5 sentence is unseen:

```python
assert words not in [train_words for train_words, _ in TAGGED_CORPUS]
```

These assertions are simple, but they protect the interpretation of the experiment. Without them, a high score might reflect memorization rather than decoding.

---

## 11. Code architecture and reproducibility

The notebook is organized so that each question is a short sequence of markdown explanation and code.

| Cell | Contents |
| ---: | --- |
| 1 | Imports, constants, tagset, vocabulary, target sentence |
| 3 | Nine-sentence tagged corpus and assertions |
| 5 | `print_table`, count estimation, smoothing, emission table, normalization checks |
| 8 | `logp`, `sequence_log_probability`, Q2 factor table and products |
| 11 | `viterbi`, `print_viterbi`, Q3 delta and backpointer tables |
| 14 | `ambiguous_emission`, ambiguity tables for $m=0.25$ and $m=0.40$ |
| 15 | Q4 Viterbi runs, transition stress, summary table |
| 18 | Four Q5 sentences, per-sentence tables, summary table |
| 20 | Three-panel figure: transition heatmap, emission heatmap, Q4 bar chart |
| 22 | Brute-force verification and pass/fail table |
| 24 | Viva summary |

The code uses only the standard library plus NumPy and Matplotlib:

```python
import itertools
import math
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
```

`itertools.product` is used only for the brute-force check. The tagger itself does not enumerate paths. `Counter` is used for counts. `math.log` and `math.exp` are used for log-space arithmetic. NumPy is used only to build the heatmap matrices.

To reproduce the notebook:

```bash
cd /home/mayookh/Dev/ai-concept-lab
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace lab-11/lab-11.ipynb
```

The notebook was executed with the project's `python3` kernel. All outputs shown in the notebook are real execution outputs, not hand-written values.

### 11.1 A note on display

The constants `START` and `END` are stored as `"<s>"` and `"</s>"`. The display helper

```python
def show_state(state):
    return "START" if state == START else "END" if state == END else state
```

maps them to readable labels in the count and probability tables. The underlying dictionaries still use the raw constants, so the scoring code and the display code do not diverge.

---

## 12. Limitations and what a real tagger adds

The notebook is a complete HMM implementation, but it is intentionally small. Its limitations are part of the lesson.

### 12.1 Sparse data

Nine sentences are enough to demonstrate the algorithm, not enough to estimate a reliable tagger. Many transitions are seen once or twice, and the add-$\alpha$ smoothing has a visible effect on the rows with small counts. A real system would use thousands or millions of annotated sentences.

### 12.2 Closed vocabulary

The emission table contains only the eight words in the corpus. An unknown word has probability zero under every tag, so no path can contain it. A real tagger needs an unknown-word model: suffix features, character-level models, morphological analysis, or a special `<UNK>` emission distribution.

### 12.3 Unsmooth emissions

The emissions are maximum-likelihood estimates with no smoothing. That is why most words have exactly one possible tag. In a realistic corpus, a word can appear with several tags, and emission smoothing is necessary to avoid zeros. Add-$\alpha$ smoothing could be applied to emissions with vocabulary size $|V|$ in the denominator.

### 12.4 First-order transitions

The model assumes $P(t_i\mid t_{i-1})$. A second-order HMM would use $P(t_i\mid t_{i-2},t_{i-1})$ and a larger state space. Higher-order models capture more context but need more data and more computation.

### 12.5 No morphology or agreement

Hindi verbs agree with their subjects in gender and number, and nouns carry case and oblique forms. The tagset here collapses all of that into `NN`, `VM`, and `VAUX`. A linguistically richer tagset would need morphological features and a model that can use them.

### 12.6 Joint versus marginal

Q2 computes a joint path probability, and Q3 maximizes it. Neither computes the marginal likelihood $P(W)$ or the posterior $P(T\mid W)$. The forward algorithm computes $P(W)$ by summing over paths; forward–backward computes posterior marginals. Viterbi returns the single best path, which is not the same as the sequence of per-position posterior modes. A tagger that wants per-token confidence should use forward–backward or a CRF.

### 12.7 Discriminative alternatives

An HMM is generative: it models $P(W,T)=P(T)P(W\mid T)$. A CRF models $P(T\mid W)$ directly and can use arbitrary overlapping features (word identity, suffixes, prefixes, neighboring words, morphology). A MEMM is discriminative at each transition. Neural taggers use contextual embeddings and sequence models such as BiLSTMs or transformers. The Viterbi recursion still appears at decoding time in many of those systems, but the scores come from a learned scorer rather than from count tables.

---

## 13. Viva quick reference

**What is the hidden state and what is the observation?**
The hidden state is the POS tag. The observation is the word. The tag sequence is hidden; the words are observed.

**What does the transition probability mean?**
$P(t_i\mid t_{i-1})$ is the probability of tag $t_i$ following tag $t_{i-1}$. It captures word-order regularities such as `VM→VAUX`.

**What does the emission probability mean?**
$P(w_i\mid t_i)$ is the probability of word $w_i$ given tag $t_i$. It captures lexical preferences such as `फूल` being more likely under `NN` than `बगीचे`.

**Why include START and END?**
They model sentence boundaries. `START` gives the initial tag distribution; `END` gives the final tag distribution. Without them, the first and last transitions would have no probability model.

**Why add-$\alpha$ smoothing to transitions?**
Maximum likelihood assigns zero to unseen transitions. A zero transition makes an entire path impossible. Add-$\alpha$ keeps every transition positive while preserving the ranking of observed counts. The denominator uses $|S|=7$ because `END` is a possible successor.

**Why not smooth the emissions?**
The notebook deliberately uses a closed vocabulary and maximum likelihood to keep the arithmetic simple and to make the effect of ambiguity explicit. A real tagger would smooth emissions or use an unknown-word model.

**Why does the joint probability use a product?**
The HMM assumptions make the transition and emission factors conditionally independent given the tags. The probability of a full path is therefore the product of the factors along the path.

**Why compute in log space?**
Products of many small probabilities underflow. Logs turn products into sums and make zero probabilities representable as $-\infty$.

**What is the difference between Q2 and Q3?**
Q2 evaluates one given path. Q3 searches over all paths for the maximum. Q2 is a scoring calculation; Q3 is dynamic programming.

**What does $\delta_i(j)$ store?**
The best joint probability of any tag prefix of length $i$ ending in tag $j$. It is the best score for that cell, not the sum over all prefixes.

**What does the backpointer store?**
The predecessor tag that achieved the maximum in $\delta_i(j)$. It is the information needed to reconstruct the path.

**Why is Viterbi better than greedy tagging?**
Greedy chooses the best tag at each position without considering future transitions. A locally good tag can lead to a bad transition later. Viterbi keeps the best prefix for every current tag, so the final path is globally optimal.

**What is the complexity of Viterbi?**
$O(n|S|^2)$ time and $O(n|S|)$ memory, where $n$ is the sentence length and $|S|$ is the number of tags.

**Why did सुंदर flip from JJ to NN in Q4?**
Because Viterbi maximizes a product of emissions and transitions. At $m=0.40$, the higher `NN` emission plus the favorable `NN→JJ` transition outweigh the higher `JJ` emission plus the weaker `JJ→JJ` transition. The decision is not made by the emission alone.

**Why is the total probability lower at $m=0.40$ even though the model is more confident about the NN reading?**
Because raising the `NN` mass for सुंदर removes mass from `फूल` and `बगीचे`, which lowers the probability of every path that uses those words. The model is choosing the best path under a different parameter table, not improving the absolute likelihood of the sentence.

**What is the marginal probability $P(W)$?**
The sum of $P(W,T)$ over all tag sequences. The brute-force check enumerates all $6^n$ sequences and would compute this sum if it accumulated rather than maximized. The forward algorithm computes it efficiently.

**What is the difference between Viterbi and forward–backward?**
Viterbi returns the single best path. Forward–backward computes marginal posterior probabilities for each tag at each position. The best path and the per-position posterior modes can differ.

**What would you improve first?**
Use a larger annotated corpus, add emission smoothing and an unknown-word model, and evaluate on held-out data. After that, consider a second-order HMM or a CRF with morphological features.

---

## 14. Minimal glossary

- **HMM** — hidden Markov model; a generative sequence model with hidden states and observed emissions.
- **Transition** — probability of moving from one hidden state to the next.
- **Emission** — probability of observing a word given a hidden state.
- **Joint probability** — $P(W,T)$, the probability of the words and one tag sequence together.
- **Marginal probability** — $P(W)$, the probability of the words after summing over all tag sequences.
- **Viterbi** — dynamic programming algorithm for the most likely hidden sequence.
- **Backpointer** — stored argmax predecessor used to reconstruct the Viterbi path.
- **Trellis** — the table of $\delta_i(j)$ values over positions and states.
- **Add-$\alpha$ smoothing** — adding a pseudocount $\alpha$ to every count so unseen events are not impossible.
- **Beam search** — a pruned approximation that keeps only the top $k$ states per position; not used here.
- **Forward algorithm** — dynamic programming algorithm that sums over paths to compute $P(W)$.
- **Forward–backward** — computes posterior marginals for each state at each position.
- **CRF** — conditional random field; a discriminative sequence model that models $P(T\mid W)$ directly.

---

## 15. Final takeaway

The lab is a small but complete example of probabilistic sequence labeling. The model is simple enough that every probability can be traced to a count, and the decoder is simple enough that every cell of the trellis can be printed. The important ideas are:

1. A tag sequence is scored by a product of local factors.
2. The product can be computed in log space without underflow.
3. Viterbi finds the global optimum by keeping the best prefix for every state.
4. Backpointers turn the score table into a path.
5. Ambiguity is resolved by the whole path, not by a single emission.
6. Brute-force enumeration is a useful correctness check for a dynamic program.

Those ideas generalize far beyond this eight-word vocabulary. The same Viterbi recursion appears in speech recognition, named-entity recognition, morphological analysis, and neural sequence models. What changes is the scorer, not the decoding principle.

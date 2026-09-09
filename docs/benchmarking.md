# Benchmarking

The claim this library rests on is that selecting a handful of tools beats sending all
of them. That claim is only worth making if it is measurable, so the benchmark ships
with the library rather than living in a blog post.

## On your own tools

```bash
toolbroker index -c toolbroker.yaml --out tools.json
```

Write `queries.jsonl`, one labelled case per line:

```jsonl
{"query": "customer wants their money back", "expected": ["billing/issue_refund"], "category": "goal"}
{"query": "the api key leaked", "expected": ["identity/rotate_api_key"], "category": "indirect"}
{"query": "who won the 1974 world cup", "expected": [], "category": "negative"}
```

```bash
toolbroker bench -c toolbroker.yaml queries.jsonl -k 5
```

A case with **no** expected tools is a negative case: the right answer is to return
nothing. Include them — a retriever that always returns its five best guesses looks
perfect on positive cases alone.

`category` is optional but worth setting. An aggregate score hides *which kinds* of
query fail, and that is the actionable part.

## Metrics

| Metric | Why |
|---|---|
| `recall@k` | Did the right tool make the cut? If it is not in the context, the agent cannot pick it. The one that matters. |
| `precision@k` | How much of what you sent was relevant. |
| `MRR` | How highly the first correct tool ranked. |
| `hit rate` | Fraction of queries where at least one expected tool appeared. |
| `context reduction` | Share of the catalogue kept out of the prompt. |

Every hit rate comes with a **95% Wilson confidence interval**. This exists so nobody
reads a two-point difference between runs as an improvement: if the intervals overlap,
the benchmark cannot tell the two apart. At 42 labelled queries a score of 0.79 has an
interval of ±0.12; at 290 it is ±0.05.

`summary()` prints failures by default. A benchmark that reports only its aggregate is a
marketing number.

## The reference suite

```bash
uv run python bench/generate.py
uv run python bench/run.py
```

Forty hand-authored tools plus filler, and **290 hand-written labelled queries** across
eight categories.

| embedder | tools | recall@5 | MRR | p95 ms |
|---|---|---|---|---|
| hashing (offline, lexical) | 200 | 0.466 | 0.408 | 0.38 |
| fastembed bge-small | 50 | 0.749 | 0.642 | 2.32 |
| fastembed bge-small | 200 | 0.702 | 0.604 | 2.55 |
| fastembed bge-small | 1000 | 0.668 | 0.580 | 3.75 |

The jump that matters is lexical to semantic — roughly 0.47 to 0.70 — which is why
`pip install 'toolbroker[fastembed]'` is the first thing to do after the quickstart.

## Which queries fail

At 200 tools with fastembed:

| category | n | hit rate | 95% CI |
|---|---|---|---|
| paraphrase | 50 | 1.000 | 0.929–1.000 |
| multi | 30 | 0.967 | 0.833–0.994 |
| distractor | 40 | 0.825 | 0.680–0.913 |
| jargon | 35 | 0.743 | 0.579–0.858 |
| indirect | 40 | 0.650 | 0.495–0.779 |
| ambiguous | 20 | 0.650 | 0.433–0.819 |
| goal | 50 | 0.620 | 0.482–0.741 |
| **negative** | 25 | **0.000** | 0.000–0.133 |

!!! warning "Paraphrase is solved and nothing else is"
    A benchmark made only of paraphrases — which is what most tool-retrieval demos
    show — reports 1.00 and tells you nothing. The categories that resemble how people
    actually talk (`goal`, `indirect`) sit around 0.62. Judge a retrieval change on
    those, not on the aggregate.

!!! danger "Negative queries fail completely by default"
    With no score floor the retriever always returns its `k` best guesses, so a question
    no tool can answer still produces five confident suggestions and the model uses one.
    Fix it with [`MinScore`](policy.md), and pick the threshold with
    `bench/threshold.py`.

## Finding your score floor

```bash
uv run python bench/threshold.py
```

| floor | overall | positive | negative | precision@5 |
|---|---|---|---|---|
| 0.00 | 0.717 | 0.785 | 0.000 | 0.163 |
| 0.50 | **0.734** | 0.774 | 0.320 | 0.193 |
| 0.55 | 0.724 | 0.725 | 0.720 | 0.242 |
| 0.60 | 0.621 | 0.600 | 0.840 | 0.321 |

ToolBroker ships **no default floor**, because the useful value tracks your embedder's
score distribution — a threshold tuned for bge-small would be wrong for another model.
On this same catalogue, noise tops out near 0.57 with bge-small and near 0.20 with the
offline hashing embedder.

Run the sweep against your own catalogue and pick from your own curve. If you have no
labelled data, `toolbroker calibrate` derives a floor from the catalogue alone — see
[Policy](policy.md#calibrating-without-labelled-data).

## End to end: does the *model* pick the right tool?

Everything above measures retrieval. `bench/e2e.py` measures the claim that actually
matters: given the tools it was handed, did the model call the right one?

```bash
uv run python bench/e2e.py --caller lexical                      # offline, no key
uv run python bench/e2e.py --caller openai --model gpt-4.1-mini
uv run python bench/e2e.py --caller anthropic --model claude-sonnet-5
```

Two conditions per query: **full** sends every tool in the catalogue, **toolbroker** sends
only the top `k`.

| tools | condition | accuracy | tool tokens / request |
|---|---|---|---|
| 50 | full | 0.334 | 2,145 |
| 50 | toolbroker k=5 | **0.438** | 217 |
| 200 | full | 0.324 | 8,128 |
| 200 | toolbroker k=5 | **0.441** | 212 |
| 1000 | full | 0.338 | 40,618 |
| 1000 | toolbroker k=5 | **0.445** | 210 |

Retrieval wins by about 0.11 absolute — a third better, relatively — at every size, and
costs **193× fewer tool tokens** at a thousand tools. That cost is per turn, on every
turn.

### Where the remaining loss comes from

A single accuracy number cannot say *whose* fault a failure is, and the two causes want
opposite fixes. Either no correct tool reached the model — retrieval's fault, and no
amount of model quality recovers it — or the right tool was sitting in the shortlist and
the model picked something else. The harness now separates them.

| tools | accuracy | retrieval misses | accuracy if retrieved |
|---|---|---|---|
| 50 | 0.438 | 37 / 290 (12.8%) | 0.502 |
| 200 | 0.441 | 50 / 290 (17.2%) | 0.533 |
| 1000 | 0.445 | 61 / 290 (21.0%) | **0.563** |

The flat headline was hiding two trends that cancel. Retrieval misses climb steadily with
catalogue size — 12.8% to 21.0% over a 20× range — while the caller gets *better* at every
step once the right tool is in front of it, because a five-tool shortlist drawn from a
thousand candidates is a cleaner choice than one drawn from fifty.

That makes the improvement target unambiguous. End-to-end accuracy at a thousand tools is
capped at 0.79 by recall@5 alone; every point of recall recovered is worth roughly half a
point end to end. Tuning the caller is worth less than tuning retrieval, which is
convenient, because retrieval is the part this library controls.

!!! note "A claim withdrawn"
    An earlier run on a 42-query set showed the `full` condition *degrading* as the
    catalogue grew. It does not reproduce on the 290-query set: `full` is flat at
    0.32–0.34 across a 20× size range. The earlier result was noise at that sample size.
    Context dilution may well hurt a real language model in a way it cannot hurt a
    deterministic caller — but that is now an untested hypothesis, not a finding.

### The ceiling nobody mentions

OpenAI rejects a request carrying more than 128 tools. Past a couple of hundred tools the
full-catalogue baseline is not merely worse, it is **unavailable**, and the harness
reports it that way rather than swallowing a wall of API errors.

Read the shape, not the absolutes: the `lexical` caller is a deterministic stand-in so
the harness runs in CI without an API key. The OpenAI and Anthropic callers are written
but **have not been run against a live API**.

## How much room is left, and where

`bench/e2e.py` says recall is the binding constraint. `bench/ceiling.py` says how much
of it is recoverable, by measuring how deep you must fetch before the right tool is even
in the pool.

```bash
uv run python bench/ceiling.py
uv run python bench/ceiling.py --sizes 1000 --misses   # name the hopeless cases
```

Recall@N over the 265 positive queries, fastembed bge-small:

| tools | @5 | @10 | @20 | @50 | @100 | @200 | unreachable |
|---|---|---|---|---|---|---|---|
| 50 | 0.860 | 0.921 | 0.958 | 1.000 | 1.000 | 1.000 | 0 (0.0%) |
| 200 | 0.811 | 0.857 | 0.906 | 0.955 | 0.992 | 0.992 | 2 (0.8%) |
| 1000 | 0.770 | 0.796 | 0.838 | 0.891 | 0.917 | 0.917 | 22 (8.3%) |

Read it as a budget. At a thousand tools, a reranker that overfetches 100 candidates and
reorders them perfectly would take recall@5 from 0.770 to 0.917 — that gap, **0.147**, is
the entire prize for any reranking strategy, and a real reranker collects a fraction of
it. That is a useful number to have before paying for a cross-encoder or an LLM call on
every query.

The last column is the more important one. **8.3% of queries never surface their
tool at any depth** — @200 is identical to @100, so the tool is not sitting just outside
the window, it is nowhere. No `k`, no fusion, no reranker recovers those, because
retrieval never had a signal to work with:

| query | expected tool |
|---|---|
| *someone left the company today* | `identity/deactivate_user` |
| *bounce the pods* | `infrastructure/restart_service` |
| *grep prod stderr* | `observability/query_logs` |
| *black friday traffic starts in an hour* | `infrastructure/scale_deployment` |

Six of the 22 are `jargon` and six are `indirect`. Nothing about "Deactivate a user
account." is close to "someone left the company today" in any embedding space, and a
better ranker cannot invent the association. **This is a description problem, and the fix
lives in the catalogue, not the retriever** — which is the honest thing for a retrieval
library to say about the part of the gap it cannot close.

### Two things that turned out not to be the answer

Both were measured before being ruled out, and both are the kind of change that is easy
to ship on intuition alone.

**Raising `k`.** The obvious lever, and it pays least where you need it most. Going
from `k=5` to `k=20` buys +0.098 at 50 tools but only +0.068 at 1000, because the
stranded 8.3% never arrive at any depth. Quadrupling the context for that is a bad
trade, and it gets worse as the catalogue grows — which is the opposite of what you
want from a lever you would reach for when the catalogue is large.

**Fusing BM25 into the retriever.** Recall@5 over the positive queries:

| retriever | 50 tools | 200 tools | 1000 tools |
|---|---|---|---|
| semantic (the default) | **0.860** | **0.811** | 0.770 |
| BM25 alone | 0.608 | 0.596 | 0.596 |
| RRF fusion | 0.792 | 0.743 | 0.762 |
| weighted 0.75 / 0.25 | 0.849 | 0.808 | 0.781 |
| weighted 0.50 / 0.50 | 0.830 | 0.781 | **0.785** |

Every weighting trails pure semantic at 50 and 200 tools, and the ordering only inverts
at 1000. The best small-catalogue fusion (0.75/0.25) is behind by 0.011 and 0.003 there
and ahead by 0.011 at a thousand; the even split is behind by 0.030 twice and ahead by
0.015 once. So the crossover is real and it is worth roughly one query in a hundred,
which does not justify changing a default that is better everywhere else. Semantic stays
the default, and fusion stays advice for large catalogues.

!!! note "And one artifact that was not to blame"
    The 1000-tool catalogue's filler had only 200 distinct descriptions — 960 of the
    1000 entries were six-way duplicates, and inspecting the results showed those
    families crowding the top 5. Rebuilding the filler with 1000 distinct descriptions
    moved recall@5 by **+0.011** for the semantic embedder and **−0.008** for the
    lexical one: noise at this sample size. The degradation with catalogue size is real
    and survives the most obvious objection to how the corpus is built.

## Selecting for turn five

Every other harness here scores a single sentence. Real agents do not get one: by turn five
the user says "cancel that one", and the referent was named three turns ago.

```bash
uv run python bench/multiturn.py
```

Thirty conversations whose final turn is deliberately under-specified, recall@5:

| tools | last turn only | with history |
|---|---|---|
| 50 | 0.533 | **0.667** |
| 200 | 0.467 | **0.633** |
| 1000 | 0.400 | **0.633** |

Worth +0.13 to +0.23, and **the gain grows with the catalogue** — more distractors means
more to disambiguate against. That is the largest single improvement measured in this
project apart from moving off the lexical embedder.

!!! warning "What this set cannot settle"
    Every conversation in it has exactly one prior user turn, so the `2 turns`, `3 turns`
    and `4 turns` rows come out identical to `1 turn` **by construction**. They are not
    evidence that depth stops helping, and `max_turns=2` is a judgement about real
    conversations being longer than the corpus, not a measurement. The current-turn
    `weight` moves recall by one or two queries out of thirty — below the resolution of a
    set that size.

## Reranking: measured, and not shipped

`bench/ceiling.py` says a perfect reranker over 100 overfetched candidates would take
recall@5 from 0.770 to 0.917 at a thousand tools. The obvious cheap way to collect some of
that is a lexical second stage — rescore a deep pool by term overlap, no model to download.
It does not work:

| tools | semantic | + lexical rerank (best config) |
|---|---|---|
| 200 | **0.811** | 0.815 |
| 1000 | 0.770 | **0.789** |

Best case is +0.019 at a thousand tools, and deeper pools *lose* at two hundred (0.811 down
to 0.777). Same shape as the hybrid-fusion result: helps only where the catalogue is
biggest, hurts everywhere else, and never by enough to justify a default.

The conclusion is the useful part. **That 0.147 of headroom needs semantic understanding**,
not another lexical signal — a cross-encoder or an LLM, which is what `LLMReranker` is for.
Nothing ships against it here because nothing here can be measured offline, and shipping an
unmeasured reranker into a project that publishes its numbers would be the wrong trade.

## Tuning with evidence

```bash
uv run python bench/ablate.py
```

Re-indexes under each enrichment variant and reports recall, so a tuning decision is a
measurement. It prints its own resolution — at 290 queries one flip moves recall by
0.003.

This is how the current `name_weight=1` default was chosen: it beat 2 and 3 for both the
lexical and the semantic embedder, at all three catalogue sizes. Two other candidate
changes were **declined** because they helped one embedder and hurt the other by a
similar margin — a trade-off is not a win, and adopting it would over-fit a default to
one model.

## What this data is and is not

Forty tools written to resemble real MCP servers: terse one-line descriptions, confusable
near-neighbours within a domain, and cross-domain lexical traps. The 290 queries are
hand-written, not templated — an earlier templated version reported accuracy about 0.10
higher.

It is still not a substitute for a real deployment. One person wrote both the tools and
the queries, so both carry that person's assumptions about phrasing, and the labels have
not been independently reviewed. The right next step is queries collected from real agent
traffic, labelled by more than one person.

Treat these numbers as a reproducible baseline for comparing *changes to ToolBroker*,
which is what they are good at — not as a prediction of accuracy on your catalogue. For
that, run `toolbroker bench` on your own tools.

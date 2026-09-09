# Benchmark

The claim this library rests on is that selecting a handful of tools beats sending all
of them. This directory is how that claim gets checked — by us, and by you against your
own catalogue.

## Run it on your own tools

```bash
toolbroker index -c toolbroker.yaml --out tools.json
# write queries.jsonl: {"query": "...", "expected": ["namespace/tool"]}
toolbroker bench -c toolbroker.yaml queries.jsonl -k 5
```

## Run the synthetic suite

```bash
uv run python bench/generate.py           # writes catalogues and query sets
uv run python bench/run.py                # scores retrieval at 50 / 200 / 1000 tools
```

## Run the end-to-end suite

`run.py` scores retrieval. `e2e.py` scores what the *model* does with what retrieval
handed it — the claim the project actually rests on.

```bash
uv run python bench/e2e.py --caller lexical                      # offline, no key
uv run python bench/e2e.py --caller openai --model gpt-4.1-mini
uv run python bench/e2e.py --caller anthropic --model claude-sonnet-5
uv run python bench/e2e.py --caller openai --out results.md      # chart-ready table
```

Two conditions per query: `full` sends the whole catalogue, `toolbroker` sends the top
`k`. With the offline caller, 290 queries, `k=5`:

| tools | condition | accuracy | tool tokens / request |
|---|---|---|---|
| 50 | full | 0.334 | 2,145 |
| 50 | toolbroker k=5 | **0.438** | 217 |
| 200 | full | 0.324 | 8,128 |
| 200 | toolbroker k=5 | **0.441** | 212 |
| 1000 | full | 0.338 | 40,618 |
| 1000 | toolbroker k=5 | **0.445** | 210 |

Retrieval wins by ~0.11 absolute (a third better, relatively) at every size, and costs
**193x fewer tool tokens** at a thousand tools — per turn, on every turn.

The report also splits the remaining loss by cause, because the two want opposite fixes:
a *retrieval miss* means no correct tool reached the model at all, and everything else
means it had the right tool in the shortlist and chose otherwise.

| tools | accuracy | retrieval misses | accuracy if retrieved |
|---|---|---|---|
| 50 | 0.438 | 37 / 290 (12.8%) | 0.502 |
| 200 | 0.441 | 50 / 290 (17.2%) | 0.533 |
| 1000 | 0.445 | 61 / 290 (21.0%) | **0.563** |

The flat headline hides two trends that cancel: misses climb with catalogue size while
the caller does *better* per shortlist, because five tools drawn from a thousand are a
cleaner choice than five drawn from fifty. Recall is therefore the binding constraint —
at a thousand tools it caps end-to-end accuracy at 0.79 — and each point of recall
recovered is worth roughly half a point end to end.

> **A claim withdrawn.** An earlier run on a 42-query set showed the `full` condition
> *degrading* as the catalogue grew. It does not reproduce here: `full` is flat at
> 0.32–0.34 across a 20x size range. The earlier result was noise at that sample size.
> Context dilution may well hurt a real language model in a way it cannot hurt this
> deterministic caller — but that is now an untested hypothesis, not a finding.

`full` becomes **not runnable** against OpenAI past 128 tools; the harness reports that
rather than swallowing API errors.

The provider callers have not been run against a live API in this repo — no key has been
used here. Treat the OpenAI and Anthropic paths as unverified until someone runs them
and commits the numbers.

## Results

All numbers below are from `bench/`, reproducible with the commands in this file. The
catalogue is 40 hand-authored tools plus filler; the labelled set is **290 hand-written
queries** across eight categories.

### Retrieval accuracy

```bash
uv run python bench/generate.py && uv run python bench/run.py
```

| embedder | tools | recall@5 | MRR | p95 ms |
|---|---|---|---|---|
| hashing (offline, lexical) | 50 | 0.515 | — | 0.19 |
| hashing (offline, lexical) | 200 | 0.466 | — | 0.38 |
| hashing (offline, lexical) | 1000 | 0.431 | — | 1.37 |
| fastembed bge-small | 50 | 0.749 | 0.642 | 2.32 |
| fastembed bge-small | 200 | 0.702 | 0.604 | 2.55 |
| fastembed bge-small | 1000 | 0.668 | 0.580 | 3.75 |

### Which queries fail

The aggregate hides the answer. Per category, at 200 tools with fastembed:

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

Two things worth taking from this.

**Paraphrase is solved and nothing else is.** A benchmark made only of paraphrases —
which is what most tool-retrieval demos show — reports 1.00 and tells you nothing. The
categories that resemble how people actually talk (`goal`, `indirect`) sit around 0.62.

**Negative queries fail completely by default.** With no score floor the retriever always
returns its five best guesses, so a question no tool can answer still produces five
confident suggestions. See `bench/threshold.py`.

## How much room is left

`e2e.py` shows recall is the binding constraint. `ceiling.py` shows how much of it any
reranker could ever recover, by measuring how deep you must fetch before the right tool
is in the pool at all.

```bash
uv run python bench/ceiling.py
uv run python bench/ceiling.py --sizes 1000 --misses    # name the hopeless cases
```

Recall@N over the 265 positive queries, fastembed bge-small:

| tools | @5 | @10 | @20 | @50 | @100 | @200 | unreachable |
|---|---|---|---|---|---|---|---|
| 50 | 0.860 | 0.921 | 0.958 | 1.000 | 1.000 | 1.000 | 0 (0.0%) |
| 200 | 0.811 | 0.857 | 0.906 | 0.955 | 0.992 | 0.992 | 2 (0.8%) |
| 1000 | 0.770 | 0.796 | 0.838 | 0.891 | 0.917 | 0.917 | 22 (8.3%) |

At a thousand tools, overfetching 100 and reranking them *perfectly* would take recall@5
from 0.770 to 0.917. That 0.147 gap is the whole prize for any reranking strategy — worth
knowing before paying for a cross-encoder or an LLM call per query.

**8.3% of queries never surface their tool at any depth**, and @200 equals @100, so those
tools are not just outside the window. `bounce the pods` → `infrastructure/restart_service`
and `someone left the company today` → `identity/deactivate_user` have no signal for any
ranker to find. That share is a description problem, not a retrieval one, and it is the
part of the gap this library cannot close from its own side.

### Ruled out, with numbers

- **Raising `k`** flattens where it is needed most: +0.098 from `k=5` to `k=20` at 50
  tools, only +0.068 at 1000, because the stranded 8.3% never arrive.
- **Fusing BM25** trails pure semantic at 50 and 200 tools under every weighting tested
  and only crosses over at 1000 (best fusion 0.785 vs semantic 0.770). Real, but worth
  about one query in a hundred, and it costs more than that at the other two sizes.
- **The duplicated filler was not to blame.** The 1000-tool catalogue had 200 distinct
  descriptions across 1000 entries, six-way duplicates crowding the top 5. Rebuilding it
  with 1000 distinct descriptions moved recall@5 by +0.011 (semantic) and −0.008
  (lexical) — noise. The scaling degradation survives the obvious objection.

## Does a bigger embedder help?

`bench/run.py` compares every embedder available on the machine. Optional ones are opted
into by environment, because one needs a running server and the other costs money:

```bash
TOOLBROKER_BENCH_OLLAMA_MODEL=nomic-embed-text uv run python bench/run.py
OPENAI_API_KEY=sk-... uv run python bench/run.py
```

The jump that matters is lexical to semantic: roughly 0.47 to 0.70 recall@5 at 200
tools. On the earlier, easier corpus a 4B-parameter local model tied bge-small; that
comparison needs re-running against this corpus before it is worth quoting.

## Ablations

```bash
uv run python bench/ablate.py
```

Re-indexes the catalogue under each enrichment variant and reports recall, so a tuning
decision is a measurement rather than an opinion. CONTRIBUTING asks for before-and-after
numbers on any change to the selection pipeline; this produces them.

It prints its own resolution — at 290 queries one flip moves recall by 0.003, against
0.024 on the old 42-query set. That eightfold improvement is what made the results below
readable at all.

### One change made

`name_weight` went from **2 to 1**. Repeating the tool name dilutes the description
without adding signal:

| variant | fastembed mean | hashing mean |
|---|---|---|
| name x1 *(new default)* | **0.706** | **0.470** |
| name x2 *(old default)* | 0.681 | 0.455 |
| name x3 | 0.668 | 0.445 |
| drop the name entirely | 0.686 | 0.389 |

Better for both embedders, at all three catalogue sizes, monotonically. Dropping the
name outright costs the lexical embedder 0.08, so 1 is the peak rather than the start of
a slide toward 0.

### Two changes declined

Removing the namespace line and doubling the description weight each helped one embedder
and hurt the other by a similar margin:

| variant | fastembed | hashing |
|---|---|---|
| no namespace line | +0.013 | −0.011 |
| description x2 | −0.012 | +0.009 |

A change that improves one model and degrades another is a trade-off, not a win, and
adopting it would be over-fitting a default to one embedder on one synthetic corpus.
Both were left alone.

### Is your usage signal good enough?

```bash
uv run python bench/usage.py                        # synthetic signals
uv run python bench/usage.py --usage my_usage.json  # your own counts
```

Boosting is a bet on your feedback. Measured on the reference catalogue, baseline
recall@5 = 0.702:

| signal | weight | recall@5 | delta |
|---|---|---|---|
| informed *(usage tracks the labels)* | 0.05 | 0.739 | +0.037 |
| informed | **0.10** | **0.757** | **+0.056** |
| informed | 0.20 | 0.752 | +0.051 |
| informed | 0.40 | 0.751 | +0.049 |
| random *(usage unrelated to relevance)* | 0.05 | 0.689 | −0.013 |
| random | **0.10** | 0.649 | **−0.053** |
| random | 0.20 | 0.540 | −0.161 |
| random | 0.40 | 0.322 | −0.380 |

The asymmetry is the point, and it is why `usage_weight` defaults to **0.1**. The gain
from a good signal *saturates* around 0.1 — going to 0.4 buys nothing. The damage from a
bad signal keeps growing: −0.05 at 0.1, −0.38 at 0.4. A higher weight is all downside.

So: record the tool that actually **worked**, not every tool the model tried. If you
cannot tell the difference, leave boosting off — no signal beats a misleading one.

## Finding your score floor

```bash
uv run python bench/threshold.py
```

Sweeps `MinScore` thresholds and reports the trade-off between answering positive
queries and correctly declining negative ones. At 200 tools with fastembed:

| floor | overall | positive | negative | precision@5 |
|---|---|---|---|---|
| 0.00 | 0.717 | 0.785 | 0.000 | 0.163 |
| 0.50 | **0.734** | 0.774 | 0.320 | 0.193 |
| 0.55 | 0.724 | 0.725 | 0.720 | 0.242 |
| 0.60 | 0.621 | 0.600 | 0.840 | 0.321 |

**ToolBroker ships no default floor**, because the useful value tracks your embedder's
score distribution — a threshold tuned for bge-small would be wrong for a different
model. Run this against your own catalogue and pick from your own curve.

## What this data is and is not

The catalogue is 40 hand-authored tools written to resemble real MCP servers: terse
one-line descriptions, confusable near-neighbours within a domain (`void_invoice` and
`cancel_shipment` both answer to "cancel it"), and cross-domain lexical traps
(`deactivate_user` and `close_ticket` both "close" something). Filler tools pad the
catalogue to size without adding labels.

The 290 queries are hand-written, not templated. The earlier version of this benchmark
generated paraphrases from templates, which measured string similarity and reported
accuracy about 0.10 higher than this set does.

It is still **not** a substitute for a real deployment. One person wrote both the tools
and the queries, so both carry that person's assumptions about phrasing, and the labels
have not been reviewed by anyone else. The right next step is queries collected from
real agent traffic against a real catalogue, labelled by more than one person.

Treat these numbers as a reproducible baseline for comparing *changes to ToolBroker*,
which is what they are good at, rather than as a prediction of accuracy on your
catalogue. For that, run `toolbroker bench` on your own tools.

## Metrics

| Metric | Why it is here |
|---|---|
| `recall@k` | Did the right tool make the cut? If it isn't in the context, the agent cannot pick it. This is the one that matters. |
| `precision@k` | How much of what we sent was relevant. |
| `MRR` | How highly the first correct tool ranked. |
| `hit rate` | Fraction of queries where at least one expected tool appeared. |
| `context reduction` | Share of the catalogue kept out of the prompt. |

Negative cases — queries where the right answer is *no tool* — are included
deliberately. A retriever that always returns its five best guesses looks perfect on
positive cases alone.

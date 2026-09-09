# Tool selection past 100 tools: what I measured, and what I got wrong

*Working draft, deliberately kept out of the docs site. The provider numbers in §6 are
the one thing still missing — see the gaps section at the end before publishing.*

---

Agent frameworks assume your tool list fits in the prompt. MCP made it trivial to attach
hundreds of tools and did nothing to help you choose among them. So I built the retrieval
and policy layer that sits in between, and — more usefully — I measured it, including the
parts where it loses.

This is that write-up. Forty hand-authored tools, 290 hand-written labelled queries across
eight categories, and a harness anyone can run. Every number below is reproducible from
`bench/`.

## 1. Retrieval beats the full catalogue, by about 0.11

Two conditions per query: send every tool, or send the five that retrieval picked.

| tools | full catalogue | top-5 | tool tokens (full) | tool tokens (top-5) |
|---|---|---|---|---|
| 50 | 0.334 | **0.438** | 2,145 | 217 |
| 200 | 0.324 | **0.441** | 8,128 | 212 |
| 1000 | 0.338 | **0.445** | 40,618 | 210 |

About a third better, at **193× less context**, per turn, on every turn.

There is also a ceiling nobody mentions: OpenAI rejects any request carrying more than 128
tools. Past a couple of hundred tools the full-catalogue baseline is not worse, it is
unavailable.

## 2. A claim I withdrew

An earlier run on 42 templated queries showed the full-catalogue condition *degrading* as
the catalogue grew — the tidiest possible result for a library like this.

It does not reproduce. On 290 hand-written queries, `full` is flat at 0.32–0.34 across a
20× size range. The earlier result was noise at that sample size, and the harness now
prints its own resolution so nobody reads a +0.02 as an improvement.

Context dilution may well hurt a real model in ways it cannot hurt a deterministic
stand-in. But that is now an untested hypothesis, not a finding.

## 3. Where the remaining loss actually comes from

A single accuracy number cannot say *whose* fault a failure is, and the two causes want
opposite fixes. Either no correct tool reached the model — retrieval's fault — or the
right tool was in the shortlist and the model picked something else.

| tools | accuracy | retrieval misses | accuracy if retrieved |
|---|---|---|---|
| 50 | 0.438 | 12.8% | 0.502 |
| 200 | 0.441 | 17.2% | 0.533 |
| 1000 | 0.445 | 21.0% | **0.563** |

The flat headline was hiding two trends that cancel. Misses climb with catalogue size,
while the caller gets *better* per shortlist — five tools drawn from a thousand are a
cleaner choice than five drawn from fifty.

That settles where effort goes. Recall is the binding constraint.

## 4. The 8.3% that no ranking change can fix

So how much recall is recoverable? Measure how deep you must fetch before the right tool
is in the pool at all:

| tools | @5 | @20 | @100 | @200 | never |
|---|---|---|---|---|---|
| 50 | 0.860 | 0.958 | 1.000 | 1.000 | 0% |
| 200 | 0.811 | 0.906 | 0.992 | 0.992 | 0.8% |
| 1000 | 0.770 | 0.838 | 0.917 | 0.917 | **8.3%** |

A perfect reranker over 100 candidates would take recall@5 from 0.770 to 0.917. That
0.147 is the entire prize for any reranking strategy — worth knowing before paying for a
cross-encoder on every query.

The last column is the more interesting one. **8.3% of queries never surface their tool at
any depth**, and @200 equals @100, so the tool is not sitting just outside the window — it
is nowhere:

| query | expected tool |
|---|---|
| *bounce the pods* | `restart_service` |
| *someone left the company today* | `deactivate_user` |
| *grep prod stderr* | `query_logs` |

Nothing about "Restart a running service." is near "bounce the pods" in any embedding
space, and a better ranker cannot invent the association. This is a description problem,
and the fix lives in the catalogue rather than the retriever. That is the honest thing for
a retrieval library to say about the part of the gap it cannot close.

## 5. Three things I measured and did not ship

The results that cost the most time are the ones that changed nothing.

**A bigger embedder buys almost nothing.** The jump that matters is lexical → semantic
(0.31 → 0.79 recall). Past that, a 4B-parameter 2560-dim model *ties* bge-small on
recall@5 at 200 tools and costs ~34× the latency. `fastembed` stays the recommended
default.

**Hybrid retrieval loses more than it wins.** Every BM25 fusion weighting trails pure
semantic at 50 and 200 tools and only crosses over at 1000 (0.785 vs 0.770). Real, worth
about one query in a hundred, and it costs more than that at the other two sizes.

**Lexical reranking does not convert the headroom.** Rescoring a deep pool by term overlap
gains +0.019 at 1000 tools and *loses* 0.03 at 200. That 0.147 needs semantic
understanding, not another lexical signal.

And one methodological scare that came to nothing: the 1000-tool catalogue's filler had
only 200 distinct descriptions across 1000 entries — six-way duplicates, visibly crowding
the top 5 for exactly the queries that were failing. It looked like the whole scaling
result might be an artifact of the generator. Rebuilding with 1000 distinct descriptions
moved recall by +0.011 and −0.008 depending on embedder: noise. The degradation is real
and survives the most obvious attack on how the corpus is built.

## 6. Real models

*This section is the one that isn't written yet. See the gaps note at the end.*

## 7. Conversation history is worth more than any tuning

Every benchmark above scores a single sentence. Real agents do not get one: by turn five
the user says "check stock there", and *there* was named three turns ago.

| tools | last turn only | with one turn of history |
|---|---|---|
| 50 | 0.533 | **0.667** |
| 200 | 0.467 | **0.633** |
| 1000 | 0.400 | **0.633** |

+0.13 to +0.23, and the gain *grows* with the catalogue. Larger than every tuning decision
in this project combined, and larger than the entire reranking headroom.

Caveat, because it matters: every conversation in that set has exactly one prior user
turn, so it cannot tell you how much history is too much. The depth knob is a judgement,
not a measurement, and the docs say so.

## 8. The part frameworks won't build

Retrieval is half of it. The other half is that nothing in the stack says *"this agent may
never call `delete_*`"*, and that turns out to be where a middleman can do things nothing
else can — because it is the only component that sees the catalogue **across time**.

**It notices when a tool changes after you approved it.** The injection risk people
discuss is a hostile description on day one. The one that gets you is a description that
changes *later*, on a server you already reviewed. The refresh diff already computed this
to decide what to re-embed; now a changed tool keeps serving its last approved version
until a human accepts the new one.

**It tells you what a policy change grants, before it ships.** Nobody can review
`deny: ["*/delete_*"] → ["*/delete_user"]` by reading it. Policy is deterministic over a
known catalogue, so `toolbroker diff main.yaml pr.yaml --fail-on-high-risk` answers it in
CI:

```
agent `support`
  ⚠ + billing/delete_order  (risk: high)
```

## 9. Eight bypasses, found by writing the attack

A library whose entire job is gating access does not get to write a document about
hypothetical vulnerabilities. I went looking instead, twice, and found eight. All
pre-release, all fixed with regression tests. The instructive ones:

- **Deny globs matched case-sensitively.** `*/delete_*` did not match `DeleteUser`, which
  plenty of APIs generate. Nothing anywhere indicated the rule was inert.
- **`first_match` short-circuited tenant isolation.** An `allow` glob earlier in the list
  granted another tenant's tool before the boundary ran. A security guarantee that depends
  on rule order is not a guarantee.
- **A renamed tool walked past quarantine.** A rename has a different id, so the diff sees
  a removal plus an addition rather than a change — and the flag meant to stop it declined
  to *approve* new tools while still serving them.

Two of the three were features that existed and did not do what their names implied. That
is the argument for doing this every time the gating surface grows, not once before
release.

The finding I cannot fix is worth stating plainly: **prompt injection through tool
descriptions**. A hostile server puts instructions in a description and that text reaches
the model; no filter detects this reliably. What a broker can do is bound the blast radius,
because policy is enforced in Python rather than by asking the model nicely. A hijacked
model still only reaches tools the operator permitted.

## What this data is not

The catalogue is synthetic — 40 hand-authored tools plus generated filler — and the
queries are mine. That is the right objection to raise, and the reason the harness, the
labelled data, and the raw results all ship in the repo: `toolbroker bench` runs against
your own tools and your own queries, and I would rather be corrected than believed.

---

## Gaps to close before publishing

1. **§6 is empty.** The OpenAI and Anthropic callers in `bench/e2e.py` are written and have
   never been run against a live API. This is the section people will read first.
2. The repo URLs point at a GitHub account that does not exist yet.
3. Decide whether to lead with §1 (the pitch) or §4 (the honest ceiling). §4 is more
   interesting and more credible; §1 is what people came for.

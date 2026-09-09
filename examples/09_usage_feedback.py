"""Learning from which tools actually get called.

Retrieval sees a description. Usage sees reality: the tool whose description
reads best is not always the one that works. ToolBroker never executes a tool,
so you report the calls; it folds them into ranking, bounded so a popular tool
can never displace a clearly more relevant one.

    python examples/09_usage_feedback.py
"""

import tempfile
from pathlib import Path

from toolbroker import Tool, ToolBroker, UsageTracker

CATALOGUE = [
    Tool(
        name="customer_lookup",
        namespace="legacy",
        description="Customer lookup. Deprecated, kept for old integrations.",
    ),
    Tool(
        name="find_customer_records",
        namespace="crm",
        description="Search for a customer by name, email, or company.",
    ),
    Tool(name="reset_password", namespace="identity", description="Send a reset link."),
]

QUERY = "customer lookup"

# A clock the example controls, so decay can be shown without waiting.
now = [0.0]

# Half-life in seconds. Sixty here so decay is visible; thirty days is the
# default and a sane production value.
tracker = UsageTracker(half_life=60.0, clock=lambda: now[0])
broker = ToolBroker(usage=tracker, usage_weight=0.3)
broker.index(CATALOGUE)


def show(label: str) -> None:
    print(label)
    for rank, hit in enumerate(broker.select(QUERY, k=2).hits, start=1):
        boost = hit.components.get("usage_boost", 0.0)
        suffix = f"  (usage boost +{boost:.4f})" if boost else ""
        print(f"  {rank}. {hit.id:<28} {hit.score:.4f}{suffix}")


print(f"query: {QUERY!r}\n")
show("the deprecated tool matches the words better, so it ranks first:")

# Your framework runs the loop; when the model calls a tool, report it.
print("\n...but users keep choosing the modern one. 40 calls reported...\n")
for _ in range(40):
    broker.record_use("crm/find_customer_records")

show("usage overturns the wording:")

# The bound: a boost adds at most `usage_weight` of a hit's own score, so a
# popular tool cannot overtake one that is clearly more relevant.
print("\nthe boost is capped, so relevance still wins where it matters:")
broker.record_use("identity/reset_password", 10_000)
print(
    f"  10,000 calls to identity/reset_password -> top hit is still "
    f"{broker.select(QUERY, k=1).tool_ids[0]}"
)

print("\ncounts decay, so last quarter's favourite stops dominating:")
print(f"  now:                 {tracker.count('crm/find_customer_records'):.1f}")
now[0] += 120  # two half-lives
print(f"  two half-lives on:   {tracker.count('crm/find_customer_records'):.1f}")

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "usage.json"
    tracker.save(path)
    # Same clock on the way back in: timestamps are absolute, so a tracker
    # saved on one time source and loaded on another sees a huge elapsed gap
    # and decays everything away. Production uses the real clock throughout.
    restored = UsageTracker.load(path, clock=lambda: now[0])
    print("\npersisted across restarts, decay included:")
    print(f"  reloaded {len(restored)} tools from {path.name}")
    print(
        f"  crm/find_customer_records is still at {restored.count('crm/find_customer_records'):.1f}"
    )

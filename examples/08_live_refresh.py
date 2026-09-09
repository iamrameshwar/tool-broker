"""Keeping a catalogue current while the process runs.

A server that indexes once at startup serves a stale catalogue forever. This
shows incremental refresh: only what changed is re-embedded, and a source that
goes down does not take its tools with it.

    python examples/08_live_refresh.py
"""

from toolbroker import PeriodicRefresher, RiskTier, Tool, ToolBroker
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.sources.base import BaseSource


class FakeMCPServer(BaseSource):
    """Stands in for an MCP server whose tool list changes underneath us."""

    def __init__(self) -> None:
        super().__init__("mcp:orders")
        self.tools = [
            Tool(name="search_orders", namespace="orders", description="Find customer orders."),
            Tool(name="track_shipment", namespace="orders", description="Where is a parcel."),
        ]
        self.down = False

    def _discover(self):
        if self.down:
            raise ConnectionError("connection refused")
        return list(self.tools)


class CountingEmbedder:
    """Counts embedding calls, because that is what refresh is trying to avoid."""

    def __init__(self) -> None:
        self._inner = HashingEmbedder(dim=256)
        self.embedded = 0

    dim = property(lambda self: self._inner.dim)
    id = property(lambda self: "counting")

    def embed(self, texts):
        self.embedded += len(texts)
        return self._inner.embed(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)


server = FakeMCPServer()
embedder = CountingEmbedder()
broker = ToolBroker(embedder=embedder, cache_embeddings=False)
broker.add_source(server)

print("initial index:")
print(f"  {broker.refresh().summary()}   embedding calls: {embedder.embedded}")

embedder.embedded = 0
print("\nnothing changed:")
print(f"  {broker.refresh().summary()}   embedding calls: {embedder.embedded}")

embedder.embedded = 0
server.tools.append(Tool(name="issue_refund", namespace="orders", description="Refund a payment."))
print("\nthe server gained one tool:")
print(f"  {broker.refresh().summary()}   embedding calls: {embedder.embedded}")

embedder.embedded = 0
server.tools[0] = Tool(
    name="search_orders",
    namespace="orders",
    description="Find customer orders.",
    risk=RiskTier.HIGH,
)
print("\nonly the risk tier changed -- policy must see it, but the vector is fine:")
report = broker.refresh()
print(f"  {report.summary()}   embedding calls: {embedder.embedded}")
print(f"  risk is now: {broker.get('orders/search_orders').risk.value}")

print("\nthe server goes down:")
server.down = True
report = broker.refresh()
print(f"  {report.summary()}")
print(f"  tools still available: {len(broker)}")
print("  a brief outage must not silently strip capabilities from every agent.")

server.down = False
server.tools = server.tools[:1]
print("\nback up, and one tool genuinely removed:")
print(f"  {broker.refresh().summary()}")

print("\nin a long-running process, drive it from a background thread:")
with PeriodicRefresher(broker, interval=300) as refresher:
    print(f"  {refresher!r}")

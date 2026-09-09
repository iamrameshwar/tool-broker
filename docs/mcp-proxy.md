# MCP proxy mode

The zero-code integration path.

Instead of attaching forty MCP servers to a client and drowning it in five hundred tool
definitions, attach one:

```bash
pip install 'toolbroker[mcp,yaml]'
toolbroker serve -c toolbroker.yaml
```

Any MCP client — Claude Desktop, an IDE, your own agent — now sees three tools:

| Tool | Purpose |
|---|---|
| `search_tools(query, limit)` | The retrieval and policy layer, exposed as a tool |
| `describe_tool(tool_id)` | The full input schema for one result |
| `call_tool(tool_id, arguments)` | Forwards the call to the server that owns it |

The two-step shape is what keeps the context small: the client permanently holds three
definitions and pulls real schemas only for the handful a query actually surfaced.

## Configuration

```yaml
embedder:
  name: fastembed
retrieval:
  mode: hybrid
sources:
  - {type: mcp, name: github, command: npx, args: ["-y", "@modelcontextprotocol/server-github"]}
  - {type: mcp, name: slack,  command: npx, args: ["-y", "@modelcontextprotocol/server-slack"]}
  - {type: mcp, name: internal, url: "https://tools.internal/mcp"}
policy:
  default_k: 5
  default:
    max_tools: 5
    deny: ["*/delete_*"]
```

Then point your client at it:

```json
{
  "mcpServers": {
    "toolbroker": {"command": "toolbroker", "args": ["serve", "-c", "/path/to/toolbroker.yaml"]}
  }
}
```

## In Python

```python
from toolbroker.proxy import ToolBrokerProxy

proxy = ToolBrokerProxy(broker, default_k=5, agent="support", scopes=frozenset({"read"}))
proxy.run(transport="stdio")  # or transport="http", host=..., port=...
```

## Staying current

A proxy outlives the servers behind it. Without a refresh interval it serves whatever
tools existed at startup, forever:

```bash
toolbroker serve -c toolbroker.yaml --refresh 300
```

```python
ToolBrokerProxy(broker, refresh_interval=300)
```

Re-discovery is incremental, so a server adding one tool costs one embedding call rather
than re-embedding the catalogue. A server that is unreachable keeps its tools rather than
having them silently deleted — see
[Keeping the catalogue current](concepts.md#keeping-the-catalogue-current).

## Discovery-only mode

```python
ToolBrokerProxy(broker, allow_calls=False)
```

`call_tool` is not exposed at all. Execution stays on the client's own connections to
the underlying servers; ToolBroker only tells it what to call.

## Security

**Policy is enforced at call time, not just at search time.** A client can invoke
`call_tool` with any id it likes, including one it learned somewhere else, so the
policy is re-evaluated on every forwarded call:

```text
ToolBrokerError: policy denies calling 'ops/delete_database':
  max_risk: risk critical exceeds ceiling high
```

!!! note "This does not violate the no-orchestration rule"
    Forwarding a single call the client already decided to make is transport.
    ToolBroker still runs no agent loop, makes no planning decision, and never chains
    calls together.

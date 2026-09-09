# toolbroker-claude-agent

[Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview) adapter for
[ToolBroker](https://github.com/iamrameshwar/tool-broker).

```bash
pip install toolbroker-claude-agent
```

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from toolbroker import ToolBroker
from toolbroker_claude_agent import ClaudeAgentAdapter

broker = ToolBroker()
broker.add_functions([search_orders, issue_refund, check_inventory])
broker.index()

adapter = ClaudeAgentAdapter()
selection = broker.select("customer wants a refund", k=3)

options = ClaudeAgentOptions(
    mcp_servers={"tools": adapter.create_server(selection.tools)},
    allowed_tools=adapter.allowed_tool_names(selection.tools, server="tools"),
)
```

`render()` returns `SdkMcpTool` objects; `create_server()` wraps them into the in-process
MCP server config the SDK expects.

## Why `allowed_tool_names` exists

The SDK gates tools by name in the form `mcp__<server>__<tool>`. Getting that string
wrong means the model sees the tool and is then refused permission to call it — a
confusing failure. The helper builds the names from the same selection, so the two
cannot drift apart.

## Result shape

Handlers must return MCP content blocks. Anything your function returns is wrapped
automatically: a `dict` that already has `content` passes through untouched, and
everything else becomes a single text block. Override with `result_formatter=` if you
need image or resource blocks.

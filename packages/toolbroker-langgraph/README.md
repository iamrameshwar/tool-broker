# toolbroker-langgraph

LangChain / LangGraph adapter for [ToolBroker](https://github.com/iamrameshwar/tool-broker).

```bash
pip install toolbroker-langgraph        # or: pip install 'toolbroker[langgraph]'
```

```python
from langgraph.prebuilt import create_react_agent
from toolbroker import ToolBroker

broker = ToolBroker()
broker.add_functions([search_orders, issue_refund])
broker.index()

tools, selection = broker.select_for("langgraph", "customer wants a refund", k=3)
agent = create_react_agent(model, tools)
```

Emits `StructuredTool` objects, which both `ToolNode` and `bind_tools` accept, so the
same output works for a graph or a plain LangChain agent.

## Remote tools

A tool backed by a local Python function is wired directly. A tool served by an MCP
server has no local callable, so pass an `invoker` rather than letting the adapter
invent one:

```python
from toolbroker_langgraph import LangGraphAdapter

adapter = LangGraphAdapter(invoker=lambda tool, kwargs: my_mcp_client.call(tool, kwargs))
```

Producing a tool that raises at call time would be worse than failing here, so the
adapter refuses unless you pass `invoker=` or `skip_uninvokable=True`.

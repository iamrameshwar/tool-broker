# toolbroker-openai-agents

[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) adapter for
[ToolBroker](https://github.com/iamrameshwar/tool-broker).

```bash
pip install toolbroker-openai-agents
```

```python
from agents import Agent, Runner
from toolbroker import ToolBroker

broker = ToolBroker()
broker.add_functions([search_orders, issue_refund, check_inventory])
broker.index()

tools, selection = broker.select_for("openai-agents", "customer wants a refund", k=3)
agent = Agent(name="support", instructions="Help the customer.", tools=tools)
result = await Runner.run(agent, "I want my money back")
```

Emits `FunctionTool` objects with the tool's own JSON Schema.

## Strict mode

`strict_json_schema` defaults to **False** here, unlike the SDK's own default. Strict
mode requires every property to be required and `additionalProperties: false`, which
most real MCP and OpenAPI schemas do not satisfy — turning it on by default would
reject perfectly good tools at runtime. Opt in when your schemas comply:

```python
OpenAIAgentsAdapter(strict=True)
```

## Remote tools

Tools with no local callable need an `invoker`, or the adapter refuses rather than
emitting a tool that fails when the model calls it:

```python
OpenAIAgentsAdapter(invoker=lambda tool, kwargs: my_client.call(tool, kwargs))
```

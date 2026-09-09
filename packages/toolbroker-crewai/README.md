# toolbroker-crewai

[CrewAI](https://www.crewai.com) adapter for
[ToolBroker](https://github.com/iamrameshwar/tool-broker).

```bash
pip install toolbroker-crewai
```

```python
from crewai import Agent, Crew, Task
from toolbroker import ToolBroker

broker = ToolBroker()
broker.add_functions([search_orders, issue_refund, check_inventory])
broker.index()

tools, selection = broker.select_for("crewai", "customer wants a refund", k=3)
agent = Agent(role="Support", goal="Help the customer", backstory="...", tools=tools)
```

CrewAI pulls a large dependency tree. That is precisely why this is a separate
package — nothing reaches the ToolBroker core, and installing it affects only you.

## Schemas

CrewAI wants `args_schema` as a **pydantic model class**, not a JSON Schema dict, so
this adapter builds one per tool with `toolbroker.schema.model_from_schema`. Awkward
parameter names are handled: `order-id` becomes `order_id` with an alias, and a
parameter called `schema` or `class` gets mangled and aliased rather than shadowing a
pydantic or Python name.

Anything the converter cannot map degrades to a permissive field. A tool the model can
still call with loose typing beats one that disappeared because its schema used a
construct nobody anticipated.

## Remote tools

Tools with no local callable need an `invoker`, or the adapter refuses rather than
emitting one that fails when the agent calls it:

```python
CrewAIAdapter(invoker=lambda tool, kwargs: my_client.call(tool, kwargs))
```

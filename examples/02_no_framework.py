"""Driving your own tool-calling loop against the OpenAI or Anthropic API.

ToolBroker hands you the tool list; you own the loop. Nothing here calls a model,
so it runs offline.

    python examples/02_no_framework.py
"""

import json

from toolbroker import ToolBroker
from toolbroker.adapters import AnthropicAdapter, OpenAIAdapter


def get_weather(city: str, units: str = "celsius") -> dict:
    """Get the current weather for a city.

    Args:
        city: City name.
        units: Either celsius or fahrenheit.
    """
    return {"city": city, "temp": 20, "units": units}


def book_flight(origin: str, destination: str, date: str) -> dict:
    """Book a flight between two airports.

    Args:
        origin: Departure airport code.
        destination: Arrival airport code.
        date: Departure date in ISO format.
    """
    return {"confirmation": "ABC123"}


def convert_currency(amount: float, source: str, target: str) -> float:
    """Convert an amount between two currencies.

    Args:
        amount: How much to convert.
        source: Source currency code.
        target: Target currency code.
    """
    return amount


broker = ToolBroker()
broker.add_functions([get_weather, book_flight, convert_currency])
broker.index()

query = "what is the weather like in Lisbon right now"

openai_tools = broker.openai_tools(query, k=1)
print("OpenAI shape:")
print(json.dumps(openai_tools, indent=2))

print("\nAnthropic shape:")
print(json.dumps(broker.anthropic_tools(query, k=1), indent=2))

# When the model replies with a tool name, map it back and call it yourself.
# In a real loop the arguments come from the model; here they are canned, keyed
# by tool so this example cannot break when ranking shifts.
CANNED_ARGS = {
    "python__get_weather": {"city": "Lisbon"},
    "python__book_flight": {"origin": "LHR", "destination": "LIS", "date": "2026-09-15"},
    "python__convert_currency": {"amount": 100.0, "source": "GBP", "target": "EUR"},
}

adapter = OpenAIAdapter()
adapter.render(broker.select(query, k=1).tools)
name = openai_tools[0]["function"]["name"]
handler = adapter.callable_for(name)
print(f"\nmodel picked {name!r} -> calling it:", handler(**CANNED_ARGS[name]))

AnthropicAdapter()  # same API, different wire shape

"""Rendering a selection as LangChain / LangGraph tools.

Needs `pip install toolbroker-langgraph`. Nothing here calls a model.

    python examples/04_langgraph.py
"""

from toolbroker_langgraph import LangGraphAdapter

from toolbroker import ToolBroker


def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    """Search available flights between two airports.

    Args:
        origin: Departure airport code.
        destination: Arrival airport code.
        date: Departure date in ISO format.
    """
    return [{"flight": "TP123", "price": 210}]


def book_hotel(city: str, nights: int) -> dict:
    """Book a hotel room in a city.

    Args:
        city: Where to stay.
        nights: How many nights.
    """
    return {"confirmation": "HOTEL-1"}


def convert_currency(amount: float, source: str, target: str) -> float:
    """Convert an amount between two currencies.

    Args:
        amount: How much to convert.
        source: Source currency code.
        target: Target currency code.
    """
    return amount * 1.08


broker = ToolBroker()
broker.add_functions([search_flights, book_hotel, convert_currency])
broker.index()

selection = broker.select("find me a flight to Lisbon next week", k=2)
tools = LangGraphAdapter().render(selection.tools)

print("selected:", list(selection.tool_ids))
for tool in tools:
    print(f"  {type(tool).__name__}: {tool.name} -> {tool.description}")

# Hand `tools` straight to a graph:
#   from langgraph.prebuilt import create_react_agent
#   agent = create_react_agent(model, tools)
print("\ninvoking the first tool directly (your framework would do this):")
print(" ", tools[0].invoke({"origin": "LHR", "destination": "LIS", "date": "2026-09-15"}))

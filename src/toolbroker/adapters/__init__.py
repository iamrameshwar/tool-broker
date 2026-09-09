"""Adapters: rendering tools into a framework's native shape."""

from .invocation import Invoker, ToolBinding, bind, bind_all, local_callable
from .raw import AnthropicAdapter, OpenAIAdapter
from .resolve import get_adapter

__all__ = [
    "AnthropicAdapter",
    "Invoker",
    "OpenAIAdapter",
    "ToolBinding",
    "bind",
    "bind_all",
    "get_adapter",
    "local_callable",
]

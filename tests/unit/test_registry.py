from __future__ import annotations

import pytest

from toolbroker.errors import PluginError
from toolbroker.index.embedders import HashingEmbedder, get_embedder
from toolbroker.registry import (
    GROUP_EMBEDDERS,
    GROUP_STORES,
    available,
    create,
    register,
    resolve,
    unregister,
)


def test_builtin_components_are_discoverable():
    # Built-ins register through the same entry points third parties use, so
    # this also proves the plugin mechanism works end to end.
    assert "hashing" in available(GROUP_EMBEDDERS)
    assert "memory" in available(GROUP_STORES)


def test_resolve_returns_the_class():
    assert resolve(GROUP_EMBEDDERS, "hashing") is HashingEmbedder


def test_create_instantiates_with_kwargs():
    assert create(GROUP_EMBEDDERS, "hashing", dim=64).dim == 64


def test_runtime_registration_shadows_entry_points():
    class Custom(HashingEmbedder):
        pass

    register(GROUP_EMBEDDERS, "custom-test", Custom)
    try:
        assert resolve(GROUP_EMBEDDERS, "custom-test") is Custom
        assert "custom-test" in available(GROUP_EMBEDDERS)
    finally:
        assert unregister(GROUP_EMBEDDERS, "custom-test") is True


def test_dotted_paths_resolve_without_registration():
    resolved = resolve(GROUP_EMBEDDERS, "toolbroker.index.embedders.hashing:HashingEmbedder")
    assert resolved is HashingEmbedder


def test_unknown_plugin_lists_what_is_installed():
    with pytest.raises(PluginError, match="Installed:"):
        resolve(GROUP_EMBEDDERS, "nonexistent")


def test_bad_dotted_path_reports_the_module():
    with pytest.raises(PluginError, match="cannot import module"):
        resolve(GROUP_EMBEDDERS, "no.such.module:Thing")


def test_bad_constructor_arguments_are_explained():
    with pytest.raises(PluginError, match="cannot construct"):
        create(GROUP_EMBEDDERS, "hashing", not_a_real_argument=1)


def test_get_embedder_helper():
    assert get_embedder("hashing", dim=32).dim == 32

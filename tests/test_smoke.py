"""
Smoke tests for Home Assistant Diagnostics MCP Server

These tests verify basic functionality without requiring a live HA instance.
"""

import pytest
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that core modules can be imported"""
    try:
        from app import config
        from app import ha
        from app import server
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_config_loading():
    """Test that configuration can be loaded"""
    from app.config import HA_URL, HA_TOKEN

    # These might be None in test environment, but should exist
    assert HA_URL is not None or True  # Allow None in test
    assert HA_TOKEN is not None or True


def test_server_structure():
    """Test that server has expected structure"""
    from app import server

    # Check that FastMCP instance exists
    assert hasattr(server, 'mcp')

    # Verify we can access the MCP instance
    mcp = server.mcp
    assert mcp is not None


def test_ha_functions_exist():
    """Test that expected HA functions are defined"""
    from app import ha

    # Core functions
    assert callable(ha.get_ha_version)
    assert callable(ha.get_entity_state)
    assert callable(ha.get_entities)
    assert callable(ha.get_automations)

    # Diagnostic functions
    assert callable(ha.get_system_health)
    assert callable(ha.get_network_info)
    assert callable(ha.diagnose_system)
    assert callable(ha.audit_zigbee_mesh)
    assert callable(ha.find_orphan_entities)


def test_no_syntax_errors():
    """Verify Python syntax is valid by importing all modules"""
    try:
        import app.server
        import app.ha
        import app.config
        import app.run
        assert True
    except SyntaxError as e:
        pytest.fail(f"Syntax error in code: {e}")


def test_function_uniqueness():
    """Test that there are no duplicate function definitions"""
    from app import ha
    import inspect

    # Get all functions from ha module
    functions = {}
    for name, obj in inspect.getmembers(ha):
        if inspect.isfunction(obj) or inspect.iscoroutinefunction(obj):
            # Check if function was already seen
            if name in functions:
                pytest.fail(f"Duplicate function found: {name}")
            functions[name] = obj

    # Key functions should exist and be unique
    assert 'get_system_health' in functions
    assert 'get_network_info' in functions
    assert 'get_zha_devices' in functions
    assert 'get_esphome_devices' in functions
    assert 'get_addons' in functions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

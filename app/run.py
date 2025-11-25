"""Entry point for running Home Assistant Diagnostics MCP server via uv/uvx tool"""

from app.server import mcp

def main():
    """Run the MCP server with stdio communication"""
    mcp.run()

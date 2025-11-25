import functools
import logging
import json
import httpx
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable, Awaitable, TypeVar, cast

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from app.ha import (
    get_ha_version, get_entity_state, call_service, get_entities,
    get_automations, restart_home_assistant, 
    cleanup_client, filter_fields, summarize_domain, get_system_overview,
    get_ha_error_log, get_entity_history
)
from app.ha import get_system_health as get_system_health_ha
from app.ha import get_integrations as get_integrations_ha
from app.ha import reload_scripts as reload_scripts_ha
from app.ha import reload_core_config as reload_core_config_ha
from app.ha import get_network_info as get_network_info_ha
from app.ha import get_zha_devices as get_zha_devices_ha
from app.ha import get_esphome_devices as get_esphome_devices_ha
from app.ha import get_addons as get_addons_ha
from app.ha import find_unavailable_entities as find_unavailable_entities_ha
from app.ha import find_stale_entities as find_stale_entities_ha
from app.ha import battery_report as battery_report_ha
from app.ha import get_repair_items as get_repair_items_ha
from app.ha import get_update_status as get_update_status_ha
from app.ha import get_entity_statistics as get_entity_statistics_ha
from app.ha import find_anomalous_entities as find_anomalous_entities_ha
from app.ha import recent_activity as recent_activity_ha
from app.ha import offline_devices_report as offline_devices_report_ha
from app.ha import identify_device as identify_device_ha
from app.ha import diagnose_issue as diagnose_issue_ha
from app.ha import diagnose_automation as diagnose_automation_ha
from app.ha import diagnose_system as diagnose_system_ha
from app.ha import auto_fix as auto_fix_ha
# NEW: High-impact hackathon tools
from app.ha import audit_zigbee_mesh as audit_zigbee_mesh_ha
from app.ha import find_orphan_entities as find_orphan_entities_ha
from app.ha import detect_automation_conflicts as detect_automation_conflicts_ha
from app.ha import energy_consumption_report as energy_consumption_report_ha
from app.ha import entity_dependency_graph as entity_dependency_graph_ha

# Type variable for generic functions
T = TypeVar('T')

# Create an MCP server
from mcp.server.fastmcp import FastMCP, Context, Image
from mcp.server.stdio import stdio_server
import mcp.types as types
mcp = FastMCP("Home Assistant Diagnostics")

def async_handler(command_type: str):
    """
    Simple decorator that logs the command
    
    Args:
        command_type: The type of command (for logging)
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            logger.info(f"Executing command: {command_type}")
            return await func(*args, **kwargs)
        return cast(Callable[..., Awaitable[T]], wrapper)
    return decorator

@mcp.tool()
@async_handler("get_version")
async def get_version() -> str:
    """
    Get the Home Assistant version
    
    Returns:
        A string with the Home Assistant version (e.g., "2025.3.0")
    """
    logger.info("Getting Home Assistant version")
    return await get_ha_version()

@mcp.tool()
@async_handler("get_entity")
async def get_entity(entity_id: str, fields: Optional[List[str]] = None, detailed: bool = False) -> dict:
    """
    Get the state of a Home Assistant entity with optional field filtering
    
    Args:
        entity_id: The entity ID to get (e.g. 'light.living_room')
        fields: Optional list of fields to include (e.g. ['state', 'attr.brightness'])
        detailed: If True, returns all entity fields without filtering
                
    Examples:
        entity_id="light.living_room" - basic state check
        entity_id="light.living_room", fields=["state", "attr.brightness"] - specific fields
        entity_id="light.living_room", detailed=True - all details
    """
    logger.info(f"Getting entity state: {entity_id}")
    if detailed:
        # Return all fields
        return await get_entity_state(entity_id, lean=False)
    elif fields:
        # Return only the specified fields
        return await get_entity_state(entity_id, fields=fields)
    else:
        # Return lean format with essential fields
        return await get_entity_state(entity_id, lean=True)

@mcp.tool()
@async_handler("entity_action")
async def entity_action(entity_id: str, action: str, params: Optional[Dict[str, Any]] = None) -> dict:
    """
    Perform an action on a Home Assistant entity (on, off, toggle)
    
    Args:
        entity_id: The entity ID to control (e.g. 'light.living_room')
        action: The action to perform ('on', 'off', 'toggle')
        params: Optional dictionary of additional parameters for the service call
    
    Returns:
        The response from Home Assistant
    
    Examples:
        entity_id="light.living_room", action="on", params={"brightness": 255}
        entity_id="switch.garden_lights", action="off"
        entity_id="climate.living_room", action="on", params={"temperature": 22.5}
    
    Domain-Specific Parameters:
        - Lights: brightness (0-255), color_temp, rgb_color, transition, effect
        - Covers: position (0-100), tilt_position
        - Climate: temperature, target_temp_high, target_temp_low, hvac_mode
        - Media players: source, volume_level (0-1)
    """
    if action not in ["on", "off", "toggle"]:
        return {"error": f"Invalid action: {action}. Valid actions are 'on', 'off', 'toggle'"}
    
    # Map action to service name
    service = action if action == "toggle" else f"turn_{action}"
    
    # Extract the domain from the entity_id
    domain = entity_id.split(".")[0]
    
    # Prepare service data
    data = {"entity_id": entity_id, **(params or {})}
    
    logger.info(f"Performing action '{action}' on entity: {entity_id} with params: {params}")
    return await call_service(domain, service, data)

@mcp.resource("ha://entities/{entity_id}")
@async_handler("get_entity_resource")
async def get_entity_resource(entity_id: str) -> str:
    """
    Get the state of a Home Assistant entity as a resource
    
    This endpoint provides a standard view with common entity information.
    For comprehensive attribute details, use the /detailed endpoint.
    
    Args:
        entity_id: The entity ID to get information for
    """
    logger.info(f"Getting entity resource: {entity_id}")
    
    # Get the entity state (using lean format for token efficiency)
    state = await get_entity_state(entity_id, lean=True)
    
    # Check if there was an error
    if "error" in state:
        return f"# Entity: {entity_id}\n\nError retrieving entity: {state['error']}"
    
    # Format the entity as markdown
    result = f"# Entity: {entity_id}\n\n"
    
    # Get friendly name if available
    friendly_name = state.get("attributes", {}).get("friendly_name")
    if friendly_name and friendly_name != entity_id:
        result += f"**Name**: {friendly_name}\n\n"
    
    # Add state
    result += f"**State**: {state.get('state')}\n\n"
    
    # Add domain info
    domain = entity_id.split(".")[0]
    result += f"**Domain**: {domain}\n\n"
    
    # Add key attributes based on domain type
    attributes = state.get("attributes", {})
    
    # Add a curated list of important attributes
    important_attrs = []
    
    # Common attributes across many domains
    common_attrs = ["device_class", "unit_of_measurement", "friendly_name"]
    
    # Domain-specific important attributes
    if domain == "light":
        important_attrs = ["brightness", "color_temp", "rgb_color", "supported_features", "supported_color_modes"] 
    elif domain == "sensor":
        important_attrs = ["unit_of_measurement", "device_class", "state_class"]
    elif domain == "climate":
        important_attrs = ["hvac_mode", "hvac_action", "temperature", "current_temperature", "target_temp_*"]
    elif domain == "media_player":
        important_attrs = ["media_title", "media_artist", "source", "volume_level", "media_content_type"]
    elif domain == "switch" or domain == "binary_sensor":
        important_attrs = ["device_class", "is_on"]
    
    # Combine with common attributes
    important_attrs.extend(common_attrs)
    
    # Deduplicate the list while preserving order
    important_attrs = list(dict.fromkeys(important_attrs))
    
    # Create and add the important attributes section
    result += "## Key Attributes\n\n"
    
    # Display only the important attributes that exist
    displayed_attrs = 0
    for attr_name in important_attrs:
        # Handle wildcard attributes (e.g., target_temp_*)
        if attr_name.endswith("*"):
            prefix = attr_name[:-1]
            matching_attrs = [name for name in attributes if name.startswith(prefix)]
            for name in matching_attrs:
                result += f"- **{name}**: {attributes[name]}\n"
                displayed_attrs += 1
        # Regular attribute match
        elif attr_name in attributes:
            attr_value = attributes[attr_name]
            if isinstance(attr_value, (list, dict)) and len(str(attr_value)) > 100:
                result += f"- **{attr_name}**: *[Complex data]*\n"
            else:
                result += f"- **{attr_name}**: {attr_value}\n"
            displayed_attrs += 1
    
    # If no important attributes were found, show a message
    if displayed_attrs == 0:
        result += "No key attributes found for this entity type.\n\n"
    
    # Add attribute count and link to detailed view
    total_attr_count = len(attributes)
    if total_attr_count > displayed_attrs:
        hidden_count = total_attr_count - displayed_attrs
        result += f"\n**Note**: Showing {displayed_attrs} of {total_attr_count} total attributes. "
        result += f"{hidden_count} additional attributes are available in the [detailed view](/api/resource/ha://entities/{entity_id}/detailed).\n\n"
    
    # Add last updated time if available
    if "last_updated" in state:
        result += f"**Last Updated**: {state['last_updated']}\n"
    
    return result

@mcp.tool()
@async_handler("list_entities")
async def list_entities(
    domain: Optional[str] = None, 
    search_query: Optional[str] = None, 
    limit: int = 100,
    fields: Optional[List[str]] = None,
    detailed: bool = False
) -> List[Dict[str, Any]]:
    """
    Get a list of Home Assistant entities with optional filtering
    
    Args:
        domain: Optional domain to filter by (e.g., 'light', 'switch', 'sensor')
        search_query: Optional search term to filter entities by name, id, or attributes
                     (Note: Does not support wildcards. To get all entities, leave this empty)
        limit: Maximum number of entities to return (default: 100)
        fields: Optional list of specific fields to include in each entity
        detailed: If True, returns all entity fields without filtering
    
    Returns:
        A list of entity dictionaries with lean formatting by default
    
    Examples:
        domain="light" - get all lights
        search_query="kitchen", limit=20 - search entities
        domain="sensor", detailed=True - full sensor details
    
    Best Practices:
        - Use lean format (default) for most operations
        - Prefer domain filtering over no filtering
        - For domain overviews, use domain_summary_tool instead of list_entities
        - Only request detailed=True when necessary for full attribute inspection
        - To get all entity types/domains, use list_entities without a domain filter, 
          then extract domains from entity_ids
    """
    log_message = "Getting entities"
    if domain:
        log_message += f" for domain: {domain}"
    if search_query:
        log_message += f" matching: '{search_query}'"
    if limit != 100:
        log_message += f" (limit: {limit})"
    if detailed:
        log_message += " (detailed format)"
    elif fields:
        log_message += f" (custom fields: {fields})"
    else:
        log_message += " (lean format)"
    
    logger.info(log_message)
    
    # Handle special case where search_query is a wildcard/asterisk - just ignore it
    if search_query == "*":
        search_query = None
        logger.info("Converting '*' search query to None (retrieving all entities)")
    
    # Use the updated get_entities function with field filtering
    return await get_entities(
        domain=domain, 
        search_query=search_query, 
        limit=limit,
        fields=fields,
        lean=not detailed  # Use lean format unless detailed is requested
    )

@mcp.resource("ha://entities")
@async_handler("get_all_entities_resource")
async def get_all_entities_resource() -> str:
    """
    Get a list of all Home Assistant entities as a resource
    
    This endpoint returns a complete list of all entities in Home Assistant, 
    organized by domain. For token efficiency with large installations,
    consider using domain-specific endpoints or the domain summary instead.
    
    Returns:
        A markdown formatted string listing all entities grouped by domain
        
    Examples:
        ```
        # Get all entities
        entities = mcp.get_resource("ha://entities")
        ```
        
    Best Practices:
        - WARNING: This endpoint can return large amounts of data with many entities
        - Prefer domain-filtered endpoints: ha://entities/domain/{domain}
        - For overview information, use domain summaries instead of full entity lists
        - Consider starting with a search if looking for specific entities
    """
    logger.info("Getting all entities as a resource")
    entities = await get_entities(lean=True)
    
    # Check if there was an error
    if isinstance(entities, dict) and "error" in entities:
        return f"Error retrieving entities: {entities['error']}"
    if len(entities) == 1 and isinstance(entities[0], dict) and "error" in entities[0]:
        return f"Error retrieving entities: {entities[0]['error']}"
    
    # Format the entities as a string
    result = "# Home Assistant Entities\n\n"
    result += f"Total entities: {len(entities)}\n\n"
    result += "⚠️ **Note**: For better performance and token efficiency, consider using:\n"
    result += "- Domain filtering: `ha://entities/domain/{domain}`\n"
    result += "- Domain summaries: `ha://entities/domain/{domain}/summary`\n"
    result += "- Entity search: `ha://search/{query}`\n\n"
    
    # Group entities by domain for better organization
    domains = {}
    for entity in entities:
        domain = entity["entity_id"].split(".")[0]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(entity)
    
    # Build the string with entities grouped by domain
    for domain in sorted(domains.keys()):
        domain_count = len(domains[domain])
        result += f"## {domain.capitalize()} ({domain_count})\n\n"
        for entity in sorted(domains[domain], key=lambda e: e["entity_id"]):
            # Get a friendly name if available
            friendly_name = entity.get("attributes", {}).get("friendly_name", "")
            result += f"- **{entity['entity_id']}**: {entity['state']}"
            if friendly_name and friendly_name != entity["entity_id"]:
                result += f" ({friendly_name})"
            result += "\n"
        result += "\n"
    
    return result

@mcp.tool()
@async_handler("search_entities_tool")
async def search_entities_tool(query: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for entities matching a query string
    
    Args:
        query: The search query to match against entity IDs, names, and attributes.
              (Note: Does not support wildcards. To get all entities, leave this blank or use list_entities tool)
        limit: Maximum number of results to return (default: 20)
    
    Returns:
        A dictionary containing search results and metadata:
        - count: Total number of matching entities found
        - results: List of matching entities with essential information
        - domains: Map of domains with counts (e.g. {"light": 3, "sensor": 2})
        
    Examples:
        query="temperature" - find temperature entities
        query="living room", limit=10 - find living room entities
        query="", limit=500 - list all entity types
        
    """
    logger.info(f"Searching for entities matching: '{query}' with limit: {limit}")
    
    # Special case - treat "*" as empty query to just return entities without filtering
    if query == "*":
        query = ""
        logger.info("Converting '*' to empty query (retrieving all entities up to limit)")
    
    # Handle empty query as a special case to just return entities up to the limit
    if not query or not query.strip():
        logger.info(f"Empty query - retrieving up to {limit} entities without filtering")
        entities = await get_entities(limit=limit, lean=True)
        
        # Check if there was an error
        if isinstance(entities, dict) and "error" in entities:
            return {"error": entities["error"], "count": 0, "results": [], "domains": {}}
        
        # No query, but we'll return a structured result anyway
        domains_count = {}
        simplified_entities = []
        
        for entity in entities:
            domain = entity["entity_id"].split(".")[0]
            
            # Count domains
            if domain not in domains_count:
                domains_count[domain] = 0
            domains_count[domain] += 1
            
            # Create simplified entity representation
            simplified_entity = {
                "entity_id": entity["entity_id"],
                "state": entity["state"],
                "domain": domain,
                "friendly_name": entity.get("attributes", {}).get("friendly_name", entity["entity_id"])
            }
            
            # Add key attributes based on domain
            attributes = entity.get("attributes", {})
            
            # Include domain-specific important attributes
            if domain == "light" and "brightness" in attributes:
                simplified_entity["brightness"] = attributes["brightness"]
            elif domain == "sensor" and "unit_of_measurement" in attributes:
                simplified_entity["unit"] = attributes["unit_of_measurement"]
            elif domain == "climate" and "temperature" in attributes:
                simplified_entity["temperature"] = attributes["temperature"]
            elif domain == "media_player" and "media_title" in attributes:
                simplified_entity["media_title"] = attributes["media_title"]
            
            simplified_entities.append(simplified_entity)
        
        # Return structured response for empty query
        return {
            "count": len(simplified_entities),
            "results": simplified_entities,
            "domains": domains_count,
            "query": "all entities (no filtering)"
        }
    
    # Normal search with non-empty query
    entities = await get_entities(search_query=query, limit=limit, lean=True)
    
    # Check if there was an error
    if isinstance(entities, dict) and "error" in entities:
        return {"error": entities["error"], "count": 0, "results": [], "domains": {}}
    
    # Prepare the results
    domains_count = {}
    simplified_entities = []
    
    for entity in entities:
        domain = entity["entity_id"].split(".")[0]
        
        # Count domains
        if domain not in domains_count:
            domains_count[domain] = 0
        domains_count[domain] += 1
        
        # Create simplified entity representation
        simplified_entity = {
            "entity_id": entity["entity_id"],
            "state": entity["state"],
            "domain": domain,
            "friendly_name": entity.get("attributes", {}).get("friendly_name", entity["entity_id"])
        }
        
        # Add key attributes based on domain
        attributes = entity.get("attributes", {})
        
        # Include domain-specific important attributes
        if domain == "light" and "brightness" in attributes:
            simplified_entity["brightness"] = attributes["brightness"]
        elif domain == "sensor" and "unit_of_measurement" in attributes:
            simplified_entity["unit"] = attributes["unit_of_measurement"]
        elif domain == "climate" and "temperature" in attributes:
            simplified_entity["temperature"] = attributes["temperature"]
        elif domain == "media_player" and "media_title" in attributes:
            simplified_entity["media_title"] = attributes["media_title"]
        
        simplified_entities.append(simplified_entity)
    
    # Return structured response
    return {
        "count": len(simplified_entities),
        "results": simplified_entities,
        "domains": domains_count,
        "query": query
    }
    
@mcp.resource("ha://search/{query}/{limit}")
@async_handler("search_entities_resource_with_limit")
async def search_entities_resource_with_limit(query: str, limit: str) -> str:
    """
    Search for entities matching a query string with a specified result limit
    
    This endpoint extends the basic search functionality by allowing you to specify
    a custom limit on the number of results returned. It's useful for both broader
    searches (larger limit) and more focused searches (smaller limit).
    
    Args:
        query: The search query to match against entity IDs, names, and attributes
        limit: Maximum number of entities to return (as a string, will be converted to int)
    
    Returns:
        A markdown formatted string with search results and a JSON summary
        
    Examples:
        ```
        # Search with a larger limit (up to 50 results)
        results = mcp.get_resource("ha://search/sensor/50")
        
        # Search with a smaller limit for focused results
        results = mcp.get_resource("ha://search/kitchen/5")
        ```
        
    Best Practices:
        - Use smaller limits (5-10) for focused searches where you need just a few matches
        - Use larger limits (30-50) for broader searches when you need more comprehensive results
        - Balance larger limits against token usage - more results means more tokens
        - Consider domain-specific searches for better precision: "light kitchen" instead of just "kitchen"
    """
    try:
        limit_int = int(limit)
        if limit_int <= 0:
            limit_int = 20
    except ValueError:
        limit_int = 20
        
    logger.info(f"Searching for entities matching: '{query}' with custom limit: {limit_int}")
    
    if not query or not query.strip():
        return "# Entity Search\n\nError: No search query provided"
    
    entities = await get_entities(search_query=query, limit=limit_int, lean=True)
    
    # Check if there was an error
    if isinstance(entities, dict) and "error" in entities:
        return f"# Entity Search\n\nError retrieving entities: {entities['error']}"
    
    # Format the search results
    result = f"# Entity Search Results for '{query}' (Limit: {limit_int})\n\n"
    
    if not entities:
        result += "No entities found matching your search query.\n"
        return result
    
    result += f"Found {len(entities)} matching entities:\n\n"
    
    # Group entities by domain for better organization
    domains = {}
    for entity in entities:
        domain = entity["entity_id"].split(".")[0]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(entity)
    
    # Build the string with entities grouped by domain
    for domain in sorted(domains.keys()):
        result += f"## {domain.capitalize()}\n\n"
        for entity in sorted(domains[domain], key=lambda e: e["entity_id"]):
            # Get a friendly name if available
            friendly_name = entity.get("attributes", {}).get("friendly_name", entity["entity_id"])
            result += f"- **{entity['entity_id']}**: {entity['state']}"
            if friendly_name != entity["entity_id"]:
                result += f" ({friendly_name})"
            result += "\n"
        result += "\n"
    
    # Add a more structured summary section for easy LLM processing
    result += "## Summary in JSON format\n\n"
    result += "```json\n"
    
    # Create a simplified JSON representation with only essential fields
    simplified_entities = []
    for entity in entities:
        simplified_entity = {
            "entity_id": entity["entity_id"],
            "state": entity["state"],
            "domain": entity["entity_id"].split(".")[0],
            "friendly_name": entity.get("attributes", {}).get("friendly_name", entity["entity_id"])
        }
        
        # Add key attributes based on domain type if they exist
        domain = entity["entity_id"].split(".")[0]
        attributes = entity.get("attributes", {})
        
        # Include domain-specific important attributes
        if domain == "light" and "brightness" in attributes:
            simplified_entity["brightness"] = attributes["brightness"]
        elif domain == "sensor" and "unit_of_measurement" in attributes:
            simplified_entity["unit"] = attributes["unit_of_measurement"]
        elif domain == "climate" and "temperature" in attributes:
            simplified_entity["temperature"] = attributes["temperature"]
        elif domain == "media_player" and "media_title" in attributes:
            simplified_entity["media_title"] = attributes["media_title"]
        
        simplified_entities.append(simplified_entity)
    
    result += json.dumps(simplified_entities, indent=2)
    result += "\n```\n"
    
    return result

# The domain_summary_tool is already implemented, no need to duplicate it

@mcp.tool()
@async_handler("domain_summary")
async def domain_summary_tool(domain: str, example_limit: int = 3) -> Dict[str, Any]:
    """
    Get a summary of entities in a specific domain
    
    Args:
        domain: The domain to summarize (e.g., 'light', 'switch', 'sensor')
        example_limit: Maximum number of examples to include for each state
    
    Returns:
        A dictionary containing:
        - total_count: Number of entities in the domain
        - state_distribution: Count of entities in each state
        - examples: Sample entities for each state
        - common_attributes: Most frequently occurring attributes
        
    Examples:
        domain="light" - get light summary
        domain="climate", example_limit=5 - climate summary with more examples
    Best Practices:
        - Use this before retrieving all entities in a domain to understand what's available    """
    logger.info(f"Getting domain summary for: {domain}")
    return await summarize_domain(domain, example_limit)

@mcp.tool()
@async_handler("system_overview")
async def system_overview() -> Dict[str, Any]:
    """
    Get a comprehensive overview of the entire Home Assistant system
    
    Returns:
        A dictionary containing:
        - total_entities: Total count of all entities
        - domains: Dictionary of domains with their entity counts and state distributions
        - domain_samples: Representative sample entities for each domain (2-3 per domain)
        - domain_attributes: Common attributes for each domain
        - area_distribution: Entities grouped by area (if available)
        
    Examples:
        Returns domain counts, sample entities, and common attributes
    Best Practices:
        - Use this as the first call when exploring an unfamiliar Home Assistant instance
        - Perfect for building context about the structure of the smart home
        - After getting an overview, use domain_summary_tool to dig deeper into specific domains
    """
    logger.info("Generating complete system overview")
    return await get_system_overview()

@mcp.resource("ha://entities/{entity_id}/detailed")
@async_handler("get_entity_resource_detailed")
async def get_entity_resource_detailed(entity_id: str) -> str:
    """
    Get detailed information about a Home Assistant entity as a resource
    
    Use this detailed view selectively when you need to:
    - Understand all available attributes of an entity
    - Debug entity behavior or capabilities
    - See comprehensive state information
    
    For routine operations where you only need basic state information,
    prefer the standard entity endpoint or specify fields in the get_entity tool.
    
    Args:
        entity_id: The entity ID to get information for
    """
    logger.info(f"Getting detailed entity resource: {entity_id}")
    
    # Get all fields, no filtering (detailed view explicitly requests all data)
    state = await get_entity_state(entity_id, lean=False)
    
    # Check if there was an error
    if "error" in state:
        return f"# Entity: {entity_id}\n\nError retrieving entity: {state['error']}"
    
    # Format the entity as markdown
    result = f"# Entity: {entity_id} (Detailed View)\n\n"
    
    # Get friendly name if available
    friendly_name = state.get("attributes", {}).get("friendly_name")
    if friendly_name and friendly_name != entity_id:
        result += f"**Name**: {friendly_name}\n\n"
    
    # Add state
    result += f"**State**: {state.get('state')}\n\n"
    
    # Add domain and entity type information
    domain = entity_id.split(".")[0]
    result += f"**Domain**: {domain}\n\n"
    
    # Add usage guidance
    result += "## Usage Note\n"
    result += "This is the detailed view showing all entity attributes. For token-efficient interactions, "
    result += "consider using the standard entity endpoint or the get_entity tool with field filtering.\n\n"
    
    # Add all attributes with full details
    attributes = state.get("attributes", {})
    if attributes:
        result += "## Attributes\n\n"
        
        # Sort attributes for better organization
        sorted_attrs = sorted(attributes.items())
        
        # Format each attribute with complete information
        for attr_name, attr_value in sorted_attrs:
            # Format the attribute value
            if isinstance(attr_value, (list, dict)):
                attr_str = json.dumps(attr_value, indent=2)
                result += f"- **{attr_name}**:\n```json\n{attr_str}\n```\n"
            else:
                result += f"- **{attr_name}**: {attr_value}\n"
    
    # Add context data section
    result += "\n## Context Data\n\n"
    
    # Add last updated time if available
    if "last_updated" in state:
        result += f"**Last Updated**: {state['last_updated']}\n"
    
    # Add last changed time if available
    if "last_changed" in state:
        result += f"**Last Changed**: {state['last_changed']}\n"
    
    # Add entity ID and context information
    if "context" in state:
        context = state["context"]
        result += f"**Context ID**: {context.get('id', 'N/A')}\n"
        if "parent_id" in context:
            result += f"**Parent Context**: {context['parent_id']}\n"
        if "user_id" in context:
            result += f"**User ID**: {context['user_id']}\n"
    
    # Add related entities suggestions
    related_domains = []
    if domain == "light":
        related_domains = ["switch", "scene", "automation"]
    elif domain == "sensor":
        related_domains = ["binary_sensor", "input_number", "utility_meter"]
    elif domain == "climate":
        related_domains = ["sensor", "switch", "fan"]
    elif domain == "media_player":
        related_domains = ["remote", "switch", "sensor"]
    
    if related_domains:
        result += "\n## Related Entity Types\n\n"
        result += "You may want to check entities in these related domains:\n"
        for related in related_domains:
            result += f"- {related}\n"
    
    return result

@mcp.resource("ha://entities/domain/{domain}")
@async_handler("list_states_by_domain_resource")
async def list_states_by_domain_resource(domain: str) -> str:
    """
    Get a list of entities for a specific domain as a resource
    
    This endpoint provides all entities of a specific type (domain). It's much more
    token-efficient than retrieving all entities when you only need entities of a 
    specific type.
    
    Args:
        domain: The domain to filter by (e.g., 'light', 'switch', 'sensor')
    
    Returns:
        A markdown formatted string with all entities in the specified domain
        
    Examples:
        ```
        # Get all lights
        lights = mcp.get_resource("ha://entities/domain/light")
        
        # Get all climate devices
        climate = mcp.get_resource("ha://entities/domain/climate")
        
        # Get all sensors
        sensors = mcp.get_resource("ha://entities/domain/sensor")
        ```
        
    Best Practices:
        - Use this endpoint when you need detailed information about all entities of a specific type
        - For a more concise overview, use the domain summary endpoint: ha://entities/domain/{domain}/summary
        - For sensors and other high-count domains, consider using a search to further filter results
    """
    logger.info(f"Getting entities for domain: {domain}")
    
    # Fixed pagination values for now
    page = 1
    page_size = 50
    
    # Get all entities for the specified domain (using lean format for token efficiency)
    entities = await get_entities(domain=domain, lean=True)
    
    # Check if there was an error
    if isinstance(entities, dict) and "error" in entities:
        return f"Error retrieving entities: {entities['error']}"
    
    # Format the entities as a string
    result = f"# {domain.capitalize()} Entities\n\n"
    
    # Pagination info (fixed for now due to MCP limitations)
    total_entities = len(entities)
    
    # List the entities
    for entity in sorted(entities, key=lambda e: e["entity_id"]):
        # Get a friendly name if available
        friendly_name = entity.get("attributes", {}).get("friendly_name", entity["entity_id"])
        result += f"- **{entity['entity_id']}**: {entity['state']}"
        if friendly_name != entity["entity_id"]:
            result += f" ({friendly_name})"
        result += "\n"
    
    # Add link to summary
    result += f"\n## Related Resources\n\n"
    result += f"- [View domain summary](/api/resource/ha://entities/domain/{domain}/summary)\n"
    
    return result

# ============================================================
# DIAGNOSTICS RESOURCES
# ============================================================

@mcp.resource("ha://diagnostics/health-score")
@async_handler("diagnostics_health_score")
async def diagnostics_health_score() -> str:
    """
    Get the system health score as a lightweight monitoring resource (Markdown format)
    
    This resource provides a quick snapshot of the overall Home Assistant
    system health without the full diagnostic details.
    
    Returns:
        Health score in markdown format
        
    Examples:
        ```
        # Get health score in markdown (human-readable)
        score = mcp.get_resource("ha://diagnostics/health-score")
        ```
        
    See also:
        - ha://diagnostics/health-score/json - JSON format
    """
    logger.info("Getting health score (markdown)")
    
    # Call the system diagnostics (without entities for speed)
    result = await diagnose_system_ha(include_entities=False)
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return f"# Error\n\nCould not retrieve health score: {error_msg}"
    
    # Extract key metrics
    health_score = result.get("global_health_score", 0)
    overall_severity = result.get("overall_severity", "unknown")
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Markdown format
    return f"""# Home Assistant Diagnostics — Health Score

**Score:** {health_score:.1f} / 100  
**Severity:** {overall_severity}  
**Timestamp:** {timestamp}  

Scores below 70 suggest the system requires attention.
"""


@mcp.resource("ha://diagnostics/health-score/json")
@async_handler("diagnostics_health_score_json")
async def diagnostics_health_score_json() -> str:
    """
    Get the system health score as a lightweight monitoring resource (JSON format)
    
    Returns:
        Health score in JSON format
        
    Examples:
        ```
        # Get health score in JSON (machine-readable)
        score = mcp.get_resource("ha://diagnostics/health-score/json")
        ```
    """
    logger.info("Getting health score (json)")
    
    # Call the system diagnostics (without entities for speed)
    result = await diagnose_system_ha(include_entities=False)
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return json.dumps({
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    # Extract key metrics
    health_score = result.get("global_health_score", 0)
    overall_severity = result.get("overall_severity", "unknown")
    timestamp = datetime.now(timezone.utc).isoformat()
    
    return json.dumps({
        "health_score": health_score,
        "overall_severity": overall_severity,
        "timestamp": timestamp
    }, indent=2)


@mcp.resource("ha://diagnostics/system")
@async_handler("diagnostics_system")
async def diagnostics_system() -> str:
    """
    Get a complete system diagnostics report (Markdown format)
    
    This resource provides comprehensive diagnostics of the entire Home Assistant
    system, including health score, issues by category, and recommendations.
    
    Returns:
        System diagnostics in markdown format
        
    Examples:
        ```
        # Get full system report in markdown
        report = mcp.get_resource("ha://diagnostics/system")
        ```
        
    See also:
        - ha://diagnostics/system/json - JSON format
    """
    logger.info("Getting system diagnostics (markdown)")
    
    # Call the system diagnostics
    result = await diagnose_system_ha(include_entities=False)
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return f"# Error\n\nCould not retrieve system diagnostics: {error_msg}"
    
    # Markdown format
    health_score = result.get("global_health_score", 0)
    overall_severity = result.get("overall_severity", "unknown")
    total_issues = result.get("total_issues", 0)
    timestamp = datetime.now(timezone.utc).isoformat()
    
    md = f"""# Home Assistant Diagnostics — System Report

**Health Score:** {health_score:.1f} / 100  
**Overall Severity:** {overall_severity}  
**Total Issues:** {total_issues}  
**Timestamp:** {timestamp}

## Issues by Category
"""
    
    # Add issues grouped by category
    issues_by_category = result.get("issues_by_category", {})
    
    if not issues_by_category:
        md += "\n*No issues detected*\n"
    else:
        for category, issues in sorted(issues_by_category.items()):
            if issues:
                md += f"\n### {category.replace('_', ' ').title()}\n"
                # Show first 5 items per category
                for issue in issues[:5]:
                    if isinstance(issue, dict):
                        desc = issue.get("description", issue.get("entity_id", str(issue)))
                    else:
                        desc = str(issue)
                    md += f"- {desc}\n"
                
                # Show count if there are more
                if len(issues) > 5:
                    md += f"- *(and {len(issues) - 5} more)*\n"
    
    # Add summary
    category_summaries = result.get("category_summaries", {})
    if category_summaries:
        md += f"\n## Category Summary\n\n"
        for category, count in sorted(category_summaries.items()):
            md += f"- **{category.replace('_', ' ').title()}**: {count} issue(s)\n"
    
    return md


@mcp.resource("ha://diagnostics/system/json")
@async_handler("diagnostics_system_json")
async def diagnostics_system_json() -> str:
    """
    Get a complete system diagnostics report (JSON format)
    
    Returns:
        System diagnostics in JSON format
        
    Examples:
        ```
        # Get full system report in JSON
        report = mcp.get_resource("ha://diagnostics/system/json")
        ```
    """
    logger.info("Getting system diagnostics (json)")
    
    # Call the system diagnostics
    result = await diagnose_system_ha(include_entities=False)
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return json.dumps({
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    return json.dumps(result, indent=2, default=str)


@mcp.resource("ha://diagnostics/entity/{entity_id}")
@async_handler("diagnostics_entity")
async def diagnostics_entity(entity_id: str) -> str:
    """
    Get diagnostics for a specific entity (Markdown format)
    
    This resource provides detailed diagnostics for a single entity,
    including root causes, recommended fixes, and auto-fix options.
    
    Args:
        entity_id: The entity ID to diagnose (e.g., 'light.kitchen')
    
    Returns:
        Entity diagnostics in markdown format
        
    Examples:
        ```
        # Get diagnostics for a light
        diag = mcp.get_resource("ha://diagnostics/entity/light.kitchen")
        ```
        
    See also:
        - ha://diagnostics/entity/{entity_id}/json - JSON format
    """
    logger.info(f"Getting entity diagnostics for {entity_id} (markdown)")
    
    # Call the entity diagnostics
    try:
        result = await diagnose_issue_ha(entity_id)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error diagnosing entity {entity_id}: {error_msg}")
        
        return f"""# Error

Could not diagnose entity `{entity_id}`:

{error_msg}

Please verify the entity ID is correct and the entity exists.
"""
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        
        return f"""# Diagnostics Error

Entity: `{entity_id}`

Could not complete diagnostics: {error_msg}
"""
    
    # Markdown format
    entity_summary = result.get("entity_summary", {})
    severity = result.get("severity", "unknown")
    timestamp = datetime.now(timezone.utc).isoformat()
    
    md = f"""# Diagnostics for entity: {entity_id}

**State:** {entity_summary.get('state', 'unknown')}  
**Severity:** {severity}  
**Last Updated:** {entity_summary.get('last_updated', 'unknown')}  
**Timestamp:** {timestamp}

"""
    
    # Root cause candidates
    root_causes = result.get("root_cause_candidates", [])
    if root_causes:
        md += "## Root Cause Candidates\n\n"
        for cause in root_causes:
            if isinstance(cause, dict):
                cause_text = cause.get("cause", str(cause))
                confidence = cause.get("confidence", "unknown")
                md += f"- {cause_text} — confidence: {confidence}\n"
            else:
                md += f"- {cause}\n"
        md += "\n"
    
    # Recommended fixes
    recommended_fixes = result.get("recommended_fixes", [])
    if recommended_fixes:
        md += "## Recommended Fixes\n\n"
        for fix in recommended_fixes:
            if isinstance(fix, dict):
                md += f"- {fix.get('description', str(fix))}\n"
            else:
                md += f"- {fix}\n"
        md += "\n"
    
    # Auto-fix options
    auto_fix_actions = result.get("auto_fix_actions_available", [])
    if auto_fix_actions:
        md += "## Auto-Fix Options\n\n"
        for action in auto_fix_actions:
            if isinstance(action, dict):
                action_name = action.get("action", "unknown")
                risk = action.get("risk", "unknown")
                desc = action.get("description", action_name)
                md += f"- {desc} ({risk} risk)\n"
            else:
                md += f"- {action}\n"
        md += "\n"
    
    # Diagnostics used
    diagnostics_used = result.get("diagnostics_used", [])
    if diagnostics_used:
        md += f"## Diagnostics Used\n\n"
        md += f"{', '.join(diagnostics_used)}\n"
    
    return md


@mcp.resource("ha://diagnostics/entity/{entity_id}/json")
@async_handler("diagnostics_entity_json")
async def diagnostics_entity_json(entity_id: str) -> str:
    """
    Get diagnostics for a specific entity (JSON format)
    
    Args:
        entity_id: The entity ID to diagnose (e.g., 'light.kitchen')
    
    Returns:
        Entity diagnostics in JSON format
        
    Examples:
        ```
        # Get diagnostics in JSON format
        diag = mcp.get_resource("ha://diagnostics/entity/sensor.temperature/json")
        ```
    """
    logger.info(f"Getting entity diagnostics for {entity_id} (json)")
    
    # Call the entity diagnostics
    try:
        result = await diagnose_issue_ha(entity_id)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error diagnosing entity {entity_id}: {error_msg}")
        
        return json.dumps({
            "error": f"Could not diagnose entity: {error_msg}",
            "entity_id": entity_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        
        return json.dumps({
            "error": error_msg,
            "entity_id": entity_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    return json.dumps(result, indent=2, default=str)


# ============================================================================
# SIGNATURE RESOURCES - Advanced Diagnostic Reports
# ============================================================================

@mcp.resource("ha://diagnostics/zigbee-mesh")
@async_handler("diagnostics_zigbee_mesh")
async def diagnostics_zigbee_mesh() -> str:
    """
    Advanced Zigbee mesh network health report (Markdown format)
    
    Provides comprehensive analysis of ZHA Zigbee mesh including:
    - Overall mesh health score (0-100)
    - LQI/RSSI distribution
    - Weak links and orphan devices
    - Power source distribution
    - Actionable recommendations
    
    Returns:
        Detailed Zigbee mesh report in Markdown format with ASCII visualizations
        
    Examples:
        ```
        # Get Zigbee mesh health report
        report = mcp.get_resource("ha://diagnostics/zigbee-mesh")
        ```
    """
    logger.info("Generating Zigbee mesh diagnostic report")
    
    try:
        result = await audit_zigbee_mesh_ha(limit=100)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error auditing Zigbee mesh: {error_msg}")
        return f"# ⚠️ Zigbee Mesh Diagnostic Error\n\n{error_msg}\n"
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return f"# ⚠️ Zigbee Mesh Diagnostic Error\n\n{error_msg}\n"
    
    # Build markdown report
    md = "# 📡 Zigbee Mesh Network Health Report\n\n"
    
    timestamp = result.get("timestamp", "N/A")
    md += f"*Generated: {timestamp}*\n\n"
    
    # Health Score (big visual)
    score = result.get("mesh_health_score", 0)
    if score >= 90:
        emoji = "🟢"
        status = "EXCELLENT"
    elif score >= 70:
        emoji = "🟡"
        status = "GOOD"
    elif score >= 50:
        emoji = "🟠"
        status = "FAIR"
    else:
        emoji = "🔴"
        status = "POOR"
    
    md += f"## {emoji} Overall Health Score: {score}/100 - {status}\n\n"
    
    # Statistics
    stats = result.get("statistics", {})
    total_devices = result.get("total_devices", 0)
    
    md += f"### 📊 Network Statistics\n\n"
    md += f"| Metric | Value |\n"
    md += f"|--------|-------|\n"
    md += f"| Total Devices | {total_devices} |\n"
    md += f"| Average LQI | {stats.get('average_lqi', 0):.1f} |\n"
    md += f"| Average RSSI | {stats.get('average_rssi', 0):.1f} dBm |\n"
    md += f"| Weak Links | {stats.get('weak_links_count', 0)} |\n"
    md += f"| Orphan Devices | {stats.get('orphan_devices_count', 0)} |\n"
    md += f"| Router Count | {result.get('router_count', 0)} |\n\n"
    
    # LQI Distribution (ASCII bar chart)
    lqi_dist = result.get("lqi_distribution", {})
    md += f"### 📈 Link Quality Distribution\n\n"
    md += "```\n"
    
    categories = [
        ("Excellent (200-255)", "excellent (200-255)", "█"),
        ("Good (150-199)    ", "good (150-199)", "▓"),
        ("Fair (120-149)    ", "fair (120-149)", "▒"),
        ("Poor (80-119)     ", "poor (80-119)", "░"),
        ("Critical (<80)    ", "critical (<80)", "▁")
    ]
    
    max_count = max(lqi_dist.values()) if lqi_dist else 1
    for label, key, char in categories:
        count = lqi_dist.get(key, 0)
        bar_length = int((count / max_count) * 40) if max_count > 0 else 0
        bar = char * bar_length
        md += f"{label}: {bar} {count}\n"
    
    md += "```\n\n"
    
    # Power Source Distribution
    power_dist = result.get("power_source_distribution", {})
    if power_dist:
        md += f"### 🔌 Power Source Distribution\n\n"
        for source, count in power_dist.items():
            md += f"- **{source}**: {count} devices\n"
        md += "\n"
    
    # Coordinator Info
    coord = result.get("coordinator_info", {})
    if coord:
        md += f"### 🎯 Coordinator Information\n\n"
        md += f"- **Name**: {coord.get('name', 'N/A')}\n"
        md += f"- **Model**: {coord.get('model', 'N/A')}\n"
        md += f"- **IEEE**: `{coord.get('ieee', 'N/A')}`\n\n"
    
    # Weak Links
    weak_links = result.get("weak_links", [])
    if weak_links:
        md += f"### ⚠️ Weak Links Detected ({len(weak_links)})\n\n"
        md += "| Device | LQI | RSSI | Severity |\n"
        md += "|--------|-----|------|----------|\n"
        for link in weak_links[:10]:  # Top 10
            md += f"| {link.get('name', 'Unknown')} | {link.get('lqi', 0)} | {link.get('rssi', 0)} dBm | {link.get('severity', 'unknown')} |\n"
        if len(weak_links) > 10:
            md += f"\n*...and {len(weak_links) - 10} more*\n"
        md += "\n"
    
    # Orphan Devices
    orphans = result.get("orphan_devices", [])
    if orphans:
        md += f"### 🚨 Orphan Devices ({len(orphans)})\n\n"
        for orphan in orphans[:5]:  # Top 5
            md += f"- **{orphan.get('name', 'Unknown')}** - {orphan.get('manufacturer', 'N/A')}\n"
        if len(orphans) > 5:
            md += f"\n*...and {len(orphans) - 5} more*\n"
        md += "\n"
    
    # Recommendations
    recommendations = result.get("recommendations", [])
    if recommendations:
        md += f"### 💡 Recommendations\n\n"
        for rec in recommendations:
            priority = rec.get("priority", "low").upper()
            issue = rec.get("issue", "N/A")
            action = rec.get("action", "N/A")
            
            if priority == "CRITICAL":
                icon = "🔴"
            elif priority == "HIGH":
                icon = "🟠"
            elif priority == "MEDIUM":
                icon = "🟡"
            else:
                icon = "🟢"
            
            md += f"{icon} **{priority}**: {issue}\n"
            md += f"   → *Action*: {action}\n\n"
    else:
        md += f"### ✅ No Issues Detected\n\nYour Zigbee mesh is healthy!\n\n"
    
    return md


@mcp.resource("ha://diagnostics/system-health")
@async_handler("diagnostics_system_health")
async def diagnostics_system_health() -> str:
    """
    Comprehensive system health diagnostic report (Markdown format)
    
    Orchestrates multiple diagnostic tools to provide:
    - Global health score
    - Issues by category (system, network, devices, entities, batteries, etc.)
    - Severity breakdown
    - Actionable recommendations
    
    Returns:
        Complete system health report in Markdown format
        
    Examples:
        ```
        # Get complete system health report
        report = mcp.get_resource("ha://diagnostics/system-health")
        ```
    """
    logger.info("Generating comprehensive system health report")
    
    try:
        result = await diagnose_system_ha(include_entities=True)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error diagnosing system: {error_msg}")
        return f"# ⚠️ System Health Diagnostic Error\n\n{error_msg}\n"
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return f"# ⚠️ System Health Diagnostic Error\n\n{error_msg}\n"
    
    # Build markdown report
    md = "# 🏥 Home Assistant System Health Report\n\n"
    
    timestamp = result.get("timestamp", "N/A")
    md += f"*Generated: {timestamp}*\n\n"
    
    # Health Score (big visual)
    score = result.get("global_health_score", 0)
    if score >= 90:
        emoji = "🟢"
        status = "EXCELLENT"
        desc = "Your system is running optimally!"
    elif score >= 70:
        emoji = "🟡"
        status = "GOOD"
        desc = "Minor issues detected, but system is stable."
    elif score >= 50:
        emoji = "🟠"
        status = "FAIR"
        desc = "Several issues require attention."
    else:
        emoji = "🔴"
        status = "CRITICAL"
        desc = "Urgent issues detected! Immediate action required."
    
    md += f"## {emoji} Overall Health Score: {score}/100 - {status}\n\n"
    md += f"*{desc}*\n\n"
    
    # Severity Breakdown
    severity_breakdown = result.get("severity_breakdown", {})
    total_issues = result.get("total_issues", 0)
    
    md += f"### 📊 Issue Summary\n\n"
    md += f"**Total Issues**: {total_issues}\n\n"
    
    if total_issues > 0:
        md += "| Severity | Count |\n"
        md += "|----------|-------|\n"
        for severity in ["critical", "high", "medium", "low"]:
            count = severity_breakdown.get(severity, 0)
            if count > 0:
                if severity == "critical":
                    icon = "🔴"
                elif severity == "high":
                    icon = "🟠"
                elif severity == "medium":
                    icon = "🟡"
                else:
                    icon = "🟢"
                md += f"| {icon} {severity.capitalize()} | {count} |\n"
        md += "\n"
    
    # Issues by Category
    issues_by_category = result.get("issues_by_category", {})
    category_summaries = result.get("category_summaries", {})
    
    if category_summaries:
        md += f"### 📋 Issues by Category\n\n"
        
        category_icons = {
            "system": "🖥️",
            "network": "🌐",
            "integrations": "🔌",
            "devices": "📱",
            "entities": "🏷️",
            "batteries": "🔋",
            "zigbee_mesh": "📡",
            "esphome": "🔧",
            "logs_errors": "📝",
            "updates": "⬆️"
        }
        
        for category, count in category_summaries.items():
            if count > 0:
                icon = category_icons.get(category, "📌")
                md += f"- {icon} **{category.replace('_', ' ').title()}**: {count} issues\n"
        md += "\n"
    
    # Detailed Issues by Category
    if issues_by_category:
        md += f"### 🔍 Detailed Findings\n\n"
        
        for category, issues in issues_by_category.items():
            if issues:
                icon = category_icons.get(category, "📌")
                md += f"#### {icon} {category.replace('_', ' ').title()}\n\n"
                
                for issue in issues[:5]:  # Top 5 per category
                    severity = issue.get("severity", "low")
                    description = issue.get("description", "N/A")
                    
                    if severity == "critical":
                        sev_icon = "🔴"
                    elif severity == "high":
                        sev_icon = "🟠"
                    elif severity == "medium":
                        sev_icon = "🟡"
                    else:
                        sev_icon = "🟢"
                    
                    md += f"{sev_icon} **{severity.upper()}**: {description}\n\n"
                
                if len(issues) > 5:
                    md += f"*...and {len(issues) - 5} more issues in this category*\n\n"
    
    # Diagnostics Used
    diagnostics_used = result.get("diagnostics_used", [])
    diagnostics_count = result.get("diagnostics_count", 0)
    
    if diagnostics_used:
        md += f"### 🔬 Diagnostics Performed\n\n"
        md += f"**Total Diagnostics**: {diagnostics_count}\n\n"
        md += ", ".join(diagnostics_used) + "\n\n"
    
    # Recommendations
    md += f"### 💡 Recommended Actions\n\n"
    
    if score >= 90:
        md += "✅ Your system is healthy! Continue with regular maintenance:\n"
        md += "- Monitor battery levels weekly\n"
        md += "- Check for updates monthly\n"
        md += "- Review automation performance periodically\n\n"
    elif score >= 70:
        md += "⚠️ Address medium/high severity issues:\n"
        md += "- Fix unavailable entities\n"
        md += "- Update outdated integrations\n"
        md += "- Replace low batteries\n\n"
    elif score >= 50:
        md += "🚨 Urgent attention required:\n"
        md += "- Fix critical/high severity issues immediately\n"
        md += "- Check error logs for recurring problems\n"
        md += "- Verify network connectivity\n"
        md += "- Consider system restart if needed\n\n"
    else:
        md += "🔴 CRITICAL ACTION REQUIRED:\n"
        md += "- Address all critical issues immediately\n"
        md += "- Check system logs for errors\n"
        md += "- Verify Home Assistant is running correctly\n"
        md += "- Consider backup and restore if corruption detected\n\n"
    
    return md


# Automation management MCP tools
@mcp.tool()
@async_handler("list_automations")
async def list_automations() -> List[Dict[str, Any]]:
    """
    Get a list of all automations from Home Assistant
    
    This function retrieves all automations configured in Home Assistant,
    including their IDs, entity IDs, state, and display names.
    
    Returns:
        A list of automation dictionaries, each containing id, entity_id, 
        state, and alias (friendly name) fields.
        
    Examples:
        Returns all automation objects with state and friendly names
    
    """
    logger.info("Getting all automations")
    try:
        # Get automations will now return data from states API, which is more reliable
        automations = await get_automations()
        
        # Handle error responses that might still occur
        if isinstance(automations, dict) and "error" in automations:
            logger.warning(f"Error getting automations: {automations['error']}")
            return []
            
        # Handle case where response is a list with error
        if isinstance(automations, list) and len(automations) == 1 and isinstance(automations[0], dict) and "error" in automations[0]:
            logger.warning(f"Error getting automations: {automations[0]['error']}")
            return []
            
        return automations
    except Exception as e:
        logger.error(f"Error in list_automations: {str(e)}")
        return []

# We already have a list_automations tool, so no need to duplicate functionality

@mcp.tool()
@async_handler("restart_ha")
async def restart_ha(ctx: Context = None) -> Dict[str, Any]:
    """
    Restart Home Assistant
    
    ⚠️ WARNING: Temporarily disrupts all Home Assistant operations
    
    Args:
        ctx: MCP Context for user confirmation
    
    Returns:
        Result of restart operation
    """
    logger.info("Restarting Home Assistant")
    return await restart_home_assistant(ctx=ctx)

@mcp.tool()
@async_handler("call_service")
async def call_service_tool(domain: str, service: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Call any Home Assistant service (low-level API access)
    
    Args:
        domain: The domain of the service (e.g., 'light', 'switch', 'automation')
        service: The service to call (e.g., 'turn_on', 'turn_off', 'toggle')
        data: Optional data to pass to the service (e.g., {'entity_id': 'light.living_room'})
    
    Returns:
        The response from Home Assistant (usually empty for successful calls)
    
    Examples:
        domain='light', service='turn_on', data={'entity_id': 'light.x', 'brightness': 255}
        domain='automation', service='reload'
        domain='fan', service='set_percentage', data={'entity_id': 'fan.x', 'percentage': 50}
    
    """
    logger.info(f"Calling Home Assistant service: {domain}.{service} with data: {data}")
    return await call_service(domain, service, data or {})

# Prompt functionality
@mcp.prompt()
def debug_automation(automation_id: str):
    """
    Help a user troubleshoot an automation that isn't working
    
    This prompt guides the user through the process of diagnosing and fixing
    issues with an existing Home Assistant automation.
    
    Args:
        automation_id: The entity ID of the automation to troubleshoot
    
    Returns:
        A list of messages for the interactive conversation
    """
    system_message = """You are a Home Assistant automation troubleshooting expert.
You'll help the user diagnose problems with their automation by checking:
1. Trigger conditions and whether they're being met
2. Conditions that might be preventing execution
3. Action configuration issues
4. Entity availability and connectivity
5. Permissions and scope issues"""
    
    user_message = f"My automation {automation_id} isn't working properly. Can you help me troubleshoot it?"
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

@mcp.prompt()
def troubleshoot_entity(entity_id: str):
    """
    Guide a user through troubleshooting issues with an entity
    
    This prompt helps diagnose and resolve problems with a specific
    Home Assistant entity that isn't functioning correctly.
    
    Args:
        entity_id: The entity ID having issues
    
    Returns:
        A list of messages for the interactive conversation
    """
    system_message = """You are a Home Assistant entity troubleshooting expert.
You'll help the user diagnose problems with their entity by checking:
1. Entity status and availability
2. Integration status
3. Device connectivity
4. Recent state changes and error patterns
5. Configuration issues
6. Common problems with this entity type"""
    
    user_message = f"My entity {entity_id} isn't working properly. Can you help me troubleshoot it?"
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

@mcp.prompt()
def automation_health_check():
    """
    Review all automations, find conflicts, redundancies, or improvement opportunities
    
    This prompt helps users perform a comprehensive review of their Home Assistant
    automations to identify issues, optimize performance, and improve reliability.
    
    Returns:
        A list of messages for the interactive conversation
    """
    system_message = """You are a Home Assistant automation expert specializing in system optimization.
You'll help the user perform a comprehensive audit of their automations by:
1. Reviewing all automations for potential conflicts (e.g., opposing actions)
2. Identifying redundant automations that could be consolidated
3. Finding inefficient trigger patterns that might cause unnecessary processing
4. Detecting missing conditions that could improve reliability
5. Suggesting template optimizations for more efficient processing
6. Uncovering potential race conditions between automations
7. Recommending structural improvements to the automation organization
8. Highlighting best practices and suggesting implementation changes"""
    
    user_message = "I'd like to do a health check on all my Home Assistant automations. Can you help me review them for conflicts, redundancies, and potential improvements?"
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


# ============================================================================
# SIGNATURE PROMPT - Ultimate Diagnostic Orchestrator
# ============================================================================

@mcp.prompt()
def diagnose_everything():
    """
    Ultimate diagnostic orchestrator - Complete system health check
    
    This prompt orchestrates ALL advanced diagnostic tools to provide:
    - Comprehensive system health analysis
    - Zigbee mesh network audit
    - Entity usage and orphan detection
    - Automation conflict analysis
    - Energy consumption tracking
    - Entity dependency mapping
    - Severity-based prioritization
    - Actionable recommendations with auto-fix suggestions
    
    Perfect for:
    - Monthly system health audits
    - Pre/post update verification
    - Troubleshooting complex issues
    - System optimization planning
    - Comprehensive demos
    
    Returns:
        A list of messages for the ultimate diagnostic conversation
    """
    system_message = """You are an expert Home Assistant diagnostic AI with access to the most advanced diagnostic tools available.

Your mission: Perform a COMPLETE system health audit using ALL available diagnostic tools.

🔬 **DIAGNOSTIC PROTOCOL**:

1. **System-Wide Analysis**
   - Run diagnose_system(include_entities=True) for global health
   - Calculate overall health score (0-100)
   - Identify severity breakdown (critical/high/medium/low)
   - Group issues by category

2. **Zigbee Mesh Network Audit** (if ZHA detected)
   - Run audit_zigbee_mesh() for mesh analysis
   - Check mesh_health_score
   - Identify weak links (LQI < 120)
   - Detect orphan devices
   - Analyze router distribution

3. **Entity Usage Analysis**
   - Run find_orphan_entities() to detect unused entities
   - Calculate orphan percentage
   - Identify cleanup opportunities
   - Find most/least used entities

4. **Automation Safety Check**
   - Run detect_automation_conflicts()
   - Detect race conditions
   - Identify potential loops
   - Find conflicting actions
   - Check unsafe modes

5. **Energy Consumption** (if energy sensors exist)
   - Run energy_consumption_report(period_hours=24)
   - Calculate total consumption
   - Identify top consumers
   - Project monthly costs
   - Suggest energy-saving opportunities

6. **Dependency Mapping**
   - Run entity_dependency_graph()
   - Identify critical entities (highly referenced)
   - Detect circular dependencies
   - Find isolated entities

📊 **REPORTING FORMAT**:

Present findings as a structured medical-style report:

```
🏥 HOME ASSISTANT COMPLETE DIAGNOSTIC REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 EXECUTIVE SUMMARY
   Overall Health Score: [X]/100 - [STATUS]
   Total Issues: [N]
   Critical: [N] | High: [N] | Medium: [N] | Low: [N]
   
🔬 DIAGNOSTIC CATEGORIES

   1️⃣ SYSTEM HEALTH
      [Summary with key metrics]
      
   2️⃣ ZIGBEE MESH NETWORK
      Mesh Health: [X]/100
      [Key findings]
      
   3️⃣ ENTITY USAGE
      Total Entities: [N]
      Orphans: [N] ([X]%)
      [Cleanup opportunities]
      
   4️⃣ AUTOMATION SAFETY
      Total Automations: [N]
      Conflicts Detected: [N]
      [Safety issues]
      
   5️⃣ ENERGY CONSUMPTION
      24h Consumption: [X] kWh
      Estimated Cost: $[X]
      [Top consumers]
      
   6️⃣ DEPENDENCY ANALYSIS
      Critical Entities: [N]
      Circular Dependencies: [N]
      [Key relationships]

🚨 CRITICAL FINDINGS
   [List critical/high severity issues with priority]

💡 RECOMMENDATIONS
   [Prioritized action items with auto-fix availability]

✅ AUTO-FIX OPPORTUNITIES
   [Issues that can be fixed with auto_fix() tool]
```

🎯 **CONVERSATION FLOW**:
1. Greet user and explain you'll perform complete diagnostic
2. Run ALL diagnostic tools (show progress)
3. Present structured report
4. Ask if user wants to:
   - Deep-dive into any category
   - Apply auto-fix for fixable issues
   - Get specific recommendations
   - Review any particular finding

⚡ **BEST PRACTICES**:
- Use markdown formatting for clarity
- Include emoji for visual organization
- Prioritize critical issues first
- Provide specific, actionable recommendations
- Mention auto_fix() availability
- Use ASCII charts/tables where helpful
- Keep token usage efficient (summarize large datasets)
- Focus on HIGH-IMPACT findings
"""

    user_message = """I want a COMPLETE diagnostic of my Home Assistant system.

Please perform a comprehensive health audit using all available diagnostic tools:
- System health analysis
- Zigbee mesh audit
- Entity usage analysis
- Automation safety check
- Energy consumption tracking
- Dependency mapping

Provide a structured report with:
✅ Overall health score
✅ Issues by severity and category
✅ Critical findings that need immediate attention
✅ Actionable recommendations
✅ Auto-fix opportunities

Let's do a full "physical exam" of my smart home! 🏥"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


# Documentation endpoint
@mcp.tool()
@async_handler("get_history")
async def get_history(entity_id: str, hours: int = 24) -> Dict[str, Any]:
    """
    Get the history of an entity's state changes
    
    Args:
        entity_id: The entity ID to get history for
        hours: Number of hours of history to retrieve (default: 24)
    
    Returns:
        A dictionary containing:
        - entity_id: The entity ID requested
        - states: List of state objects with timestamps
        - count: Number of state changes found
        - first_changed: Timestamp of earliest state change
        - last_changed: Timestamp of most recent state change
        
    Examples:
        entity_id="light.living_room" - get 24h history
        entity_id="sensor.temperature", hours=168 - get 7 day history
    Best Practices:
        - Keep hours reasonable (24-72) for token efficiency
        - Use for entities with discrete state changes rather than continuously changing sensors
        - Consider the state distribution rather than every individual state    
    """
    logger.info(f"Getting history for entity: {entity_id}, hours: {hours}")
    
    try:
        # Get entity history from Home Assistant API
        history_data = await get_entity_history(entity_id, hours)
        
        # Check for errors from the API call
        if isinstance(history_data, dict) and "error" in history_data:
            return {
                "entity_id": entity_id,
                "error": history_data["error"],
                "states": [],
                "count": 0
            }
        
        # The result from the API is a list of lists of state changes
        # We need to flatten it and process it
        states = []
        if history_data and isinstance(history_data, list):
            for state_list in history_data:
                states.extend(state_list)
        
        if not states:
            return {
                "entity_id": entity_id,
                "states": [],
                "count": 0,
                "first_changed": None,
                "last_changed": None,
                "note": "No state changes found in the specified timeframe."
            }
        
        # Sort states by last_changed timestamp
        states.sort(key=lambda x: x.get("last_changed", ""))
        
        # Extract first and last changed timestamps
        first_changed = states[0].get("last_changed")
        last_changed = states[-1].get("last_changed")
        
        return {
            "entity_id": entity_id,
            "states": states,
            "count": len(states),
            "first_changed": first_changed,
            "last_changed": last_changed
        }
    except Exception as e:
        logger.error(f"Error processing history for {entity_id}: {str(e)}")
        return {
            "entity_id": entity_id,
            "error": f"Error processing history: {str(e)}",
            "states": [],
            "count": 0
        }

@mcp.tool()
@async_handler("get_error_log")
async def get_error_log() -> Dict[str, Any]:
    """
    Get the Home Assistant error log for troubleshooting
    
    Returns:
        A dictionary containing:
        - log_text: The full error log text
        - error_count: Number of ERROR entries found
        - warning_count: Number of WARNING entries found
        - integration_mentions: Map of integration names to mention counts
        - error: Error message if retrieval failed
        
    Examples:
        Returns errors, warnings count and integration mentions
    Best Practices:
        - Use this tool when troubleshooting specific Home Assistant errors
        - Look for patterns in repeated errors
        - Pay attention to timestamps to correlate errors with events
        - Focus on integrations with many mentions in the log    
    """
    logger.info("Getting Home Assistant error log")
    return await get_ha_error_log()


@mcp.tool()
@async_handler("get_system_health")
async def get_system_health() -> Dict[str, Any]:
    """
    Get system health information including CPU, RAM, disk usage, and supervisor status
    
    Returns:
        A dictionary containing:
        - version: Home Assistant version
        - location_name: Location name
        - time_zone: Time zone
        - unit_system: Unit system (metric/imperial)
        - supervisor: Supervisor information (if running HA OS/Supervised)
        - host: Host information including disk usage
        - components: Number of loaded components
        - config_dir: Configuration directory path
        
    Examples:
        Returns complete system health information
    Best Practices:
        - Check this first when diagnosing performance issues
        - Monitor disk usage to prevent database issues
        - Verify supervisor health on HA OS installations
        - Use to confirm version before suggesting updates
    """
    logger.info("Getting system health information")
    return await get_system_health_ha()


@mcp.tool()
@async_handler("list_integrations")
async def list_integrations() -> Dict[str, Any]:
    """
    Get list of all loaded integrations and their entity counts
    
    Returns:
        A dictionary containing:
        - total_integrations: Total number of integrations
        - integrations: List of all integration names
        - entity_counts: Dictionary mapping domains to entity counts
        - integrations_with_entities: Number of integrations with entities
        
    Examples:
        Returns all loaded integrations with entity statistics
    Best Practices:
        - Use to identify which integrations are loaded
        - Check entity counts to find integrations with issues
        - Compare with expected integrations to find missing ones
        - Useful for troubleshooting integration-specific problems
    """
    logger.info("Getting list of integrations")
    return await get_integrations_ha()


@mcp.tool()
@async_handler("reload_scripts")
async def reload_scripts() -> Dict[str, Any]:
    """
    Reload all scripts from YAML configuration
    
    Returns:
        Success status of the reload operation
        
    Examples:
        Reloads scripts without restarting Home Assistant
    Best Practices:
        - Use after editing script YAML files
        - Faster than full restart
        - Scripts will be unavailable briefly during reload
        - Check error log if scripts don't reload correctly
    """
    logger.info("Reloading scripts")
    return await reload_scripts_ha()


@mcp.tool()
@async_handler("reload_core_config")
async def reload_core_config() -> Dict[str, Any]:
    """
    Reload Home Assistant core configuration from YAML
    
    Returns:
        Success status of the reload operation
        
    Examples:
        Reloads core configuration without full restart
    Best Practices:
        - Use after editing configuration.yaml
        - Reloads customize, core config, and more
        - Much faster than restarting
        - Some changes still require full restart
        - Check error log if reload fails
    """
    logger.info("Reloading core configuration")
    return await reload_core_config_ha()


@mcp.tool()
@async_handler("get_network_info")
async def get_network_info() -> Dict[str, Any]:
    """
    Get network configuration and connectivity information
    
    Returns:
        A dictionary containing:
        - hostname: System hostname (if available)
        - interfaces: Network interfaces (if running HA OS)
        - external_url: External URL configuration
        - internal_url: Internal URL configuration
        - location_name: Location name
        - latitude: Latitude coordinate
        - longitude: Longitude coordinate
        
    Examples:
        Returns network configuration and connectivity details
    Best Practices:
        - Use to diagnose connectivity issues
        - Check external/internal URL configuration
        - Verify network interfaces on HA OS
        - Useful for remote access troubleshooting
    """
    logger.info("Getting network information")
    return await get_network_info_ha()


@mcp.tool()
@async_handler("get_zha_devices")
async def get_zha_devices() -> Dict[str, Any]:
    """
    Get list of ZHA (Zigbee) devices with their status and link quality
    
    Returns:
        A dictionary containing:
        - total_devices: Total number of Zigbee devices
        - online: Number of available devices
        - offline: Number of unavailable devices
        - devices: List of device details with LQI, RSSI, battery info
        
    Examples:
        Returns Zigbee device list with connection quality
    Best Practices:
        - Use to diagnose Zigbee network issues
        - Check LQI values to identify weak connections
        - Monitor offline devices for troubleshooting
        - Verify power source and battery levels
    """
    logger.info("Getting ZHA devices")
    return await get_zha_devices_ha()


@mcp.tool()
@async_handler("get_esphome_devices")
async def get_esphome_devices() -> Dict[str, Any]:
    """
    Get list of ESPHome devices with their connection status
    
    Returns:
        A dictionary containing:
        - total_devices: Total number of ESPHome devices
        - online: Number of connected devices
        - offline: Number of disconnected devices
        - devices: List of devices with entity counts
        
    Examples:
        Returns ESPHome device list with connection status
    Best Practices:
        - Use to check ESPHome device connectivity
        - Identify devices that frequently disconnect
        - Monitor entity availability
        - Useful for network troubleshooting
    """
    logger.info("Getting ESPHome devices")
    return await get_esphome_devices_ha()


@mcp.tool()
@async_handler("get_addons")
async def get_addons() -> Dict[str, Any]:
    """
    Get list of Home Assistant addons with their status
    
    Returns:
        A dictionary containing:
        - total_addons: Total number of addons
        - installed: Number of installed addons
        - running: Number of running addons
        - stopped: Number of stopped addons
        - updates_available: Number of addons with updates
        - addons: List of addon details
        
    Examples:
        Returns addon list with status and update info
    Best Practices:
        - Use to check addon health and status
        - Monitor for available updates
        - Troubleshoot addon startup issues
        - Verify addon versions
    """
    logger.info("Getting addons list")
    return await get_addons_ha()


# ============================================================================
# DIAGNOSTIC TOOLS
# ============================================================================

@mcp.tool()
@async_handler("find_unavailable_entities")
async def find_unavailable_entities() -> Dict[str, Any]:
    """
    Find all entities that are currently unavailable
    
    Returns:
        A dictionary containing:
        - total_unavailable: Count of unavailable entities
        - by_domain: Entities grouped by domain
        - domain_counts: Count per domain
        - entities: List of unavailable entities (limited to 50)
        
    Examples:
        Detects broken integrations, offline devices, misconfigured entities
    Best Practices:
        - Run this regularly to catch connectivity issues
        - Check after integration updates or HA restarts
        - Correlate with error_log for root cause analysis
        - Pay attention to entire domains being unavailable (integration issue)
    """
    logger.info("Finding unavailable entities")
    return await find_unavailable_entities_ha()


@mcp.tool()
@async_handler("find_stale_entities")
async def find_stale_entities(hours: int = 2) -> Dict[str, Any]:
    """
    Find entities that haven't updated in the specified time period (frozen sensors)
    
    Args:
        hours: Number of hours to consider entity as stale (default: 2)
    
    Returns:
        A dictionary containing:
        - total_stale: Count of stale entities
        - threshold_hours: Hours threshold used
        - by_domain: Entities grouped by domain
        - domain_counts: Count per domain
        - entities: List of stale entities with hours_stale (limited to 50)
        
    Examples:
        hours=2 - find sensors frozen for 2+ hours
        hours=24 - find sensors frozen for a day
    Best Practices:
        - Use to detect frozen sensors or polling failures
        - Common causes: network issues, device sleep mode, integration bugs
        - Cross-reference with battery_report for battery-powered devices
        - Different thresholds for different sensor types (motion: 1hr, temp: 6hrs)
    """
    logger.info(f"Finding stale entities (threshold: {hours} hours)")
    return await find_stale_entities_ha(hours)


@mcp.tool()
@async_handler("battery_report")
async def battery_report() -> Dict[str, Any]:
    """
    Get report of all battery-powered devices with their battery levels
    
    Returns:
        A dictionary containing:
        - total_battery_entities: Total count of battery entities
        - low_battery_count: Count below 20%
        - critical_count: Count below 10%
        - low_battery: List of low battery devices (sorted by level)
        - all_batteries: All battery entities sorted by level (limited to 50)
        
    Examples:
        Returns comprehensive battery status for preventive maintenance
    Best Practices:
        - Run weekly to plan battery replacements
        - Focus on critical_count for immediate action
        - Low battery can cause device disconnections
        - Zigbee devices especially sensitive to low battery
        - Consider seasonal patterns (cold weather drains batteries faster)
    """
    logger.info("Generating battery report")
    return await battery_report_ha()


@mcp.tool()
@async_handler("get_repair_items")
async def get_repair_items() -> Dict[str, Any]:
    """
    Get all pending repair issues detected by Home Assistant
    
    Returns:
        A dictionary containing:
        - total_issues: Total count of issues
        - critical_count: Critical severity issues
        - error_count: Error severity issues
        - warning_count: Warning severity issues
        - by_severity: Issues grouped by severity (critical/error/warning/info)
        - by_domain: Issues grouped by integration domain
        
    Examples:
        Returns HA's built-in issue detection results
    Best Practices:
        - Address critical issues immediately
        - Check after HA updates or integration changes
        - Some issues auto-resolve after restart
        - Use learn_more_url for detailed fix instructions
        - Dismissed issues may reappear if root cause not fixed
    """
    logger.info("Getting repair items")
    return await get_repair_items_ha()


@mcp.tool()
@async_handler("get_update_status")
async def get_update_status() -> Dict[str, Any]:
    """
    Get status of all available updates (core, integrations, addons, devices)
    
    Returns:
        A dictionary containing:
        - total_updates_available: Count of available updates
        - core_updates: Home Assistant core updates
        - addon_updates: Supervisor addon updates
        - device_updates: Device firmware updates (ESPHome, etc)
        - all_updates: Complete list of available updates
        - up_to_date_count: Count of entities already up to date
        
    Examples:
        Returns categorized update information with version details
    Best Practices:
        - Review before applying updates (check release notes)
        - Backup before core updates
        - Test addon updates in non-production first
        - Device updates usually safe but verify compatibility
        - Check release_url for breaking changes
    """
    logger.info("Getting update status")
    return await get_update_status_ha()


@mcp.tool()
@async_handler("get_entity_statistics")
async def get_entity_statistics(entity_id: str, period_hours: int = 24) -> Dict[str, Any]:
    """
    Get statistical analysis for an entity over a time period
    
    Args:
        entity_id: The entity to analyze
        period_hours: Number of hours of history (default: 24, max: 168)
    
    Returns:
        A dictionary containing:
        - For numeric sensors: min, max, mean, median, std_dev, recent_value
        - For non-numeric: state_distribution, total_changes
        - Sample of recent state changes
        
    Examples:
        entity_id="sensor.temperature", period_hours=24
        entity_id="switch.living_room", period_hours=168
    Best Practices:
        - Use for trend analysis and anomaly detection
        - Check std_dev to identify unstable sensors
        - Compare mean vs recent_value for drift detection
        - Useful for capacity planning (battery drain, storage, etc)
    """
    logger.info(f"Getting statistics for {entity_id} over {period_hours} hours")
    return await get_entity_statistics_ha(entity_id, period_hours)


@mcp.tool()
@async_handler("find_anomalous_entities")
async def find_anomalous_entities() -> Dict[str, Any]:
    """
    Find entities with anomalous or impossible values
    
    Returns:
        A dictionary containing:
        - total_anomalies: Count of all anomalies found
        - impossible_values_count: Out-of-range values
        - frozen_sensors_count: Sensors not updating
        - anomalies: Detailed list by category
        
    Examples:
        Detects battery values outside 0-100%, temperatures outside reasonable ranges,
        humidity outside 0-100%, and other physical impossibilities
    Best Practices:
        - Run regularly to catch sensor malfunctions
        - Impossible values usually indicate hardware failure
        - Check sensor calibration if values are consistently off
        - May indicate need for sensor replacement or reconfiguration
    """
    logger.info("Finding anomalous entities")
    return await find_anomalous_entities_ha()


@mcp.tool()
@async_handler("recent_activity")
async def recent_activity(hours: int = 24) -> Dict[str, Any]:
    """
    Get recent activity from Home Assistant logbook
    
    Args:
        hours: Number of hours of activity to retrieve (default: 24)
    
    Returns:
        A dictionary containing:
        - total_events: Count of all events
        - by_domain: Event counts grouped by domain
        - most_active_entities: Top 10 most active entities
        - recent_events: Sample of 20 most recent events
        
    Examples:
        hours=24 - last 24 hours of activity
        hours=1 - last hour (useful for debugging)
    Best Practices:
        - Use to understand system behavior patterns
        - Identify noisy entities (too many state changes)
        - Debug automation triggers
        - Detect unusual activity patterns
        - Cross-reference with error_log for troubleshooting
    """
    logger.info(f"Getting recent activity for last {hours} hours")
    return await recent_activity_ha(hours)


@mcp.tool()
@async_handler("offline_devices_report")
async def offline_devices_report() -> Dict[str, Any]:
    """
    Get comprehensive report of offline/unavailable devices
    
    Returns:
        A dictionary containing:
        - total_offline_devices: Count of completely offline devices
        - total_unavailable_entities: Count of unavailable entities
        - by_manufacturer: Offline device counts by manufacturer
        - offline_devices: List of offline devices with details
        - entity_summary: Summary by domain
        
    Examples:
        Returns devices where ALL entities are unavailable
    Best Practices:
        - Use to identify connectivity issues by manufacturer
        - Check via_device to find hub/bridge problems
        - Manufacturer patterns may indicate integration issues
        - Combine with get_network_info for network troubleshooting
        - Check battery_report for battery-powered offline devices
    """
    logger.info("Generating offline devices report")
    return await offline_devices_report_ha()

# ============================================================================
# ADVANCED DIAGNOSTIC TOOLS
# ============================================================================

@mcp.tool()
@async_handler("identify_device")
async def identify_device(
    device_id_or_entity_id: str,
    pattern: str = "auto",
    duration: int = 3
) -> Dict[str, Any]:
    """
    Physically identify a device by making it flash, beep, or otherwise signal
    
    This tool helps locate devices in the real world by controlling them in
    distinctive patterns. Works with lights (flash), switches (toggle),
    media players (beep/TTS), and platform-specific methods (ZHA, ESPHome).
    
    Args:
        device_id_or_entity_id: Device ID or entity ID to identify
        pattern: Identification pattern ('auto' recommended, or 'flash', 'toggle', 'beep')
        duration: Duration in seconds for identification (default: 3)
    
    Returns:
        A dictionary containing:
        - success: Whether identification was successful
        - device_id: Device ID (if found)
        - entity_id: Entity ID used
        - entities_found: All entities for this device
        - platform_detected: Platform information
        - actions_executed: List of actions performed
        - alternative_methods: Manual identification methods if auto failed
        - notes: Additional information
    
    Examples:
        device_id_or_entity_id="light.living_room" - Flash living room light
        device_id_or_entity_id="abc123xyz", pattern="auto" - Auto-detect best method
        device_id_or_entity_id="switch.garden", duration=5 - Toggle for 5 seconds
    
    Best Practices:
        - Use 'auto' pattern to let the tool select best method
        - Works best with lights (visible flashing)
        - For ZHA devices, uses Zigbee identify cluster when available
        - Check alternative_methods if automatic identification fails
        - Useful for finding devices during installation or troubleshooting
    """
    logger.info(f"Identifying device: {device_id_or_entity_id}")
    return await identify_device_ha(device_id_or_entity_id, pattern, duration)


@mcp.tool()
@async_handler("diagnose_issue")
async def diagnose_issue(entity_id: str, ctx: Context = None) -> Dict[str, Any]:
    """
    Comprehensive diagnostic combining all available diagnostic tools
    
    This is the "AI Doctor" tool that performs deep analysis of any entity
    by checking availability, statistics, battery, logs, device status,
    platform-specific issues, and recent activity to identify root causes.
    
    Args:
        entity_id: Entity ID to diagnose (e.g., 'sensor.temperature')
        ctx: MCP Context for progress reporting
    
    Returns:
        A dictionary containing:
        - success: Whether diagnosis completed
        - entity_id: Entity being diagnosed
        - entity_summary: Basic entity information
        - severity: Issue severity (low/medium/high/critical)
        - root_cause_candidates: List of potential causes with confidence levels
        - recommended_fixes: Suggested actions to resolve issues
        - auto_fix_actions_available: Actions that can be automated
        - diagnostics_used: List of diagnostic tools used
        - diagnostics_count: Number of diagnostics performed
    
    Examples:
        entity_id="sensor.bedroom_motion_battery" - Diagnose battery sensor
        entity_id="light.living_room" - Diagnose light entity
        entity_id="sensor.temperature" - Diagnose temperature sensor
    
    Best Practices:
        - Combines find_unavailable_entities, find_stale_entities, battery_report,
          get_entity_statistics, find_anomalous_entities, recent_activity,
          offline_devices_report, get_error_log, and platform-specific checks
        - Provides severity levels to prioritize issues
        - Suggests both manual and automated fixes
        - Use for any entity experiencing problems
        - Check root_cause_candidates for most likely issues
        - Follow recommended_fixes in order of priority
    """
    if ctx:
        await ctx.info(f"🔍 Diagnosing entity: {entity_id}")
    
    logger.info(f"Diagnosing entity: {entity_id}")
    return await diagnose_issue_ha(entity_id, ctx=ctx)


# ============================================================================
# ORCHESTRATION & AUTO-FIX TOOLS
# ============================================================================

@mcp.tool()
@async_handler("diagnose_automation")
async def diagnose_automation(automation_id: str, ctx: Context = None) -> Dict[str, Any]:
    """
    Analyze an automation to find why it is not triggering or behaving as expected
    
    This tool performs comprehensive analysis of a Home Assistant automation by
    checking its state, trigger history, involved entities, error logs, and patterns
    to identify potential issues preventing proper execution.
    
    Args:
        automation_id: Entity ID of the automation (e.g., 'automation.living_room_lights')
        ctx: MCP Context for progress reporting
    
    Returns:
        A dictionary containing:
        - success: Whether analysis completed
        - automation_id: Automation being analyzed
        - automation_summary: State, last_triggered, mode, etc.
        - severity: Issue severity (low/medium/high/critical)
        - root_cause_candidates: Potential causes with confidence levels
        - recommended_fixes: Suggested actions to resolve issues
        - auto_fix_actions_available: Actions that can be automated
        - entities_involved: Entities referenced in automation
        - automation_events_24h: Trigger count in last 24 hours
        - diagnostics_used: List of diagnostic tools used
        - diagnostics_count: Number of diagnostics performed
    
    Examples:
        automation_id="automation.living_room_lights" - Diagnose lights automation
        automation_id="automation.morning_routine" - Diagnose morning routine
    
    Best Practices:
        - Use when automation not triggering as expected
        - Check automation_events_24h for loop detection
        - Review root_cause_candidates for most likely issues
        - Verifies automation is ON, entities valid, no errors in logs
        - Detects rapid re-triggering and potential loops
        - Analyzes mode (single/restart/queued) for blocking issues
    """
    logger.info(f"Diagnosing automation: {automation_id}")
    return await diagnose_automation_ha(automation_id, ctx=ctx)


@mcp.tool()
@async_handler("diagnose_system")
async def diagnose_system(include_entities: bool = False, ctx: Context = None) -> Dict[str, Any]:
    """
    Global system-level diagnostic orchestrator
    
    Performs comprehensive health check of entire Home Assistant system by calling
    all diagnostic tools, aggregating findings, scoring overall health (0-100%),
    and returning structured report grouped by categories.
    
    This is the "complete physical exam" for your Home Assistant instance.
    
    Args:
        include_entities: If True, includes detailed entity breakdown (more verbose)
        ctx: MCP Context for progress reporting
    
    Returns:
        A dictionary containing:
        - success: Whether diagnosis completed
        - timestamp: When diagnosis was performed
        - global_health_score: Overall health (0-100%)
        - overall_severity: System-wide severity (low/medium/high/critical)
        - severity_breakdown: Count of issues by severity
        - issues_by_category: Issues grouped by:
          * system (core HA health)
          * network (connectivity)
          * integrations (loaded integrations)
          * devices (offline devices)
          * entities (unavailable/stale)
          * batteries (low/critical batteries)
          * zigbee_mesh (ZHA signal quality)
          * esphome (ESPHome devices)
          * logs_errors (error/warning counts)
          * updates (available updates)
        - category_summaries: Issue count per category
        - total_issues: Total number of issues found
        - diagnostics_used: List of all diagnostic tools used
        - diagnostics_count: Number of diagnostics performed
        - entity_details: (if include_entities=True) Detailed entity breakdown
    
    Examples:
        diagnose_system() - Basic system health check
        diagnose_system(include_entities=True) - Detailed health check with entity breakdown
    
    Best Practices:
        - Run periodically to monitor system health
        - Health score <70: investigate high/critical issues
        - Health score <50: urgent attention needed
        - Check issues_by_category for targeted troubleshooting
        - Use include_entities=False for quick overview
        - Combines data from 14 diagnostic tools
        - Identifies weak Zigbee signals (LQI <120)
        - Detects offline devices by manufacturer
        - Calculates health score: 100 - (critical*10 + high*5 + medium*2 + low*0.5)
    """
    if ctx:
        await ctx.info("🔍 Starting comprehensive Home Assistant system diagnosis...")
    
    logger.info(f"Diagnosing system with include_entities={include_entities}")
    return await diagnose_system_ha(include_entities, ctx=ctx)


@mcp.tool()
@async_handler("auto_fix")
async def auto_fix(entity_id: str = None, scope: str = "auto", ctx: Context = None) -> Dict[str, Any]:
    """
    Perform safe, low-risk automated corrective actions
    
    This function automatically applies fixes that are safe and non-destructive.
    Can work on specific entity or perform global system fixes.
    
    SAFETY GUARANTEES:
    - NEVER restarts Home Assistant automatically
    - NEVER applies destructive changes
    - ONLY executes low-risk actions
    - ALWAYS logs actions taken
    - ALWAYS provides before/after snapshots
    
    Args:
        entity_id: Optional entity ID to fix (if None, performs global fixes)
        scope: Scope of fixes ('auto', 'entity', 'global')
        ctx: MCP Context for progress reporting and user confirmation
    
    Returns:
        A dictionary containing:
        - success: Whether auto-fix completed
        - entity_id: Entity fixed (if entity-specific)
        - scope: 'entity' or 'global'
        - actions_taken: List of successful fixes with timestamps
        - actions_skipped: List of skipped actions with reasons
        - risk_levels: Risk assessment for each action
        - before_snapshot: State before fixes
        - after_snapshot: State after fixes (includes improvement for global)
        - total_actions_taken: Count of successful actions
        - total_actions_skipped: Count of skipped actions
        - timestamp: When auto-fix was performed
    
    Examples:
        auto_fix() - Perform safe global system fixes
        auto_fix(entity_id="automation.living_room") - Fix specific automation
        auto_fix(entity_id="sensor.temperature") - Fix specific sensor
    
    Best Practices:
        - Entity-specific mode:
          * Runs diagnose_issue first
          * Applies suggested low-risk fixes
          * Includes before/after state snapshots
          * Can turn on disabled automations
        
        - Global mode:
          * Runs diagnose_system first
          * Reloads core config (safe)
          * Reloads scripts (safe)
          * Includes health score improvement metric
        
        - Safe actions executed:
          * reload_core_config
          * reload_scripts
          * turn_on_automation
        
        - Actions NEVER executed automatically:
          * restart_ha (too risky)
          * Any medium/high/critical risk actions
        
        - Review actions_skipped to see what was not safe to auto-apply
        - For global fixes, check after_snapshot.improvement for health score change
    """
    logger.info(f"Auto-fixing entity_id={entity_id}, scope={scope}")
    return await auto_fix_ha(entity_id, scope, ctx=ctx)


# ============================================================================
# NEW HIGH-IMPACT DIAGNOSTIC TOOLS (Hackathon Features)
# ============================================================================

@mcp.tool()
@async_handler("audit_zigbee_mesh")
async def audit_zigbee_mesh(limit: int = 100) -> Dict[str, Any]:
    """
    Advanced Zigbee mesh network analysis using ZHA diagnostic data.
    
    Analyzes the complete Zigbee mesh topology to identify:
    - LQI/RSSI distribution and weak links
    - Orphaned devices (no route to coordinator)
    - Routing path analysis
    - Coordinator health
    - Mesh stability metrics
    
    Args:
        limit: Maximum number of devices to analyze (default: 100)
    
    Returns:
        Dictionary containing:
        - mesh_health_score: Overall mesh health (0-100)
        - coordinator_info: Coordinator device details
        - lqi_distribution: Distribution of link quality
        - weak_links: Devices with LQI < 120
        - orphan_devices: Devices with no connection
        - routing_table: Device routing paths
        - recommendations: Suggested improvements
    
    Use Cases:
        - Identify weak points in Zigbee network
        - Plan router placement
        - Troubleshoot connectivity issues
        - Optimize mesh topology
    
    Performance:
        - Execution time: 3-5 seconds
        - Calls ZHA WebSocket API
        - Analyzes up to 100 devices
    
    Best Practices:
        - Run weekly to monitor mesh health
        - Add routers for weak links (LQI < 120)
        - Keep mesh_health_score above 80
        - Replace orphaned devices or improve placement
    """
    logger.info(f"Auditing Zigbee mesh network (limit: {limit})")
    return await audit_zigbee_mesh_ha(limit)


@mcp.tool()
@async_handler("find_orphan_entities")
async def find_orphan_entities() -> Dict[str, Any]:
    """
    Find entities that are NOT referenced in any automation, script, or scene.
    
    This tool identifies "orphan" entities that exist in Home Assistant but are
    not being used anywhere. Useful for:
    - Cleaning up unused entities
    - Identifying forgotten devices
    - Optimizing system performance
    - Finding misconfigured entities
    
    Returns:
        Dictionary containing:
        - total_entities: Total entity count
        - total_orphans: Count of unreferenced entities
        - orphans_by_domain: Orphans grouped by domain
        - orphan_entities: List of orphan entity details
        - usage_statistics: Entities sorted by usage count
    
    Use Cases:
        - System cleanup and optimization
        - Identify unused integrations
        - Find test entities to remove
        - Audit entity usage
    
    Performance:
        - Execution time: 5-10 seconds
        - Analyzes all automations, scripts, scenes
        - Parses entity references in configurations
    
    Best Practices:
        - Run monthly for system maintenance
        - Review orphans before deletion (some may be UI-only)
        - High orphan percentage (>30%) suggests cleanup needed
        - Check most_used_entities to understand critical dependencies
    """
    logger.info("Finding orphan entities")
    return await find_orphan_entities_ha()


@mcp.tool()
@async_handler("detect_automation_conflicts")
async def detect_automation_conflicts() -> Dict[str, Any]:
    """
    Detect race conditions, loops, and conflicts in automations.
    
    Analyzes all automations to identify:
    - Race conditions (multiple automations triggering on same entity)
    - Infinite loops (automation A triggers B, B triggers A)
    - Redundant automations (identical triggers)
    - Conflicting actions (opposing commands)
    - Unsafe modes (parallel execution risks)
    
    Returns:
        Dictionary containing:
        - total_automations: Total automation count
        - total_conflicts: Number of conflicts detected
        - race_conditions: Automations competing for same entities
        - potential_loops: Circular automation dependencies
        - redundant_automations: Duplicate trigger patterns
        - conflicting_actions: Opposing automation behaviors
        - unsafe_modes: Automations without proper mode settings
        - recommendations: Suggested fixes
    
    Use Cases:
        - Prevent automation conflicts
        - Identify infinite loop risks
        - Optimize automation organization
        - Improve system stability
    
    Performance:
        - Execution time: 8-15 seconds
        - Analyzes all automation configs
        - Deep parsing of triggers/conditions/actions
    
    Best Practices:
        - Run after adding/modifying automations
        - Address critical severity issues immediately
        - Use mode='restart' or 'queued' for rapid triggers
        - Add conditions to prevent race conditions
        - Review potential_loops carefully (can cause system instability)
    """
    logger.info("Detecting automation conflicts")
    return await detect_automation_conflicts_ha()


@mcp.tool()
@async_handler("energy_consumption_report")
async def energy_consumption_report(period_hours: int = 24) -> Dict[str, Any]:
    """
    Generate comprehensive energy consumption report from Home Assistant Energy Dashboard.
    
    Analyzes energy data to provide:
    - Total consumption by period
    - Consumption trends
    - Peak usage hours
    - Device-level consumption (if available)
    - Cost estimation
    - Comparison to previous periods
    
    Args:
        period_hours: Hours of data to analyze (default: 24, max: 168)
    
    Returns:
        Dictionary containing:
        - total_consumption: Total energy used (kWh)
        - consumption_by_hour: Hourly breakdown
        - peak_hours: Hours with highest consumption
        - device_consumption: Per-device usage (if available)
        - cost_estimate: Estimated cost
        - trends: Consumption patterns
        - recommendations: Energy saving suggestions
    
    Use Cases:
        - Monitor energy usage
        - Identify energy-hungry devices
        - Plan energy-saving automations
        - Cost tracking and budgeting
    
    Performance:
        - Execution time: 4-8 seconds
        - Queries energy sensor history
        - Analyzes multiple energy entities
    
    Best Practices:
        - Run daily for trend monitoring
        - Focus on top_consumers for optimization
        - Create automations to reduce peak usage
        - Set energy_rate for accurate cost estimates
        - Use monthly_projection for budgeting
    """
    logger.info(f"Generating energy consumption report (period: {period_hours}h)")
    return await energy_consumption_report_ha(period_hours)


@mcp.tool()
@async_handler("entity_dependency_graph")
async def entity_dependency_graph(entity_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate dependency graph showing which entities reference other entities.
    
    Creates a map of entity relationships by analyzing:
    - Automation triggers and actions
    - Script sequences
    - Template sensors and helpers
    - Group entities
    - Scene definitions
    
    Args:
        entity_id: Optional specific entity to analyze (if None, analyzes all)
    
    Returns:
        Dictionary containing:
        - dependency_map: Entity relationships
        - entities_referenced: Most-referenced entities
        - entities_referencing: Entities that reference others
        - isolated_entities: Entities with no dependencies
        - circular_dependencies: Potential circular references
    
    Use Cases:
        - Understand entity relationships
        - Identify critical entities (highly referenced)
        - Find impact of removing an entity
        - Detect circular dependencies
        - Plan automation reorganization
    
    Performance:
        - Execution time: 10-20 seconds (full analysis)
        - Execution time: 2-4 seconds (single entity)
        - Parses all automation/script/template configs
    
    Best Practices:
        - Use entity_id parameter for targeted analysis
        - Review most_referenced_entities to identify critical dependencies
        - Check isolated_entities for potential cleanup
        - Address circular_dependencies (can cause issues)
        - Use this before removing entities to understand impact
    """
    logger.info(f"Generating entity dependency graph" + (f" for {entity_id}" if entity_id else ""))
    return await entity_dependency_graph_ha(entity_id)


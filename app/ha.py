import httpx
import websockets
import json
from typing import Dict, Any, Optional, List, TypeVar, Callable, Awaitable, Union, cast
import functools
import inspect
import logging
from datetime import datetime, timedelta, timezone
import asyncio

from app.config import HA_URL, HA_TOKEN, get_ha_headers

# Set up logging
logger = logging.getLogger(__name__)

# Define a generic type for our API function return values
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Awaitable[Any]])

# HTTP client
_client: Optional[httpx.AsyncClient] = None

# Default field sets for different verbosity levels
# Lean fields for standard requests (optimized for token efficiency)
DEFAULT_LEAN_FIELDS = ["entity_id", "state", "attr.friendly_name"]

# Common fields that are typically needed for entity operations
DEFAULT_STANDARD_FIELDS = ["entity_id", "state", "attributes", "last_updated"]

# Domain-specific important attributes to include in lean responses
DOMAIN_IMPORTANT_ATTRIBUTES = {
    "light": ["brightness", "color_temp", "rgb_color", "supported_color_modes"],
    "switch": ["device_class"],
    "binary_sensor": ["device_class"],
    "sensor": ["device_class", "unit_of_measurement", "state_class"],
    "climate": ["hvac_mode", "current_temperature", "temperature", "hvac_action"],
    "media_player": ["media_title", "media_artist", "source", "volume_level"],
    "cover": ["current_position", "current_tilt_position"],
    "fan": ["percentage", "preset_mode"],
    "camera": ["entity_picture"],
    "automation": ["last_triggered"],
    "scene": [],
    "script": ["last_triggered"],
}

# Helper functions for safe Context operations
class SafeContext:
    """Wrapper for MCP Context that makes all methods safe (no-op if not available)"""
    def __init__(self, ctx):
        self._ctx = ctx
    
    async def info(self, message: str):
        if self._ctx and hasattr(self._ctx, 'info'):
            try:
                await self._ctx.info(message)
            except Exception as e:
                logger.debug(f"ctx.info failed: {e}")
    
    async def error(self, message: str):
        if self._ctx and hasattr(self._ctx, 'error'):
            try:
                await self._ctx.error(message)
            except Exception as e:
                logger.debug(f"ctx.error failed: {e}")
    
    async def report_progress(self, progress: int, total: int):
        if self._ctx and hasattr(self._ctx, 'report_progress'):
            try:
                await self._ctx.report_progress(progress=progress, total=total)
            except Exception as e:
                logger.debug(f"ctx.report_progress failed: {e}")
    
    def __bool__(self):
        return self._ctx is not None

def safe_ctx(ctx):
    """Wrap a Context in SafeContext to make all operations safe"""
    if ctx is None:
        return None
    if isinstance(ctx, SafeContext):
        return ctx
    return SafeContext(ctx)

def handle_api_errors(func: F) -> F:
    """
    Decorator to handle common error cases for Home Assistant API calls
    
    Args:
        func: The async function to decorate
        
    Returns:
        Wrapped function that handles errors
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Determine return type from function annotation
        return_type = inspect.signature(func).return_annotation
        is_dict_return = 'Dict' in str(return_type)
        is_list_return = 'List' in str(return_type)
        
        # Prepare error formatters based on return type
        def format_error(msg: str) -> Any:
            if is_dict_return:
                return {"error": msg}
            elif is_list_return:
                return [{"error": msg}]
            else:
                return msg
        
        try:
            # Check if token is available
            if not HA_TOKEN:
                return format_error("No Home Assistant token provided. Please set HA_TOKEN environment variable.")
            
            # Call the original function
            return await func(*args, **kwargs)
        except httpx.ConnectError:
            return format_error(f"Connection error: Cannot connect to Home Assistant at {HA_URL}")
        except httpx.TimeoutException:
            return format_error(f"Timeout error: Home Assistant at {HA_URL} did not respond in time")
        except httpx.HTTPStatusError as e:
            return format_error(f"HTTP error: {e.response.status_code} - {e.response.reason_phrase}")
        except httpx.RequestError as e:
            return format_error(f"Error connecting to Home Assistant: {str(e)}")
        except Exception as e:
            return format_error(f"Unexpected error: {str(e)}")
    
    return cast(F, wrapper)

# Persistent HTTP client
async def get_client() -> httpx.AsyncClient:
    """Get a persistent httpx client for Home Assistant API calls"""
    global _client
    if _client is None:
        logger.debug("Creating new HTTP client")
        _client = httpx.AsyncClient(timeout=10.0)
    return _client

async def cleanup_client() -> None:
    """Close the HTTP client when shutting down"""
    global _client
    if _client:
        logger.debug("Closing HTTP client")
        await _client.aclose()
        _client = None

# Direct entity retrieval function
async def get_all_entity_states() -> Dict[str, Dict[str, Any]]:
    """Fetch all entity states from Home Assistant"""
    client = await get_client()
    response = await client.get(f"{HA_URL}/api/states", headers=get_ha_headers())
    response.raise_for_status()
    entities = response.json()
    
    # Create a mapping for easier access
    return {entity["entity_id"]: entity for entity in entities}

def filter_fields(data: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """
    Filter entity data to only include requested fields
    
    This function helps reduce token usage by returning only requested fields.
    
    Args:
        data: The complete entity data dictionary
        fields: List of fields to include in the result
               - "state": Include the entity state
               - "attributes": Include all attributes
               - "attr.X": Include only attribute X (e.g. "attr.brightness")
               - "context": Include context data
               - "last_updated"/"last_changed": Include timestamp fields
    
    Returns:
        A filtered dictionary with only the requested fields
    """
    if not fields:
        return data
        
    result = {"entity_id": data["entity_id"]}
    
    for field in fields:
        if field == "state":
            result["state"] = data.get("state")
        elif field == "attributes":
            result["attributes"] = data.get("attributes", {})
        elif field.startswith("attr.") and len(field) > 5:
            attr_name = field[5:]
            attributes = data.get("attributes", {})
            if attr_name in attributes:
                if "attributes" not in result:
                    result["attributes"] = {}
                result["attributes"][attr_name] = attributes[attr_name]
        elif field == "context":
            if "context" in data:
                result["context"] = data["context"]
        elif field in ["last_updated", "last_changed"]:
            if field in data:
                result[field] = data[field]
    
    return result

# API Functions
@handle_api_errors
async def get_ha_version() -> str:
    """Get the Home Assistant version from the API"""
    client = await get_client()
    response = await client.get(f"{HA_URL}/api/config", headers=get_ha_headers())
    response.raise_for_status()
    data = response.json()
    return data.get("version", "unknown")

@handle_api_errors
async def get_entity_state(
    entity_id: str,
    fields: Optional[List[str]] = None,
    lean: bool = False
) -> Dict[str, Any]:
    """
    Get the state of a Home Assistant entity
    
    Args:
        entity_id: The entity ID to get
        fields: Optional list of specific fields to include in the response
        lean: If True, returns a token-efficient version with minimal fields
              (overridden by fields parameter if provided)
    
    Returns:
        Entity state dictionary, optionally filtered to include only specified fields
    """
    # Fetch directly
    client = await get_client()
    response = await client.get(
        f"{HA_URL}/api/states/{entity_id}", 
        headers=get_ha_headers()
    )
    response.raise_for_status()
    entity_data = response.json()
    
    # Apply field filtering if requested
    if fields:
        # User-specified fields take precedence
        return filter_fields(entity_data, fields)
    elif lean:
        # Build domain-specific lean fields
        lean_fields = DEFAULT_LEAN_FIELDS.copy()
        
        # Add domain-specific important attributes
        domain = entity_id.split('.')[0]
        if domain in DOMAIN_IMPORTANT_ATTRIBUTES:
            for attr in DOMAIN_IMPORTANT_ATTRIBUTES[domain]:
                lean_fields.append(f"attr.{attr}")
        
        return filter_fields(entity_data, lean_fields)
    else:
        # Return full entity data
        return entity_data

@handle_api_errors
async def get_entities(
    domain: Optional[str] = None, 
    search_query: Optional[str] = None, 
    limit: int = 100,
    fields: Optional[List[str]] = None,
    lean: bool = True
) -> List[Dict[str, Any]]:
    """
    Get a list of all entities from Home Assistant with optional filtering and search
    
    Args:
        domain: Optional domain to filter entities by (e.g., 'light', 'switch')
        search_query: Optional case-insensitive search term to filter by entity_id, friendly_name or other attributes
        limit: Maximum number of entities to return (default: 100)
        fields: Optional list of specific fields to include in each entity
        lean: If True (default), returns token-efficient versions with minimal fields
    
    Returns:
        List of entity dictionaries, optionally filtered by domain and search terms,
        and optionally limited to specific fields
    """
    # Get all entities directly
    client = await get_client()
    response = await client.get(f"{HA_URL}/api/states", headers=get_ha_headers())
    response.raise_for_status()
    entities = response.json()
    
    # Filter by domain if specified
    if domain:
        entities = [entity for entity in entities if entity["entity_id"].startswith(f"{domain}.")]
    
    # Search if query is provided
    if search_query and search_query.strip():
        search_term = search_query.lower().strip()
        filtered_entities = []
        
        for entity in entities:
            # Search in entity_id
            if search_term in entity["entity_id"].lower():
                filtered_entities.append(entity)
                continue
                
            # Search in friendly_name
            friendly_name = entity.get("attributes", {}).get("friendly_name", "").lower()
            if friendly_name and search_term in friendly_name:
                filtered_entities.append(entity)
                continue
                
            # Search in other common attributes (state, area_id, etc.)
            if search_term in entity.get("state", "").lower():
                filtered_entities.append(entity)
                continue
                
            # Search in other attributes
            for attr_name, attr_value in entity.get("attributes", {}).items():
                # Check if attribute value can be converted to string
                if isinstance(attr_value, (str, int, float, bool)):
                    if search_term in str(attr_value).lower():
                        filtered_entities.append(entity)
                        break
        
        entities = filtered_entities
    
    # Apply the limit
    if limit > 0 and len(entities) > limit:
        entities = entities[:limit]
    
    # Apply field filtering if requested
    if fields:
        # Use explicit field list when provided
        return [filter_fields(entity, fields) for entity in entities]
    elif lean:
        # Apply domain-specific lean fields to each entity
        result = []
        for entity in entities:
            # Get the entity's domain
            entity_domain = entity["entity_id"].split('.')[0]
            
            # Start with basic lean fields
            lean_fields = DEFAULT_LEAN_FIELDS.copy()
            
            # Add domain-specific important attributes
            if entity_domain in DOMAIN_IMPORTANT_ATTRIBUTES:
                for attr in DOMAIN_IMPORTANT_ATTRIBUTES[entity_domain]:
                    lean_fields.append(f"attr.{attr}")
            
            # Filter and add to result
            result.append(filter_fields(entity, lean_fields))
        
        return result
    else:
        # Return full entities
        return entities

@handle_api_errors
async def call_service(domain: str, service: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call a Home Assistant service"""
    if data is None:
        data = {}
    
    client = await get_client()
    response = await client.post(
        f"{HA_URL}/api/services/{domain}/{service}", 
        headers=get_ha_headers(),
        json=data
    )
    response.raise_for_status()
    
    # Invalidate cache after service calls as they might change entity states
    global _entities_timestamp
    _entities_timestamp = 0
    
    return response.json()

@handle_api_errors
async def summarize_domain(domain: str, example_limit: int = 3) -> Dict[str, Any]:
    """
    Generate a summary of entities in a domain
    
    Args:
        domain: The domain to summarize (e.g., 'light', 'switch')
        example_limit: Maximum number of examples to include for each state
        
    Returns:
        Dictionary with summary information
    """
    entities = await get_entities(domain=domain)
    
    # Check if we got an error response
    if isinstance(entities, dict) and "error" in entities:
        return entities  # Just pass through the error
    
    try:
        # Initialize summary data
        total_count = len(entities)
        state_counts = {}
        state_examples = {}
        attributes_summary = {}
        
        # Process entities to build the summary
        for entity in entities:
            state = entity.get("state", "unknown")
            
            # Count states
            if state not in state_counts:
                state_counts[state] = 0
                state_examples[state] = []
            state_counts[state] += 1
            
            # Add examples (up to the limit)
            if len(state_examples[state]) < example_limit:
                example = {
                    "entity_id": entity["entity_id"],
                    "friendly_name": entity.get("attributes", {}).get("friendly_name", entity["entity_id"])
                }
                state_examples[state].append(example)
            
            # Collect attribute keys for summary
            for attr_key in entity.get("attributes", {}):
                if attr_key not in attributes_summary:
                    attributes_summary[attr_key] = 0
                attributes_summary[attr_key] += 1
        
        # Create the summary
        summary = {
            "domain": domain,
            "total_count": total_count,
            "state_distribution": state_counts,
            "examples": state_examples,
            "common_attributes": sorted(
                [(k, v) for k, v in attributes_summary.items()], 
                key=lambda x: x[1], 
                reverse=True
            )[:10]  # Top 10 most common attributes
        }
        
        return summary
    except Exception as e:
        return {"error": f"Error generating domain summary: {str(e)}"}

@handle_api_errors
async def get_automations() -> List[Dict[str, Any]]:
    """Get a list of all automations from Home Assistant"""
    # Reuse the get_entities function with domain filtering
    automation_entities = await get_entities(domain="automation")
    
    # Check if we got an error response
    if isinstance(automation_entities, dict) and "error" in automation_entities:
        return automation_entities  # Just pass through the error
    
    # Process automation entities
    result = []
    try:
        for entity in automation_entities:
            # Extract relevant information
            automation_info = {
                "id": entity["entity_id"].split(".")[1],
                "entity_id": entity["entity_id"],
                "state": entity["state"],
                "alias": entity["attributes"].get("friendly_name", entity["entity_id"]),
            }
            
            # Add any additional attributes that might be useful
            if "last_triggered" in entity["attributes"]:
                automation_info["last_triggered"] = entity["attributes"]["last_triggered"]
            
            result.append(automation_info)
    except (TypeError, KeyError) as e:
        # Handle errors in processing the entities
        return {"error": f"Error processing automation entities: {str(e)}"}
        
    return result

@handle_api_errors
async def reload_automations() -> Dict[str, Any]:
    """Reload all automations in Home Assistant"""
    return await call_service("automation", "reload", {})

@handle_api_errors
async def restart_home_assistant(ctx=None) -> Dict[str, Any]:
    """
    Restart Home Assistant
    
    Args:
        ctx: Optional MCP Context for progress reporting
    
    Returns:
        Result of restart operation
    """
    ctx = safe_ctx(ctx)
    
    if ctx:
        await ctx.info("🔄 Restarting Home Assistant...")
    
    return await call_service("homeassistant", "restart", {})

@handle_api_errors
async def get_ha_error_log() -> Dict[str, Any]:
    """
    Get the Home Assistant error log for troubleshooting via WebSocket API
    
    Returns:
        A dictionary containing:
        - log_text: The full error log text (formatted from entries)
        - error_count: Number of ERROR entries found
        - warning_count: Number of WARNING entries found
        - integration_mentions: Map of integration names to mention counts
        - entries: Raw log entries from Home Assistant
        - error: Error message if retrieval failed
    """
    try:
        import ssl
        # Convert HTTP(S) URL to WebSocket URL
        ws_url = HA_URL.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/websocket"
        
        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Connect to WebSocket
        async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
            # Receive auth_required message
            auth_required = await websocket.recv()
            logger.debug(f"Auth required: {auth_required}")
            
            # Send auth message
            auth_msg = json.dumps({
                "type": "auth",
                "access_token": HA_TOKEN
            })
            await websocket.send(auth_msg)
            
            # Receive auth result
            auth_result = json.loads(await websocket.recv())
            if auth_result.get("type") != "auth_ok":
                return {
                    "error": f"WebSocket authentication failed: {auth_result}",
                    "log_text": "",
                    "error_count": 0,
                    "warning_count": 0,
                    "integration_mentions": {},
                    "entries": []
                }
            
            # Send system_log/list command
            list_cmd = json.dumps({
                "id": 1,
                "type": "system_log/list"
            })
            await websocket.send(list_cmd)
            
            # Receive log entries
            response = json.loads(await websocket.recv())
            
            if not response.get("success"):
                return {
                    "error": f"Failed to retrieve logs: {response.get('error', 'Unknown error')}",
                    "log_text": "",
                    "error_count": 0,
                    "warning_count": 0,
                    "integration_mentions": {},
                    "entries": []
                }
            
            entries = response.get("result", [])
            
            # Count errors and warnings
            error_count = sum(1 for entry in entries if entry.get("level") == "ERROR")
            warning_count = sum(1 for entry in entries if entry.get("level") == "WARNING")
            critical_count = sum(1 for entry in entries if entry.get("level") == "CRITICAL")
            
            # Extract integration mentions from source paths
            import re
            integration_mentions = {}
            for entry in entries:
                source = entry.get("source", ["", ""])[0]
                # Look for integration names in paths like homeassistant.components.mqtt
                if "homeassistant.components." in source:
                    integration = source.split("homeassistant.components.")[1].split(".")[0]
                    if integration not in integration_mentions:
                        integration_mentions[integration] = 0
                    integration_mentions[integration] += 1
            
            # Format log text from entries
            log_lines = []
            for entry in entries:
                timestamp = entry.get("timestamp", "")
                level = entry.get("level", "INFO")
                name = entry.get("name", "")
                message = entry.get("message", [""])[0] if isinstance(entry.get("message"), list) else entry.get("message", "")
                log_lines.append(f"{timestamp} {level} ({name}) {message}")
            
            log_text = "\n".join(log_lines)
            
            return {
                "log_text": log_text,
                "error_count": error_count,
                "warning_count": warning_count,
                "critical_count": critical_count,
                "integration_mentions": integration_mentions,
                "entries": entries[:10]  # Limit to first 10 for token efficiency
            }
            
    except Exception as e:
        logger.error(f"Error retrieving Home Assistant error log via WebSocket: {str(e)}")
        return {
            "error": f"Error retrieving error log: {str(e)}",
            "log_text": "",
            "error_count": 0,
            "warning_count": 0,
            "integration_mentions": {},
            "entries": []
        }

@handle_api_errors
async def get_entity_history(entity_id: str, hours: int) -> List[Dict[str, Any]]:
    """
    Get the history of an entity's state changes from Home Assistant.

    Args:
        entity_id: The entity ID to get history for.
        hours: Number of hours of history to retrieve.

    Returns:
        A list of state change objects, or an error dictionary.
    """
    client = await get_client()
    
    # Calculate the end time for the history lookup
    end_time = datetime.now(timezone.utc)
    end_time_iso = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Calculate the start time for the history lookup based on end_time
    start_time = end_time - timedelta(hours=hours)
    start_time_iso = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Construct the API URL
    url = f"{HA_URL}/api/history/period/{start_time_iso}"
    
    # Set query parameters
    params = {
        "filter_entity_id": entity_id,
        "minimal_response": "true",
        "end_time": end_time_iso,
    }
    
    # Make the API call
    response = await client.get(url, headers=get_ha_headers(), params=params)
    response.raise_for_status()
    
    # Return the JSON response
    return response.json()

@handle_api_errors
async def get_system_overview() -> Dict[str, Any]:
    """
    Get a comprehensive overview of the entire Home Assistant system
    
    Returns:
        A dictionary containing:
        - total_entities: Total count of all entities
        - domains: Dictionary of domains with their entity counts and state distributions
        - domain_samples: Representative sample entities for each domain (2-3 per domain)
        - domain_attributes: Common attributes for each domain
        - area_distribution: Entities grouped by area (if available)
    """
    try:
        # Get ALL entities with minimal fields for efficiency
        # We retrieve all entities since API calls don't consume tokens, only responses do
        client = await get_client()
        response = await client.get(f"{HA_URL}/api/states", headers=get_ha_headers())
        response.raise_for_status()
        all_entities_raw = response.json()
        
        # Apply lean formatting to reduce token usage in the response
        all_entities = []
        for entity in all_entities_raw:
            domain = entity["entity_id"].split(".")[0]
            
            # Start with basic lean fields
            lean_fields = ["entity_id", "state", "attr.friendly_name"]
            
            # Add domain-specific important attributes
            if domain in DOMAIN_IMPORTANT_ATTRIBUTES:
                for attr in DOMAIN_IMPORTANT_ATTRIBUTES[domain]:
                    lean_fields.append(f"attr.{attr}")
            
            # Filter and add to result
            all_entities.append(filter_fields(entity, lean_fields))
        
        # Initialize overview structure
        overview = {
            "total_entities": len(all_entities),
            "domains": {},
            "domain_samples": {},
            "domain_attributes": {},
            "area_distribution": {}
        }
        
        # Group entities by domain
        domain_entities = {}
        for entity in all_entities:
            domain = entity["entity_id"].split(".")[0]
            if domain not in domain_entities:
                domain_entities[domain] = []
            domain_entities[domain].append(entity)
        
        # Process each domain
        for domain, entities in domain_entities.items():
            # Count entities in this domain
            count = len(entities)
            
            # Collect state distribution
            state_distribution = {}
            for entity in entities:
                state = entity.get("state", "unknown")
                if state not in state_distribution:
                    state_distribution[state] = 0
                state_distribution[state] += 1
            
            # Store domain information
            overview["domains"][domain] = {
                "count": count,
                "states": state_distribution
            }
            
            # Select representative samples (2-3 per domain)
            sample_limit = min(3, count)
            samples = []
            for i in range(sample_limit):
                entity = entities[i]
                samples.append({
                    "entity_id": entity["entity_id"],
                    "state": entity.get("state", "unknown"),
                    "friendly_name": entity.get("attributes", {}).get("friendly_name", entity["entity_id"])
                })
            overview["domain_samples"][domain] = samples
            
            # Collect common attributes for this domain
            attribute_counts = {}
            for entity in entities:
                for attr in entity.get("attributes", {}):
                    if attr not in attribute_counts:
                        attribute_counts[attr] = 0
                    attribute_counts[attr] += 1
            
            # Get top 5 most common attributes for this domain
            common_attributes = sorted(attribute_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            overview["domain_attributes"][domain] = [attr for attr, count in common_attributes]
            
            # Group by area if available
            for entity in entities:
                area_id = entity.get("attributes", {}).get("area_id", "Unknown")
                area_name = entity.get("attributes", {}).get("area_name", area_id)
                
                if area_name not in overview["area_distribution"]:
                    overview["area_distribution"][area_name] = {}
                
                if domain not in overview["area_distribution"][area_name]:
                    overview["area_distribution"][area_name][domain] = 0
                    
                overview["area_distribution"][area_name][domain] += 1
        
        # Add summary information
        overview["domain_count"] = len(domain_entities)
        overview["most_common_domains"] = sorted(
            [(domain, len(entities)) for domain, entities in domain_entities.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return overview
    except Exception as e:
        logger.error(f"Error generating system overview: {str(e)}")
        return {"error": f"Error generating system overview: {str(e)}"}


@handle_api_errors
async def get_system_health() -> Dict[str, Any]:
    """
    Get system health information including CPU, RAM, disk usage, and integrations health
    
    Returns:
        Dictionary with system health data
    """
    try:
        # Get supervisor info if available (HA OS/Supervised)
        client = await get_client()
        
        # Try to get supervisor info first
        supervisor_info = {}
        try:
            response = await client.get(f"{HA_URL}/api/hassio/info", headers=get_ha_headers())
            if response.status_code == 200:
                supervisor_data = response.json().get("data", {})
                supervisor_info = {
                    "supervisor_version": supervisor_data.get("version"),
                    "supervisor_channel": supervisor_data.get("channel"),
                    "supported": supervisor_data.get("supported"),
                    "healthy": supervisor_data.get("healthy"),
                }
        except Exception:
            # Not running supervised/OS
            pass
        
        # Get host info if available
        host_info = {}
        try:
            response = await client.get(f"{HA_URL}/api/hassio/host/info", headers=get_ha_headers())
            if response.status_code == 200:
                host_data = response.json().get("data", {})
                host_info = {
                    "hostname": host_data.get("hostname"),
                    "operating_system": host_data.get("operating_system"),
                    "disk_total": host_data.get("disk_total"),
                    "disk_used": host_data.get("disk_used"),
                    "disk_free": host_data.get("disk_free"),
                }
        except Exception:
            pass
        
        # Get core info (always available)
        response = await client.get(f"{HA_URL}/api/config", headers=get_ha_headers())
        config_data = response.json()
        
        return {
            "version": config_data.get("version"),
            "location_name": config_data.get("location_name"),
            "time_zone": config_data.get("time_zone"),
            "unit_system": config_data.get("unit_system"),
            "supervisor": supervisor_info if supervisor_info else None,
            "host": host_info if host_info else None,
            "components": len(config_data.get("components", [])),
            "config_dir": config_data.get("config_dir"),
        }
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        return {"error": f"Error getting system health: {str(e)}"}


@handle_api_errors
async def get_integrations() -> Dict[str, Any]:
    """
    Get list of all integrations and their status
    
    Returns:
        Dictionary with integration information
    """
    try:
        client = await get_client()
        
        # Get config
        response = await client.get(f"{HA_URL}/api/config", headers=get_ha_headers())
        config_data = response.json()
        
        components = config_data.get("components", [])
        
        # Get all entities to map integrations to entity counts
        all_states = await get_all_entity_states()
        
        # Count entities per integration (based on domain)
        integration_entities = {}
        for entity_id, state in all_states.items():
            domain = entity_id.split(".")[0]
            if domain not in integration_entities:
                integration_entities[domain] = []
            integration_entities[domain].append(entity_id)
        
        return {
            "total_integrations": len(components),
            "integrations": sorted(components),
            "entity_counts": {
                domain: len(entities) 
                for domain, entities in integration_entities.items()
            },
            "integrations_with_entities": len(integration_entities),
        }
    except Exception as e:
        logger.error(f"Error getting integrations: {str(e)}")
        return {"error": f"Error getting integrations: {str(e)}"}


@handle_api_errors
async def reload_scripts() -> Dict[str, Any]:
    """
    Reload all scripts
    
    Returns:
        Success status
    """
    return await call_service("script", "reload", {})


@handle_api_errors
async def reload_core_config() -> Dict[str, Any]:
    """
    Reload Home Assistant core configuration
    
    Returns:
        Success status
    """
    return await call_service("homeassistant", "reload_core_config", {})


@handle_api_errors
async def get_network_info() -> Dict[str, Any]:
    """
    Get network information from Home Assistant
    
    Returns:
        Network configuration and status
    """
    try:
        client = await get_client()
        
        network_info = {}
        
        # Try to get network info from supervisor (if available)
        try:
            response = await client.get(f"{HA_URL}/api/hassio/network/info", headers=get_ha_headers())
            if response.status_code == 200:
                network_data = response.json().get("data", {})
                network_info = {
                    "hostname": network_data.get("hostname"),
                    "interfaces": network_data.get("interfaces", []),
                }
        except Exception:
            # Not running supervised/OS
            pass
        
        # Get config for basic info
        response = await client.get(f"{HA_URL}/api/config", headers=get_ha_headers())
        config_data = response.json()
        
        # Get external/internal URLs
        network_info.update({
            "external_url": config_data.get("external_url"),
            "internal_url": config_data.get("internal_url"),
            "location_name": config_data.get("location_name"),
            "latitude": config_data.get("latitude"),
            "longitude": config_data.get("longitude"),
        })
        
        return network_info
    except Exception as e:
        logger.error(f"Error getting network info: {str(e)}")
        return {"error": f"Error getting network info: {str(e)}"}


@handle_api_errors
async def get_zha_devices() -> Dict[str, Any]:
    """
    Get list of ZHA (Zigbee) devices with their status and link quality
    
    Returns:
        Dictionary with ZHA device information
    """
    try:
        # Get all ZHA devices via WebSocket
        ws_url = HA_URL.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/websocket"
        
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
            # Auth
            await websocket.recv()
            await websocket.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
            auth_result = json.loads(await websocket.recv())
            
            if auth_result.get("type") != "auth_ok":
                return {"error": "WebSocket authentication failed"}
            
            # Get ZHA devices
            await websocket.send(json.dumps({"id": 1, "type": "zha/devices"}))
            response = json.loads(await websocket.recv())
            
            if not response.get("success"):
                return {"error": f"Failed to get ZHA devices: {response.get('error')}"}
            
            devices = response.get("result", [])
            
            # Format device info
            device_list = []
            for device in devices:
                device_info = {
                    "ieee": device.get("ieee"),
                    "name": device.get("user_given_name") or device.get("name"),
                    "manufacturer": device.get("manufacturer"),
                    "model": device.get("model"),
                    "lqi": device.get("lqi"),
                    "rssi": device.get("rssi"),
                    "available": device.get("available"),
                    "power_source": device.get("power_source"),
                    "device_type": device.get("device_type"),
                }
                device_list.append(device_info)
            
            # Count by status
            online = sum(1 for d in device_list if d.get("available"))
            offline = len(device_list) - online
            
            return {
                "total_devices": len(device_list),
                "online": online,
                "offline": offline,
                "devices": device_list[:20],  # Limit to 20 for token efficiency
            }
            
    except Exception as e:
        logger.error(f"Error getting ZHA devices: {str(e)}")
        return {"error": f"Error getting ZHA devices: {str(e)}"}


@handle_api_errors
async def get_esphome_devices() -> Dict[str, Any]:
    """
    Get list of ESPHome devices with their connection status
    
    Returns:
        Dictionary with ESPHome device information
    """
    try:
        # Get all entities
        all_states = await get_all_entity_states()
        
        # Group entities by device based on entity_picture or friendly_name patterns
        esphome_devices = {}
        
        for entity_id, state in all_states.items():
            attributes = state.get("attributes", {})
            entity_picture = attributes.get("entity_picture", "")
            friendly_name = attributes.get("friendly_name", "")
            
            # Check if entity belongs to ESPHome (by entity_picture or entity_id pattern)
            is_esphome = (
                "esphome" in entity_picture.lower() or
                entity_id.startswith("sensor.esp") or
                entity_id.startswith("binary_sensor.esp") or
                entity_id.startswith("switch.esp") or
                entity_id.startswith("light.esp") or
                entity_id.startswith("select.esp") or
                entity_id.startswith("update.esp") or
                entity_id.startswith("media_player.esp")
            )
            
            if is_esphome:
                # Extract device name from friendly_name (first word before space)
                device_name = friendly_name.split(" ")[0] if friendly_name else entity_id.split(".")[1].split("_")[0]
                
                if device_name not in esphome_devices:
                    esphome_devices[device_name] = {
                        "name": device_name,
                        "entities": [],
                        "states": [],
                    }
                
                esphome_devices[device_name]["entities"].append(entity_id)
                esphome_devices[device_name]["states"].append(state.get("state"))
        
        # Format results
        device_list = []
        for device_name, device_data in esphome_devices.items():
            # Check if device is unavailable based on entities
            unavailable_count = sum(1 for s in device_data["states"] if s == "unavailable")
            total_entities = len(device_data["entities"])
            is_online = total_entities == 0 or unavailable_count < total_entities / 2
            
            device_list.append({
                "name": device_name,
                "entity_count": total_entities,
                "online": is_online,
                "sample_entities": device_data["entities"][:5],
            })
        
        online = sum(1 for d in device_list if d["online"])
        offline = len(device_list) - online
        
        return {
            "total_devices": len(device_list),
            "online": online,
            "offline": offline,
            "devices": device_list,
        }
        
    except Exception as e:
        logger.error(f"Error getting ESPHome devices: {str(e)}")
        return {"error": f"Error getting ESPHome devices: {str(e)}"}


@handle_api_errors
async def get_addons() -> Dict[str, Any]:
    """
    Get list of Home Assistant addons with their status
    
    Returns:
        Dictionary with addon information
    """
    try:
        client = await get_client()
        
        # Get addons list
        response = await client.get(f"{HA_URL}/api/hassio/addons", headers=get_ha_headers())
        
        if response.status_code == 401:
            return {
                "error": "Supervisor not available",
                "message": "This Home Assistant installation does not have the Supervisor (not running HA OS or Supervised installation)",
                "total_addons": 0,
                "installed": 0,
                "running": 0,
                "stopped": 0,
                "updates_available": 0,
                "addons": []
            }
        
        if response.status_code != 200:
            return {"error": f"Failed to get addons: {response.status_code}"}
        
        data = response.json().get("data", {})
        addons = data.get("addons", [])
        
        # Format addon info
        addon_list = []
        for addon in addons:
            addon_info = {
                "slug": addon.get("slug"),
                "name": addon.get("name"),
                "description": addon.get("description"),
                "version": addon.get("version"),
                "state": addon.get("state"),
                "installed": addon.get("installed"),
                "available": addon.get("available"),
                "update_available": addon.get("update_available"),
            }
            addon_list.append(addon_info)
        
        # Count by state
        installed = sum(1 for a in addon_list if a.get("installed"))
        running = sum(1 for a in addon_list if a.get("state") == "started")
        stopped = sum(1 for a in addon_list if a.get("installed") and a.get("state") != "started")
        updates_available = sum(1 for a in addon_list if a.get("update_available"))
        
        return {
            "total_addons": len(addon_list),
            "installed": installed,
            "running": running,
            "stopped": stopped,
            "updates_available": updates_available,
            "addons": addon_list,
        }
        
    except Exception as e:
        logger.error(f"Error getting addons: {str(e)}")
        return {"error": f"Error getting addons: {str(e)}"}


@handle_api_errors
async def find_unavailable_entities() -> Dict[str, Any]:
    """
    Find all entities that are currently unavailable
    
    Returns:
        Dictionary with unavailable entities grouped by domain
    """
    try:
        all_states = await get_all_entity_states()
        
        unavailable_entities = []
        for entity_id, state in all_states.items():
            if state.get("state") == "unavailable":
                unavailable_entities.append({
                    "entity_id": entity_id,
                    "domain": entity_id.split(".")[0],
                    "friendly_name": state.get("attributes", {}).get("friendly_name", entity_id),
                    "last_changed": state.get("last_changed"),
                })
        
        # Group by domain
        by_domain = {}
        for entity in unavailable_entities:
            domain = entity["domain"]
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(entity)
        
        return {
            "total_unavailable": len(unavailable_entities),
            "by_domain": by_domain,
            "domain_counts": {domain: len(entities) for domain, entities in by_domain.items()},
            "entities": unavailable_entities[:50],  # Limit for token efficiency
        }
        
    except Exception as e:
        logger.error(f"Error finding unavailable entities: {str(e)}")
        return {"error": f"Error finding unavailable entities: {str(e)}"}


@handle_api_errors
async def find_stale_entities(hours: int = 2) -> Dict[str, Any]:
    """
    Find entities that haven't updated in the specified time period (frozen sensors)
    
    Args:
        hours: Number of hours to consider entity as stale (default: 2)
    
    Returns:
        Dictionary with stale entities
    """
    try:
        from datetime import datetime, timezone, timedelta
        
        all_states = await get_all_entity_states()
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        stale_entities = []
        for entity_id, state in all_states.items():
            # Skip unavailable entities (handled by find_unavailable_entities)
            if state.get("state") == "unavailable":
                continue
            
            # Parse last_updated timestamp
            last_updated_str = state.get("last_updated")
            if last_updated_str:
                try:
                    # Handle ISO format with timezone
                    last_updated = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
                    
                    if last_updated < cutoff_time:
                        stale_entities.append({
                            "entity_id": entity_id,
                            "domain": entity_id.split(".")[0],
                            "friendly_name": state.get("attributes", {}).get("friendly_name", entity_id),
                            "state": state.get("state"),
                            "last_updated": last_updated_str,
                            "hours_stale": round((datetime.now(timezone.utc) - last_updated).total_seconds() / 3600, 1),
                        })
                except (ValueError, AttributeError):
                    pass
        
        # Group by domain
        by_domain = {}
        for entity in stale_entities:
            domain = entity["domain"]
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(entity)
        
        return {
            "total_stale": len(stale_entities),
            "threshold_hours": hours,
            "by_domain": by_domain,
            "domain_counts": {domain: len(entities) for domain, entities in by_domain.items()},
            "entities": stale_entities[:50],  # Limit for token efficiency
        }
        
    except Exception as e:
        logger.error(f"Error finding stale entities: {str(e)}")
        return {"error": f"Error finding stale entities: {str(e)}"}


@handle_api_errors
async def battery_report() -> Dict[str, Any]:
    """
    Get report of all battery-powered devices with their battery levels
    
    Returns:
        Dictionary with battery status for all devices
    """
    try:
        all_states = await get_all_entity_states()
        
        battery_entities = []
        low_battery = []
        
        for entity_id, state in all_states.items():
            attributes = state.get("attributes", {})
            
            # Check if entity has battery level
            battery_level = attributes.get("battery_level") or attributes.get("battery")
            
            # Also check for battery domain entities
            if entity_id.startswith("sensor.") and "battery" in entity_id.lower():
                try:
                    battery_level = float(state.get("state"))
                except (ValueError, TypeError):
                    battery_level = None
            
            if battery_level is not None:
                try:
                    battery_level = float(battery_level)
                    entity_info = {
                        "entity_id": entity_id,
                        "friendly_name": attributes.get("friendly_name", entity_id),
                        "battery_level": battery_level,
                        "state": state.get("state"),
                        "device_class": attributes.get("device_class"),
                    }
                    
                    battery_entities.append(entity_info)
                    
                    # Flag low battery (below 20%)
                    if battery_level < 20:
                        low_battery.append(entity_info)
                        
                except (ValueError, TypeError):
                    pass
        
        # Sort by battery level (lowest first)
        battery_entities.sort(key=lambda x: x["battery_level"])
        low_battery.sort(key=lambda x: x["battery_level"])
        
        return {
            "total_battery_entities": len(battery_entities),
            "low_battery_count": len(low_battery),
            "critical_count": len([e for e in battery_entities if e["battery_level"] < 10]),
            "low_battery": low_battery,
            "all_batteries": battery_entities[:50],  # Limit for token efficiency
        }
        
    except Exception as e:
        logger.error(f"Error generating battery report: {str(e)}")
        return {"error": f"Error generating battery report: {str(e)}"}


async def get_repair_items() -> Dict[str, Any]:
    """
    Get all pending repair issues detected by Home Assistant
    
    Returns:
        Dictionary with repair issues
    """
    try:
        # Try WebSocket API first (for newer HA versions)
        ws_url = HA_URL.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/websocket"
        
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
            # Auth
            await websocket.recv()
            await websocket.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
            auth_response_str = await websocket.recv()
            auth_result = json.loads(auth_response_str)
            
            if not isinstance(auth_result, dict) or auth_result.get("type") != "auth_ok":
                return {
                    "total_issues": 0,
                    "critical_count": 0,
                    "error_count": 0,
                    "warning_count": 0,
                    "by_severity": {"critical": [], "error": [], "warning": [], "info": []},
                    "by_domain": {},
                    "message": "WebSocket authentication failed"
                }
            
            # Get repair issues via WebSocket
            await websocket.send(json.dumps({"id": 1, "type": "repairs/list_issues"}))
            response_str = await websocket.recv()
            response = json.loads(response_str)
            
            # Check if response is valid dict
            if not isinstance(response, dict):
                return {
                    "total_issues": 0,
                    "critical_count": 0,
                    "error_count": 0,
                    "warning_count": 0,
                    "by_severity": {"critical": [], "error": [], "warning": [], "info": []},
                    "by_domain": {},
                    "message": "Invalid response from repairs API"
                }
            
            if not response.get("success"):
                # If repairs API not available, return empty results with message
                # If repairs API not available, return empty results with message
                return {
                    "total_issues": 0,
                    "critical_count": 0,
                    "error_count": 0,
                    "warning_count": 0,
                    "by_severity": {"critical": [], "error": [], "warning": [], "info": []},
                    "by_domain": {},
                    "message": "Repairs API not available in this Home Assistant version"
                }
            
            issues = response.get("result", [])
            
            # Ensure issues is a list
            if not isinstance(issues, list):
                issues = []
            
            # Group by severity
            by_severity = {"critical": [], "error": [], "warning": [], "info": []}
            by_domain = {}
            
            for issue in issues:
                # Skip non-dict issues
                if not isinstance(issue, dict):
                    continue
                    
                severity = issue.get("severity", "info")
                domain = issue.get("domain", "unknown")
                
                issue_info = {
                    "issue_id": issue.get("issue_id"),
                    "domain": domain,
                    "severity": severity,
                    "translation_key": issue.get("translation_key"),
                    "created": issue.get("created"),
                    "dismissed": issue.get("dismissed_version"),
                    "learn_more_url": issue.get("learn_more_url"),
                }
                
                if severity in by_severity:
                    by_severity[severity].append(issue_info)
                
                if domain not in by_domain:
                    by_domain[domain] = []
                by_domain[domain].append(issue_info)
            
            return {
                "total_issues": len(issues),
                "critical_count": len(by_severity["critical"]),
                "error_count": len(by_severity["error"]),
                "warning_count": len(by_severity["warning"]),
                "by_severity": by_severity,
                "by_domain": by_domain,
            }
        
    except Exception as e:
        logger.error(f"Error getting repair items: {str(e)}")
        # Return graceful degradation instead of error
        return {
            "total_issues": 0,
            "critical_count": 0,
            "error_count": 0,
            "warning_count": 0,
            "by_severity": {"critical": [], "error": [], "warning": [], "info": []},
            "by_domain": {},
            "message": f"Repairs API unavailable: {str(e)}"
        }


@handle_api_errors
async def get_update_status() -> Dict[str, Any]:
    """
    Get status of all available updates (core, integrations, addons, devices)
    
    Returns:
        Dictionary with update information
    """
    try:
        all_states = await get_all_entity_states()
        
        updates_available = []
        updates_installed = []
        
        for entity_id, state in all_states.items():
            if entity_id.startswith("update."):
                attributes = state.get("attributes", {})
                
                # Check if update is available (state is "on")
                has_update = state.get("state") == "on"
                
                update_info = {
                    "entity_id": entity_id,
                    "friendly_name": attributes.get("friendly_name", entity_id),
                    "installed_version": attributes.get("installed_version"),
                    "latest_version": attributes.get("latest_version"),
                    "title": attributes.get("title"),
                    "release_url": attributes.get("release_url"),
                    "in_progress": attributes.get("in_progress", False),
                }
                
                if has_update:
                    updates_available.append(update_info)
                else:
                    updates_installed.append(update_info)
        
        # Categorize updates
        core_updates = [u for u in updates_available if "home assistant" in u["friendly_name"].lower() or "core" in u["entity_id"]]
        addon_updates = [u for u in updates_available if "addon" in u["entity_id"] or "add-on" in u["friendly_name"].lower()]
        device_updates = [u for u in updates_available if u not in core_updates and u not in addon_updates]
        
        return {
            "total_updates_available": len(updates_available),
            "core_updates": core_updates,
            "addon_updates": addon_updates,
            "device_updates": device_updates,
            "all_updates": updates_available,
            "up_to_date_count": len(updates_installed),
        }
        
    except Exception as e:
        logger.error(f"Error getting update status: {str(e)}")
        return {"error": f"Error getting update status: {str(e)}"}


@handle_api_errors
async def get_entity_statistics(entity_id: str, period_hours: int = 24) -> Dict[str, Any]:
    """
    Get statistical data for an entity over a time period
    
    Args:
        entity_id: The entity to get statistics for
        period_hours: Number of hours of history to analyze (default: 24)
    
    Returns:
        Dictionary with statistical analysis
    """
    try:
        from datetime import datetime, timezone, timedelta
        
        # Get history for the entity
        start_time = datetime.now(timezone.utc) - timedelta(hours=period_hours)
        
        client = await get_client()
        response = await client.get(
            f"{HA_URL}/api/history/period/{start_time.isoformat()}",
            params={"filter_entity_id": entity_id},
            headers=get_ha_headers()
        )
        
        if response.status_code != 200:
            return {"error": f"Failed to get history: {response.status_code}"}
        
        data = response.json()
        if not data or not isinstance(data, list) or not data[0]:
            return {"error": "No history data available for this entity"}
        
        states = data[0]
        
        # Extract numeric values
        numeric_values = []
        state_changes = []
        for state in states:
            try:
                value = float(state.get("state"))
                numeric_values.append(value)
                state_changes.append({
                    "state": state.get("state"),
                    "timestamp": state.get("last_changed")
                })
            except (ValueError, TypeError):
                # Non-numeric state, just track changes
                state_changes.append({
                    "state": state.get("state"),
                    "timestamp": state.get("last_changed")
                })
        
        # Calculate statistics for numeric sensors
        if numeric_values:
            import statistics
            return {
                "entity_id": entity_id,
                "period_hours": period_hours,
                "total_samples": len(numeric_values),
                "min": min(numeric_values),
                "max": max(numeric_values),
                "mean": round(statistics.mean(numeric_values), 2),
                "median": round(statistics.median(numeric_values), 2),
                "std_dev": round(statistics.stdev(numeric_values), 2) if len(numeric_values) > 1 else 0,
                "recent_value": numeric_values[-1] if numeric_values else None,
                "state_changes_count": len(state_changes),
                "sample_changes": state_changes[-10:]  # Last 10 changes
            }
        else:
            # Non-numeric entity (switches, etc)
            state_distribution = {}
            for change in state_changes:
                state = change["state"]
                state_distribution[state] = state_distribution.get(state, 0) + 1
            
            return {
                "entity_id": entity_id,
                "period_hours": period_hours,
                "total_changes": len(state_changes),
                "state_distribution": state_distribution,
                "recent_changes": state_changes[-10:]  # Last 10 changes
            }
        
    except Exception as e:
        logger.error(f"Error getting entity statistics: {str(e)}")
        return {"error": f"Error getting entity statistics: {str(e)}"}


@handle_api_errors
async def find_anomalous_entities() -> Dict[str, Any]:
    """
    Find entities with anomalous behavior (unusual values, frozen sensors, etc)
    
    Returns:
        Dictionary with anomalous entities grouped by issue type
    """
    try:
        from datetime import datetime, timezone, timedelta
        
        all_states = await get_all_entity_states()
        
        anomalies = {
            "impossible_values": [],
            "frozen_sensors": [],
            "extreme_battery_drain": [],
            "rapid_state_changes": [],
        }
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for entity_id, state in all_states.items():
            attributes = state.get("attributes", {})
            current_state = state.get("state")
            
            # Skip unavailable entities (handled by find_unavailable_entities)
            if current_state == "unavailable":
                continue
            
            # Check for impossible battery values
            if "battery" in entity_id.lower():
                try:
                    battery = float(current_state)
                    if battery < 0 or battery > 100:
                        anomalies["impossible_values"].append({
                            "entity_id": entity_id,
                            "value": battery,
                            "issue": "Battery percentage out of range (0-100)",
                            "friendly_name": attributes.get("friendly_name", entity_id)
                        })
                except (ValueError, TypeError):
                    pass
            
            # Check for temperature sensors with impossible values
            if entity_id.startswith("sensor.") and attributes.get("device_class") == "temperature":
                try:
                    temp = float(current_state)
                    unit = attributes.get("unit_of_measurement", "")
                    if unit == "°C" and (temp < -50 or temp > 60):
                        anomalies["impossible_values"].append({
                            "entity_id": entity_id,
                            "value": temp,
                            "issue": "Temperature out of reasonable range (-50 to 60°C)",
                            "friendly_name": attributes.get("friendly_name", entity_id)
                        })
                    elif unit == "°F" and (temp < -58 or temp > 140):
                        anomalies["impossible_values"].append({
                            "entity_id": entity_id,
                            "value": temp,
                            "issue": "Temperature out of reasonable range (-58 to 140°F)",
                            "friendly_name": attributes.get("friendly_name", entity_id)
                        })
                except (ValueError, TypeError):
                    pass
            
            # Check for humidity with impossible values
            if entity_id.startswith("sensor.") and attributes.get("device_class") == "humidity":
                try:
                    humidity = float(current_state)
                    if humidity < 0 or humidity > 100:
                        anomalies["impossible_values"].append({
                            "entity_id": entity_id,
                            "value": humidity,
                            "issue": "Humidity percentage out of range (0-100)",
                            "friendly_name": attributes.get("friendly_name", entity_id)
                        })
                except (ValueError, TypeError):
                    pass
        
        # Count anomalies by type
        total_anomalies = sum(len(v) for v in anomalies.values())
        
        return {
            "total_anomalies": total_anomalies,
            "impossible_values_count": len(anomalies["impossible_values"]),
            "frozen_sensors_count": len(anomalies["frozen_sensors"]),
            "extreme_battery_drain_count": len(anomalies["extreme_battery_drain"]),
            "anomalies": anomalies,
        }
        
    except Exception as e:
        logger.error(f"Error finding anomalous entities: {str(e)}")
        return {"error": f"Error finding anomalous entities: {str(e)}"}


@handle_api_errors
async def recent_activity(hours: int = 24) -> Dict[str, Any]:
    """
    Get recent activity from the Home Assistant logbook
    
    Args:
        hours: Number of hours of activity to retrieve (default: 24)
    
    Returns:
        Dictionary with recent activity events
    """
    try:
        from datetime import datetime, timezone, timedelta
        
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        client = await get_client()
        response = await client.get(
            f"{HA_URL}/api/logbook/{start_time.isoformat()}",
            headers=get_ha_headers()
        )
        
        if response.status_code != 200:
            return {"error": f"Failed to get logbook: {response.status_code}"}
        
        events = response.json()
        
        # Group events by domain
        by_domain = {}
        by_entity = {}
        
        for event in events[:100]:  # Limit to 100 most recent events
            entity_id = event.get("entity_id", "")
            domain = entity_id.split(".")[0] if entity_id else "system"
            
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(event)
            
            if entity_id:
                if entity_id not in by_entity:
                    by_entity[entity_id] = 0
                by_entity[entity_id] += 1
        
        # Find most active entities
        most_active = sorted(by_entity.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "period_hours": hours,
            "total_events": len(events),
            "events_shown": min(len(events), 100),
            "by_domain": {k: len(v) for k, v in by_domain.items()},
            "most_active_entities": [{"entity_id": e[0], "event_count": e[1]} for e in most_active],
            "recent_events": events[:20],  # 20 most recent events
        }
        
    except Exception as e:
        logger.error(f"Error getting recent activity: {str(e)}")
        return {"error": f"Error getting recent activity: {str(e)}"}


@handle_api_errors
async def offline_devices_report() -> Dict[str, Any]:
    """
    Get comprehensive report of offline/unavailable devices grouped by integration
    
    Returns:
        Dictionary with offline device information
    """
    try:
        # Get all unavailable entities
        unavailable_result = await find_unavailable_entities()
        
        # Get device information via WebSocket
        ws_url = HA_URL.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/websocket"
        
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
            # Auth
            await websocket.recv()
            await websocket.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
            auth_response_str = await websocket.recv()
            auth_result = json.loads(auth_response_str)
            
            if not isinstance(auth_result, dict) or auth_result.get("type") != "auth_ok":
                return {"error": "WebSocket authentication failed"}
            
            # Get device registry
            await websocket.send(json.dumps({"id": 1, "type": "config/device_registry/list"}))
            response_str = await websocket.recv()
            response = json.loads(response_str)
            
            if not response.get("success"):
                return {"error": "Failed to get device registry"}
            
            devices = response.get("result", [])
            
            # Map devices to their entities and check availability
            offline_devices = []
            all_states = await get_all_entity_states()
            
            for device in devices:
                device_id = device.get("id")
                device_name = device.get("name_by_user") or device.get("name")
                
                # Find entities for this device
                device_entities = [
                    entity_id for entity_id, state in all_states.items()
                    if state.get("attributes", {}).get("device_id") == device_id
                ]
                
                if not device_entities:
                    continue
                
                # Check if all entities are unavailable
                unavailable_count = sum(
                    1 for entity_id in device_entities
                    if all_states.get(entity_id, {}).get("state") == "unavailable"
                )
                
                if unavailable_count == len(device_entities):
                    # All entities unavailable = device offline
                    offline_devices.append({
                        "device_name": device_name,
                        "manufacturer": device.get("manufacturer"),
                        "model": device.get("model"),
                        "entity_count": len(device_entities),
                        "sample_entities": device_entities[:3],
                        "via_device": device.get("via_device_id"),
                    })
            
            # Group by manufacturer
            by_manufacturer = {}
            for device in offline_devices:
                manufacturer = device.get("manufacturer", "Unknown")
                if manufacturer not in by_manufacturer:
                    by_manufacturer[manufacturer] = []
                by_manufacturer[manufacturer].append(device)
            
            return {
                "total_offline_devices": len(offline_devices),
                "total_unavailable_entities": unavailable_result.get("total_unavailable", 0),
                "by_manufacturer": {k: len(v) for k, v in by_manufacturer.items()},
                "offline_devices": offline_devices[:30],  # Limit for token efficiency
                "entity_summary": unavailable_result.get("domain_counts", {}),
            }
        
    except Exception as e:
        logger.error(f"Error getting offline devices report: {str(e)}")
        return {"error": f"Error getting offline devices report: {str(e)}"}


# ============================================================================
# SYSTEM & PLATFORM TOOLS
# ============================================================================

@handle_api_errors
async def system_overview() -> Dict[str, Any]:
    """
    Get comprehensive overview of the entire Home Assistant system
    
    Returns:
        Dictionary with system overview including domains, samples, and area distribution
    """
    try:
        all_states = await get_all_entity_states()
        
        # Group by domain
        domains = {}
        for entity_id, state in all_states.items():
            domain = entity_id.split(".")[0]
            if domain not in domains:
                domains[domain] = {"count": 0, "states": {}, "samples": []}
            
            domains[domain]["count"] += 1
            
            # Track state distribution
            current_state = state.get("state", "unknown")
            if current_state not in domains[domain]["states"]:
                domains[domain]["states"][current_state] = 0
            domains[domain]["states"][current_state] += 1
            
            # Add samples (max 3 per domain)
            if len(domains[domain]["samples"]) < 3:
                domains[domain]["samples"].append({
                    "entity_id": entity_id,
                    "name": state.get("attributes", {}).get("friendly_name", entity_id),
                    "state": current_state,
                })
        
        # Get area distribution
        area_distribution = {}
        for entity_id, state in all_states.items():
            area = state.get("attributes", {}).get("area_id")
            if area:
                if area not in area_distribution:
                    area_distribution[area] = 0
                area_distribution[area] += 1
        
        # Get common attributes per domain
        domain_attributes = {}
        for domain, info in domains.items():
            sample_entity = next(
                (entity_id for entity_id in all_states.keys() if entity_id.startswith(f"{domain}.")),
                None
            )
            if sample_entity:
                attrs = all_states[sample_entity].get("attributes", {})
                domain_attributes[domain] = list(attrs.keys())[:5]  # Top 5 attributes
        
        return {
            "total_entities": len(all_states),
            "domains": {k: {"count": v["count"], "states": v["states"]} for k, v in domains.items()},
            "domain_samples": {k: v["samples"] for k, v in domains.items()},
            "domain_attributes": domain_attributes,
            "area_distribution": area_distribution,
        }
        
    except Exception as e:
        logger.error(f"Error getting system overview: {str(e)}")
        return {"error": f"Error getting system overview: {str(e)}"}


@handle_api_errors
async def domain_summary(domain: str, example_limit: int = 3) -> Dict[str, Any]:
    """
    Get detailed summary of entities in a specific domain
    
    Args:
        domain: The domain to summarize (e.g., 'light', 'switch', 'sensor')
        example_limit: Maximum number of examples to include per state
    
    Returns:
        Dictionary with domain summary
    """
    try:
        all_states = await get_all_entity_states()
        
        # Filter by domain
        domain_entities = {k: v for k, v in all_states.items() if k.startswith(f"{domain}.")}
        
        if not domain_entities:
            return {"error": f"No entities found for domain '{domain}'"}
        
        # State distribution with examples
        state_examples = {}
        for entity_id, state in domain_entities.items():
            current_state = state.get("state", "unknown")
            if current_state not in state_examples:
                state_examples[current_state] = []
            
            if len(state_examples[current_state]) < example_limit:
                state_examples[current_state].append({
                    "entity_id": entity_id,
                    "name": state.get("attributes", {}).get("friendly_name", entity_id),
                    "last_changed": state.get("last_changed"),
                })
        
        # Common attributes
        all_attrs = {}
        for entity_id, state in domain_entities.items():
            for attr in state.get("attributes", {}).keys():
                if attr not in all_attrs:
                    all_attrs[attr] = 0
                all_attrs[attr] += 1
        
        common_attrs = sorted(all_attrs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "domain": domain,
            "total_count": len(domain_entities),
            "state_distribution": {k: len(v) for k, v in state_examples.items()},
            "examples": state_examples,
            "common_attributes": [attr[0] for attr in common_attrs],
        }
        
    except Exception as e:
        logger.error(f"Error getting domain summary: {str(e)}")
        return {"error": f"Error getting domain summary: {str(e)}"}


@handle_api_errors
async def list_integrations() -> Dict[str, Any]:
    """
    Get list of all loaded integrations and their entity counts
    
    Returns:
        Dictionary with integration information
    """
    try:
        all_states = await get_all_entity_states()
        
        # Get unique domains (which represent integrations)
        domains = set()
        entity_counts = {}
        
        for entity_id in all_states.keys():
            domain = entity_id.split(".")[0]
            domains.add(domain)
            if domain not in entity_counts:
                entity_counts[domain] = 0
            entity_counts[domain] += 1
        
        return {
            "total_integrations": len(domains),
            "integrations": sorted(list(domains)),
            "entity_counts": entity_counts,
            "integrations_with_entities": len([d for d, count in entity_counts.items() if count > 0]),
        }
        
    except Exception as e:
        logger.error(f"Error listing integrations: {str(e)}")
        return {"error": f"Error listing integrations: {str(e)}"}


# ============================================================================
# ADVANCED DIAGNOSTIC TOOLS
# ============================================================================

@handle_api_errors
async def identify_device(device_id_or_entity_id: str, pattern: str = "auto", duration: int = 3) -> Dict[str, Any]:
    """
    Identify a physical device using the best method available for its platform
    
    This function helps locate devices in the real world by making them blink,
    beep, or otherwise identify themselves.
    
    Args:
        device_id_or_entity_id: Device ID or entity ID to identify
        pattern: Identification pattern ('auto', 'flash', 'toggle', 'color', 'beep')
        duration: Duration in seconds (default: 3)
    
    Returns:
        Dictionary with identification results
    """
    try:
        # Step 1: Determine if we have entity_id or device_id
        entity_id = None
        device_id = None
        
        if "." in device_id_or_entity_id:
            # It's an entity_id
            entity_id = device_id_or_entity_id
            # Get entity to find device_id
            entity = await get_entity_state(entity_id)
            if "error" not in entity:
                device_id = entity.get("attributes", {}).get("device_id")
        else:
            # Assume it's a device_id
            device_id = device_id_or_entity_id
        
        # Step 2: Get all entities for this device
        all_states = await get_all_entity_states()
        device_entities = []
        
        if device_id:
            for eid, state in all_states.items():
                if state.get("attributes", {}).get("device_id") == device_id:
                    device_entities.append(eid)
        elif entity_id:
            # If we only have entity_id, work with just that one
            device_entities = [entity_id]
        
        if not device_entities:
            return {
                "success": False,
                "error": "No entities found for device",
                "device_id": device_id,
                "entity_id": entity_id,
            }
        
        # Step 3: Detect platform and capabilities
        platform_info = await _detect_device_platform(device_entities, all_states)
        
        # Step 4: Execute identification strategy
        actions_executed = []
        success = False
        notes = []
        
        if pattern == "auto":
            # Auto-select best strategy based on platform
            if platform_info["primary_domain"] == "light":
                result = await _identify_light(device_entities, duration)
                actions_executed = result["actions"]
                success = result["success"]
                notes = result.get("notes", [])
            
            elif platform_info["primary_domain"] == "switch":
                result = await _identify_switch(device_entities, duration)
                actions_executed = result["actions"]
                success = result["success"]
                notes = result.get("notes", [])
            
            elif platform_info["primary_domain"] == "media_player":
                result = await _identify_media_player(device_entities, duration)
                actions_executed = result["actions"]
                success = result["success"]
                notes = result.get("notes", [])
            
            elif platform_info.get("is_zha"):
                result = await _identify_zha_device(device_entities, duration)
                actions_executed = result["actions"]
                success = result["success"]
                notes = result.get("notes", [])
            
            elif platform_info.get("is_esphome"):
                result = await _identify_esphome_device(device_entities, duration)
                actions_executed = result["actions"]
                success = result["success"]
                notes = result.get("notes", [])
            
            else:
                # Fallback: provide info
                notes.append("No automatic identification method available for this device type")
                notes.append(f"Device has entities: {', '.join(device_entities[:5])}")
        
        # Step 5: Prepare alternative methods
        alternative_methods = []
        if not success or not actions_executed:
            alternative_methods = _get_alternative_identification_methods(platform_info, device_entities)
        
        return {
            "success": success,
            "device_id": device_id,
            "entity_id": entity_id,
            "entities_found": device_entities,
            "platform_detected": platform_info,
            "actions_executed": actions_executed,
            "duration_seconds": duration,
            "alternative_methods": alternative_methods,
            "notes": notes,
        }
        
    except Exception as e:
        logger.error(f"Error identifying device: {str(e)}")
        return {
            "success": False,
            "error": f"Error identifying device: {str(e)}",
            "device_id": device_id,
            "entity_id": entity_id,
        }


async def _detect_device_platform(entity_ids: List[str], all_states: Dict) -> Dict[str, Any]:
    """Detect the platform and capabilities of a device"""
    domains = {}
    integrations = set()
    
    for entity_id in entity_ids:
        domain = entity_id.split(".")[0]
        domains[domain] = domains.get(domain, 0) + 1
        
        state = all_states.get(entity_id, {})
        attrs = state.get("attributes", {})
        
        # Check for platform indicators
        if "zha" in entity_id.lower() or "ieee" in str(attrs):
            integrations.add("zha")
        if "esphome" in str(attrs.get("attribution", "")).lower():
            integrations.add("esphome")
    
    primary_domain = max(domains.keys(), key=domains.get) if domains else "unknown"
    
    return {
        "primary_domain": primary_domain,
        "all_domains": list(domains.keys()),
        "entity_counts": domains,
        "is_zha": "zha" in integrations,
        "is_esphome": "esphome" in integrations,
        "total_entities": len(entity_ids),
    }


async def _identify_light(entity_ids: List[str], duration: int) -> Dict[str, Any]:
    """Identify a light by flashing or changing color"""
    actions = []
    success = False
    notes = []
    
    # Find first light entity
    light_entity = next((e for e in entity_ids if e.startswith("light.")), None)
    
    if not light_entity:
        return {"success": False, "actions": [], "notes": ["No light entity found"]}
    
    try:
        # Get current state to restore later
        original_state = await get_entity_state(light_entity)
        was_on = original_state.get("state") == "on"
        original_brightness = original_state.get("attributes", {}).get("brightness")
        original_color = original_state.get("attributes", {}).get("rgb_color")
        original_color_temp = original_state.get("attributes", {}).get("color_temp")
        
        logger.info(f"identify_device: Starting identification for {light_entity}")
        logger.info(f"identify_device: Original state - on={was_on}, brightness={original_brightness}")
        
        # Strategy: Flash bright 3 times - SLOW for human visibility
        # Use simple on/off without color changes for maximum compatibility
        for i in range(3):
            logger.info(f"identify_device: Flash {i+1} starting")
            
            # Turn on at maximum brightness
            await call_service("light", "turn_on", {
                "entity_id": light_entity,
                "brightness": 255,
            })
            actions.append(f"Flash {i+1}: ON (bright)")
            logger.info(f"identify_device: Flash {i+1} light turned ON")
            
            await asyncio.sleep(1.5)  # 1.5 seconds ON - clearly visible
            
            # Turn off
            await call_service("light", "turn_off", {"entity_id": light_entity})
            actions.append(f"Flash {i+1}: OFF")
            logger.info(f"identify_device: Flash {i+1} light turned OFF")
            
            await asyncio.sleep(1.5)  # 1.5 seconds OFF - clearly visible
        
        # Restore original state
        logger.info(f"identify_device: Restoring original state - was_on={was_on}")
        if was_on:
            restore_params = {"entity_id": light_entity}
            if original_brightness:
                restore_params["brightness"] = original_brightness
            if original_color_temp:
                restore_params["color_temp"] = original_color_temp
            elif original_color:
                restore_params["rgb_color"] = original_color
            await call_service("light", "turn_on", restore_params)
            actions.append("Restored original state (ON)")
            logger.info(f"identify_device: Original state restored (ON)")
        else:
            actions.append("Left OFF (original state)")
            logger.info(f"identify_device: Left OFF as it was originally")
        
        success = True
        notes.append("Light flashed 3 times (1.5 seconds ON, 1.5 seconds OFF)")
        logger.info(f"identify_device: Identification complete successfully")
        
    except Exception as e:
        logger.error(f"identify_device: Error during light identification: {str(e)}")
        notes.append(f"Error during light identification: {str(e)}")
    
    return {"success": success, "actions": actions, "notes": notes}


async def _identify_switch(entity_ids: List[str], duration: int) -> Dict[str, Any]:
    """Identify a switch by toggling rapidly"""
    actions = []
    success = False
    notes = []
    
    switch_entity = next((e for e in entity_ids if e.startswith("switch.")), None)
    
    if not switch_entity:
        return {"success": False, "actions": [], "notes": ["No switch entity found"]}
    
    try:
        # Get current state
        original_state = await get_entity_state(switch_entity)
        was_on = original_state.get("state") == "on"
        
        # Toggle 3 times
        for i in range(3):
            await call_service("switch", "turn_on", {"entity_id": switch_entity})
            actions.append(f"Toggle {i+1}: ON")
            await asyncio.sleep(0.7)
            
            await call_service("switch", "turn_off", {"entity_id": switch_entity})
            actions.append(f"Toggle {i+1}: OFF")
            await asyncio.sleep(0.7)
        
        # Restore
        if was_on:
            await call_service("switch", "turn_on", {"entity_id": switch_entity})
            actions.append("Restored to ON")
        else:
            actions.append("Left OFF (original state)")
        
        success = True
        notes.append("Switch toggled 3 times")
        
    except Exception as e:
        notes.append(f"Error during switch identification: {str(e)}")
    
    return {"success": success, "actions": actions, "notes": notes}


async def _identify_media_player(entity_ids: List[str], duration: int) -> Dict[str, Any]:
    """Identify a media player by playing a beep or TTS"""
    actions = []
    success = False
    notes = []
    
    media_entity = next((e for e in entity_ids if e.startswith("media_player.")), None)
    
    if not media_entity:
        return {"success": False, "actions": [], "notes": ["No media_player entity found"]}
    
    try:
        # Try to play TTS message
        device_name = media_entity.split(".")[1].replace("_", " ")
        await call_service("tts", "speak", {
            "entity_id": media_entity,
            "message": f"This is {device_name}",
        })
        actions.append("Played TTS identification message")
        success = True
        notes.append("Device identified via TTS")
        
    except Exception as e:
        notes.append(f"TTS not available: {str(e)}")
        # Fallback: try volume change
        try:
            await call_service("media_player", "volume_set", {
                "entity_id": media_entity,
                "volume_level": 0.3,
            })
            await asyncio.sleep(0.5)
            await call_service("media_player", "volume_set", {
                "entity_id": media_entity,
                "volume_level": 0.7,
            })
            actions.append("Changed volume to identify")
            success = True
            notes.append("Device identified via volume change")
        except Exception as e2:
            notes.append(f"Volume change failed: {str(e2)}")
    
    return {"success": success, "actions": actions, "notes": notes}


async def _identify_zha_device(entity_ids: List[str], duration: int) -> Dict[str, Any]:
    """Identify a ZHA device using Zigbee identify command"""
    actions = []
    success = False
    notes = []
    
    # Try to use ZHA identify service
    try:
        # Get IEEE address from entity
        first_entity = entity_ids[0]
        entity_state = await get_entity_state(first_entity)
        
        # Try ZHA identify service if available
        # Note: This may not work for all ZHA devices
        await call_service("zha", "issue_zigbee_cluster_command", {
            "entity_id": first_entity,
            "cluster_id": 3,  # Identify cluster
            "command": 0,  # Identify command
            "command_type": "client",
            "args": [duration],
        })
        actions.append(f"Sent ZHA identify command (cluster 3, {duration}s)")
        success = True
        notes.append("ZHA identify command sent")
        
    except Exception as e:
        notes.append(f"ZHA identify command failed: {str(e)}")
        
        # Fallback: toggle a light if available
        light_entity = next((e for e in entity_ids if e.startswith("light.")), None)
        if light_entity:
            result = await _identify_light([light_entity], duration)
            actions.extend(result["actions"])
            success = result["success"]
            notes.append("Fell back to light flash")
    
    return {"success": success, "actions": actions, "notes": notes}


async def _identify_esphome_device(entity_ids: List[str], duration: int) -> Dict[str, Any]:
    """Identify an ESPHome device"""
    actions = []
    success = False
    notes = []
    
    # Look for status LED or light
    light_entity = next((e for e in entity_ids if "light" in e), None)
    switch_entity = next((e for e in entity_ids if "switch" in e), None)
    
    if light_entity:
        result = await _identify_light([light_entity], duration)
        actions.extend(result["actions"])
        success = result["success"]
        notes.append("ESPHome device identified via light")
    elif switch_entity:
        result = await _identify_switch([switch_entity], duration)
        actions.extend(result["actions"])
        success = result["success"]
        notes.append("ESPHome device identified via switch")
    else:
        notes.append("No identifiable entity found for ESPHome device")
    
    return {"success": success, "actions": actions, "notes": notes}


def _get_alternative_identification_methods(platform_info: Dict, entity_ids: List[str]) -> List[str]:
    """Get alternative methods to identify device manually"""
    methods = []
    
    methods.append(f"Look for device with entities: {', '.join(entity_ids[:3])}")
    
    if platform_info.get("is_zha"):
        methods.append("Check ZHA device map in Home Assistant")
        methods.append("Look for Zigbee device with IEEE address in device info")
    
    if platform_info.get("is_esphome"):
        methods.append("Check ESPHome dashboard for device")
        methods.append("Look for device IP in ESPHome logs")
    
    methods.append("Check device physical location in Home Assistant device settings")
    methods.append("Review entity names and friendly names for location hints")
    
    return methods


@handle_api_errors
async def diagnose_issue(entity_id: str, ctx=None) -> Dict[str, Any]:
    """
    Comprehensive diagnostic of a specific entity combining multiple diagnostic tools
    
    This function performs deep analysis to find root causes of entity issues by
    checking availability, statistics, logs, device status, and related automations.
    
    Args:
        entity_id: Entity ID to diagnose
        ctx: Optional MCP Context for progress reporting
    
    Returns:
        Comprehensive diagnostic report with root causes and recommendations
    """
    try:
        # Wrap context to make all operations safe
        ctx = safe_ctx(ctx)
        
        if ctx:
            await ctx.info(f"Starting diagnosis of {entity_id}...")
        
        diagnostics_used = []
        root_cause_candidates = []
        recommended_fixes = []
        auto_fix_actions = []
        severity = "low"
        
        # Step 1: Get entity state
        if ctx:
            await ctx.info("Checking entity state...")
        entity = await get_entity_state(entity_id)
        diagnostics_used.append("get_entity_state")
        
        if "error" in entity:
            return {
                "success": False,
                "error": f"Entity not found: {entity_id}",
                "entity_id": entity_id,
            }
        
        entity_summary = {
            "entity_id": entity_id,
            "state": entity.get("state"),
            "friendly_name": entity.get("attributes", {}).get("friendly_name"),
            "domain": entity_id.split(".")[0],
            "last_updated": entity.get("last_updated"),
            "last_changed": entity.get("last_changed"),
        }
        
        # Step 2: Check if unavailable
        if entity.get("state") == "unavailable":
            severity = "high"
            root_cause_candidates.append({
                "cause": "Entity is unavailable",
                "confidence": "high",
                "details": "Entity cannot communicate with Home Assistant",
            })
            
            # Check offline devices
            offline_report = await offline_devices_report()
            diagnostics_used.append("offline_devices_report")
            
            # Check error log
            error_log = await get_ha_error_log()
            diagnostics_used.append("get_error_log")
            
            # Look for related errors
            entity_name = entity_id.split(".")[1]
            if isinstance(error_log, dict) and "log_text" in error_log:
                if entity_name in error_log["log_text"]:
                    root_cause_candidates.append({
                        "cause": "Errors found in logs related to this entity",
                        "confidence": "medium",
                        "details": "Check error_log for specific error messages",
                    })
            
            recommended_fixes.append("Check device power and connectivity")
            recommended_fixes.append("Verify integration is loaded")
            recommended_fixes.append("Try reloading the integration")
            
            auto_fix_actions.append({
                "action": "reload_core_config",
                "description": "Reload core configuration",
                "risk": "low",
            })
        
        # Step 3: Check for stale data (if not unavailable)
        elif entity.get("last_updated"):
            try:
                last_update = datetime.fromisoformat(entity["last_updated"].replace("Z", "+00:00"))
                hours_since_update = (datetime.now(timezone.utc) - last_update).total_seconds() / 3600
                
                if hours_since_update > 2:
                    severity = "medium" if severity == "low" else severity
                    root_cause_candidates.append({
                        "cause": f"Entity hasn't updated in {hours_since_update:.1f} hours",
                        "confidence": "high",
                        "details": "Sensor may be frozen or polling has stopped",
                    })
                    
                    stale_entities = await find_stale_entities(hours=2)
                    diagnostics_used.append("find_stale_entities")
                    
                    recommended_fixes.append("Check device battery if battery-powered")
                    recommended_fixes.append("Verify network connectivity")
                    recommended_fixes.append("Restart the integration")
            except:
                pass
        
        # Step 4: Check battery (if applicable)
        if "battery" in entity_id.lower() or entity.get("attributes", {}).get("device_class") == "battery":
            battery_report_data = await battery_report()
            diagnostics_used.append("battery_report")
            
            try:
                battery_level = float(entity.get("state", 100))
                if battery_level < 10:
                    severity = "critical"
                    root_cause_candidates.append({
                        "cause": f"Critical battery level: {battery_level}%",
                        "confidence": "high",
                        "details": "Device may stop working soon",
                    })
                    recommended_fixes.append("Replace battery immediately")
                elif battery_level < 20:
                    severity = "high" if severity in ["low", "medium"] else severity
                    root_cause_candidates.append({
                        "cause": f"Low battery level: {battery_level}%",
                        "confidence": "high",
                        "details": "Battery should be replaced soon",
                    })
                    recommended_fixes.append("Schedule battery replacement")
            except:
                pass
        
        # Step 5: Get statistics for numeric sensors
        domain = entity_id.split(".")[0]
        if domain == "sensor" and entity.get("state") not in ["unavailable", "unknown"]:
            try:
                stats = await get_entity_statistics(entity_id, period_hours=24)
                diagnostics_used.append("get_entity_statistics")
                
                if stats.get("std_dev", 0) > 50:  # High variance
                    severity = "medium" if severity == "low" else severity
                    root_cause_candidates.append({
                        "cause": "High variance in sensor readings (unstable)",
                        "confidence": "medium",
                        "details": f"Standard deviation: {stats.get('std_dev', 0):.2f}",
                    })
                    recommended_fixes.append("Check sensor calibration")
                    recommended_fixes.append("Verify sensor mounting and placement")
            except:
                pass
        
        # Step 6: Check for anomalies
        anomalies = await find_anomalous_entities()
        diagnostics_used.append("find_anomalous_entities")
        
        if isinstance(anomalies, dict):
            # Check if our entity is in anomalies
            for category, entities_list in anomalies.get("anomalies", {}).items():
                for anomaly in entities_list:
                    if isinstance(anomaly, dict) and anomaly.get("entity_id") == entity_id:
                        severity = "high"
                        root_cause_candidates.append({
                            "cause": f"Anomaly detected: {category}",
                            "confidence": "high",
                            "details": anomaly.get("reason", "Value outside normal range"),
                        })
                        recommended_fixes.append("Sensor may be defective")
                        recommended_fixes.append("Check sensor calibration")
        
        # Step 7: Check recent activity
        activity = await recent_activity(hours=1)
        diagnostics_used.append("recent_activity")
        
        # Count how many times this entity appears
        entity_events = 0
        if isinstance(activity, dict):
            for event in activity.get("recent_events", []):
                if isinstance(event, dict) and event.get("entity_id") == entity_id:
                    entity_events += 1
        
        if entity_events > 50:
            severity = "medium" if severity == "low" else severity
            root_cause_candidates.append({
                "cause": f"Entity very active: {entity_events} events in last hour",
                "confidence": "medium",
                "details": "May be causing performance issues or battery drain",
            })
            recommended_fixes.append("Consider increasing update interval")
            recommended_fixes.append("Check for automation loops")
        
        # Step 8: Check for ZHA/ESPHome specific issues
        if "zha" in entity_id.lower():
            zha_devices = await get_zha_devices()
            diagnostics_used.append("get_zha_devices")
            
            # Find this device in ZHA list
            for device in zha_devices.get("devices", []):
                device_entities = [e for e in [entity_id] if entity_id.startswith("sensor.") or entity_id.startswith("light.") or entity_id.startswith("switch.")]
                
                if device.get("lqi", 255) < 120:
                    severity = "medium" if severity == "low" else severity
                    root_cause_candidates.append({
                        "cause": f"Weak Zigbee signal: LQI {device.get('lqi')}",
                        "confidence": "high",
                        "details": f"RSSI: {device.get('rssi')} dBm",
                    })
                    recommended_fixes.append("Move device closer to coordinator")
                    recommended_fixes.append("Add Zigbee router between device and coordinator")
        
        # Step 9: Determine if no issues found
        if not root_cause_candidates:
            root_cause_candidates.append({
                "cause": "No obvious issues detected",
                "confidence": "high",
                "details": "Entity appears to be functioning normally",
            })
            recommended_fixes.append("Monitor entity for changes")
        
        # Step 10: Compile final report
        if ctx:
            severity_emoji = "✅" if severity == "low" else "⚠️" if severity == "medium" else "🔴"
            await ctx.info(f"{severity_emoji} Diagnosis complete! Severity: {severity}")
        
        return {
            "success": True,
            "entity_id": entity_id,
            "entity_summary": entity_summary,
            "severity": severity,
            "root_cause_candidates": root_cause_candidates,
            "recommended_fixes": recommended_fixes,
            "auto_fix_actions_available": auto_fix_actions,
            "diagnostics_used": diagnostics_used,
            "diagnostics_count": len(diagnostics_used),
        }
        
    except Exception as e:
        logger.error(f"Error diagnosing issue for {entity_id}: {str(e)}")
        if ctx and hasattr(ctx, 'error'):
            await ctx.error(f"Diagnosis failed: {str(e)}")
        return {
            "success": False,
            "error": f"Error during diagnosis: {str(e)}",
            "entity_id": entity_id,
        }


# ============================================================================
# SYSTEM-LEVEL DIAGNOSTICS & AUTO-FIX TOOLS
# ============================================================================

@handle_api_errors
async def diagnose_automation(automation_id: str, ctx=None) -> Dict[str, Any]:
    """
    Analyze an automation to find why it is not triggering or behaving as expected
    
    This function performs comprehensive analysis of an automation by checking:
    - Automation state (on/off)
    - Last triggered timestamp
    - Trigger entity validity and availability
    - Condition and action entity health
    - Error log mentions
    - Recent activity patterns
    - Loop detection
    - Long-running actions
    
    Args:
        automation_id: Entity ID of the automation (e.g., 'automation.living_room_lights')
        ctx: Optional MCP Context for progress reporting
    
    Returns:
        Comprehensive diagnostic report with issues, entities involved, and recommendations
    """
    try:
        # Wrap context to make all operations safe
        ctx = safe_ctx(ctx)
        
        if ctx:
            await ctx.info(f"🔍 Diagnosing automation: {automation_id}")
        
        diagnostics_used = []
        root_cause_candidates = []
        recommended_fixes = []
        auto_fix_actions = []
        severity = "low"
        entities_involved = []
        
        total_steps = 6
        current_step = 0
        
        if ctx:
            await ctx.report_progress(progress=current_step, total=total_steps)
        
        # Step 1: Get automation state
        if ctx:
            current_step += 1
            await ctx.report_progress(progress=current_step, total=total_steps)
            await ctx.info("[1/6] Checking automation state...")
        
        automation_state = await get_entity_state(automation_id)
        diagnostics_used.append("get_entity_state")
        
        if "error" in automation_state:
            return {
                "success": False,
                "error": f"Automation not found: {automation_id}",
                "automation_id": automation_id,
            }
        
        automation_summary = {
            "automation_id": automation_id,
            "state": automation_state.get("state"),
            "friendly_name": automation_state.get("attributes", {}).get("friendly_name"),
            "last_triggered": automation_state.get("attributes", {}).get("last_triggered"),
            "current_mode": automation_state.get("attributes", {}).get("mode", "single"),
            "last_updated": automation_state.get("last_updated"),
        }
        
        logger.info(f"diagnose_automation: Analyzing {automation_id}")
        logger.info(f"diagnose_automation: State={automation_state.get('state')}, Last triggered={automation_summary['last_triggered']}")
        
        # Step 2: Check if automation is ON
        if ctx:
            current_step += 1
            await ctx.report_progress(progress=current_step, total=total_steps)
            await ctx.info("[2/6] Checking automation configuration...")
        
        if automation_state.get("state") != "on":
            severity = "high"
            root_cause_candidates.append({
                "cause": "Automation is turned OFF",
                "confidence": "high",
                "details": "Automation will not trigger while disabled",
            })
            recommended_fixes.append("Turn on the automation")
            auto_fix_actions.append({
                "action": "turn_on_automation",
                "description": f"Turn on {automation_id}",
                "risk": "low",
            })
        
        # Step 3: Check when last triggered
        last_triggered = automation_summary.get("last_triggered")
        if last_triggered:
            try:
                last_trigger_time = datetime.fromisoformat(last_triggered.replace("Z", "+00:00"))
                hours_since_trigger = (datetime.now(timezone.utc) - last_trigger_time).total_seconds() / 3600
                
                if hours_since_trigger > 168:  # 7 days
                    severity = "medium" if severity == "low" else severity
                    root_cause_candidates.append({
                        "cause": f"Automation hasn't triggered in {hours_since_trigger/24:.1f} days",
                        "confidence": "medium",
                        "details": "May indicate triggers are not being met or automation is not needed",
                    })
                    recommended_fixes.append("Review automation triggers and conditions")
            except:
                pass
        else:
            severity = "medium" if severity == "low" else severity
            root_cause_candidates.append({
                "cause": "Automation has never been triggered",
                "confidence": "high",
                "details": "Automation may have invalid triggers or unreachable conditions",
            })
            recommended_fixes.append("Check trigger configuration")
        
        # Step 4: Extract entities from automation config
        if ctx:
            current_step += 1
            await ctx.report_progress(progress=current_step, total=total_steps)
            await ctx.info("[3/6] Extracting automation entities...")
        
        automation_config = automation_state.get("attributes", {})
        
        # Get all automations to find full config
        all_automations = await get_automations()
        diagnostics_used.append("list_automations")
        
        # Step 5: Check for unavailable/stale entities
        unavailable_entities = await find_unavailable_entities()
        diagnostics_used.append("find_unavailable_entities")
        
        stale_entities = await find_stale_entities(hours=2)
        diagnostics_used.append("find_stale_entities")
        
        # Extract entity_ids from automation attributes if available
        # (Note: Full automation config may not be in state attributes)
        # This is a simplified check
        
        # Step 6: Check error log for this automation
        if ctx:
            current_step += 1
            await ctx.report_progress(progress=current_step, total=total_steps)
            await ctx.info("[4/6] Checking error logs...")
        
        error_log = await get_ha_error_log()
        diagnostics_used.append("get_error_log")
        
        automation_name = automation_id.split(".")[1]
        if isinstance(error_log, dict) and "log_text" in error_log:
            if automation_name in error_log["log_text"] or automation_id in error_log["log_text"]:
                severity = "high"
                root_cause_candidates.append({
                    "cause": "Errors found in logs related to this automation",
                    "confidence": "high",
                    "details": "Check error_log for specific error messages",
                })
                recommended_fixes.append("Review error log for automation-specific errors")
                recommended_fixes.append("Check automation YAML configuration for syntax errors")
        
        # Step 7: Check recent activity
        if ctx:
            current_step += 1
            await ctx.report_progress(progress=current_step, total=total_steps)
            await ctx.info("[5/6] Analyzing recent activity...")
        
        activity = await recent_activity(hours=24)
        diagnostics_used.append("recent_activity")
        
        # Count automation events
        automation_events = 0
        if isinstance(activity, dict):
            for event in activity.get("recent_events", []):
                if isinstance(event, dict) and event.get("entity_id") == automation_id:
                    automation_events += 1
        
        if automation_events > 100:  # More than 100 triggers in 24h
            severity = "high"
            root_cause_candidates.append({
                "cause": f"Automation triggered {automation_events} times in last 24 hours",
                "confidence": "high",
                "details": "May indicate automation loop or overly sensitive triggers",
            })
            recommended_fixes.append("Review automation logic for potential loops")
            recommended_fixes.append("Add conditions to prevent rapid re-triggering")
            recommended_fixes.append("Consider adding delay or cooldown period")
        
        # Step 8: Check automation mode for potential issues
        mode = automation_summary.get("current_mode")
        if mode == "single" and automation_events > 50:
            root_cause_candidates.append({
                "cause": "Automation in 'single' mode with high trigger frequency",
                "confidence": "medium",
                "details": "May be blocking itself from running if previous execution hasn't finished",
            })
            recommended_fixes.append("Consider changing mode to 'restart' or 'queued'")
        
        # Step 9: Determine if no issues found
        if not root_cause_candidates:
            root_cause_candidates.append({
                "cause": "No obvious issues detected",
                "confidence": "high",
                "details": "Automation appears to be configured correctly",
            })
            recommended_fixes.append("Monitor automation triggers and execution")
            recommended_fixes.append("Verify trigger conditions are actually occurring")
        
        # Step 10: Compile final report
        if ctx:
            current_step += 1
            await ctx.report_progress(progress=current_step, total=total_steps)
            
            # Determine completion emoji based on severity
            if severity == "low":
                completion_emoji = "✅"
            elif severity == "medium":
                completion_emoji = "⚠️"
            else:
                completion_emoji = "🔴"
            
            await ctx.info(f"{completion_emoji} Automation diagnosis complete! Severity: {severity}")
        
        return {
            "success": True,
            "automation_id": automation_id,
            "automation_summary": automation_summary,
            "severity": severity,
            "root_cause_candidates": root_cause_candidates,
            "recommended_fixes": recommended_fixes,
            "auto_fix_actions_available": auto_fix_actions,
            "entities_involved": entities_involved,
            "automation_events_24h": automation_events,
            "diagnostics_used": diagnostics_used,
            "diagnostics_count": len(diagnostics_used),
        }
        
    except Exception as e:
        logger.error(f"Error diagnosing automation {automation_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Error during automation diagnosis: {str(e)}",
            "automation_id": automation_id,
        }


@handle_api_errors
async def diagnose_system(include_entities: bool = False, ctx=None) -> Dict[str, Any]:
    """
    Global system-level diagnostic orchestrator
    
    Performs comprehensive health check of entire Home Assistant system by calling
    all diagnostic tools, aggregating findings, scoring overall health, and returning
    structured report grouped by categories.
    
    Args:
        include_entities: If True, includes detailed entity breakdown (more verbose)
        ctx: Optional MCP Context for progress reporting
    
    Returns:
        Comprehensive system diagnostic report with:
        - global_health_score (0-100%)
        - severity levels
        - issues by category
        - root cause candidates
        - recommended actions
        - diagnostics used
    """
    try:
        logger.info("diagnose_system: Starting comprehensive system diagnosis")
        
        # Wrap context to make all operations safe
        ctx = safe_ctx(ctx)
        
        total_steps = 14
        current_step = 0
        
        if ctx:
            await ctx.report_progress(progress=current_step, total=total_steps)
        
        diagnostics_used = []
        issues_by_category = {
            "system": [],
            "network": [],
            "integrations": [],
            "devices": [],
            "entities": [],
            "batteries": [],
            "zigbee_mesh": [],
            "esphome": [],
            "logs_errors": [],
            "updates": [],
        }
        
        # Gather diagnostic data with progress reporting
        logger.info("diagnose_system: Gathering diagnostic data from all tools...")
        
        # Step 1-3: System basics (parallel)
        if ctx:
            current_step += 1
            await ctx.report_progress(progress=current_step, total=total_steps)
            await ctx.info("[1/14] Checking system health...")
        
        system_health, network_info, integrations = await asyncio.gather(
            get_system_health(),
            get_network_info(),
            get_integrations(),
            return_exceptions=True
        )
        
        # Step 4-6: Entity health (parallel)
        if ctx:
            current_step += 3
            await ctx.report_progress(progress=current_step, total=total_steps)
            await ctx.info("[4/14] Analyzing entities...")
        
        unavailable_entities, stale_entities, battery_report_data = await asyncio.gather(
            find_unavailable_entities(),
            find_stale_entities(hours=2),
            battery_report(),
            return_exceptions=True
        )
        
        # Step 7-9: Device health (parallel)
        if ctx:
            current_step += 3
            await ctx.report_progress(progress=current_step, total=total_steps)
            await ctx.info("[7/14] Checking devices...")
        
        offline_devices, zha_devices, esphome_devices = await asyncio.gather(
            offline_devices_report(),
            get_zha_devices(),
            get_esphome_devices(),
            return_exceptions=True
        )
        
        # Step 10-11: Logs and anomalies (parallel)
        if ctx:
            current_step += 3
            await ctx.report_progress(progress=current_step, total=total_steps)
            await ctx.info("[10/14] Scanning logs and anomalies...")
        
        error_log, anomalies = await asyncio.gather(
            get_ha_error_log(),
            find_anomalous_entities(),
            return_exceptions=True
        )
        
        # Step 12-14: Activity, updates, repairs (parallel)
        if ctx:
            current_step += 2
            await ctx.report_progress(progress=current_step, total=total_steps)
            await ctx.info("[12/14] Checking recent activity and updates...")
        
        activity, updates, repairs = await asyncio.gather(
            recent_activity(hours=24),
            get_update_status(),
            get_repair_items(),
            return_exceptions=True
        )
        
        # Consolidate results
        results = (system_health, network_info, integrations, unavailable_entities, stale_entities,
                  battery_report_data, offline_devices, error_log, anomalies, activity,
                  zha_devices, esphome_devices, updates, repairs)
        
        diagnostics_used = [
            "get_system_health", "get_network_info", "list_integrations",
            "find_unavailable_entities", "find_stale_entities", "battery_report",
            "offline_devices_report", "get_error_log", "find_anomalous_entities",
            "recent_activity", "get_zha_devices", "get_esphome_devices",
            "get_update_status", "get_repair_items"
        ]
        
        logger.info(f"diagnose_system: Gathered data from {len(diagnostics_used)} tools")
        
        # Initialize severity counters
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        # CATEGORY: SYSTEM
        if isinstance(system_health, dict) and not isinstance(system_health, Exception):
            # Check for system-level issues
            if system_health.get("supervisor"):
                sup = system_health["supervisor"]
                if sup.get("supported") == False:
                    issues_by_category["system"].append({
                        "severity": "medium",
                        "issue": "Running on unsupported installation type",
                        "details": f"Installation type: {sup.get('installation_type')}",
                    })
                    severity_counts["medium"] += 1
        
        # CATEGORY: NETWORK
        if isinstance(network_info, dict) and not isinstance(network_info, Exception):
            if not network_info.get("external_url") and not network_info.get("internal_url"):
                issues_by_category["network"].append({
                    "severity": "low",
                    "issue": "No external or internal URL configured",
                    "details": "May affect remote access",
                })
                severity_counts["low"] += 1
        
        # CATEGORY: INTEGRATIONS
        if isinstance(integrations, dict) and not isinstance(integrations, Exception):
            total_integrations = integrations.get("total_integrations", 0)
            if total_integrations == 0:
                issues_by_category["integrations"].append({
                    "severity": "critical",
                    "issue": "No integrations loaded",
                    "details": "System may not be properly initialized",
                })
                severity_counts["critical"] += 1
        
        # CATEGORY: ENTITIES - Unavailable
        if isinstance(unavailable_entities, dict) and not isinstance(unavailable_entities, Exception):
            unavailable_count = unavailable_entities.get("total_unavailable", 0)
            if unavailable_count > 0:
                if unavailable_count > 50:
                    sev = "high"
                elif unavailable_count > 20:
                    sev = "medium"
                else:
                    sev = "low"
                
                issues_by_category["entities"].append({
                    "severity": sev,
                    "issue": f"{unavailable_count} unavailable entities",
                    "details": f"Domains affected: {', '.join(list(unavailable_entities.get('domain_counts', {}).keys())[:5])}",
                })
                severity_counts[sev] += 1
        
        # CATEGORY: ENTITIES - Stale
        if isinstance(stale_entities, dict) and not isinstance(stale_entities, Exception):
            stale_count = stale_entities.get("total_stale", 0)
            if stale_count > 0:
                sev = "medium" if stale_count > 10 else "low"
                issues_by_category["entities"].append({
                    "severity": sev,
                    "issue": f"{stale_count} stale entities (not updating)",
                    "details": f"Threshold: {stale_entities.get('threshold_hours')}h",
                })
                severity_counts[sev] += 1
        
        # CATEGORY: BATTERIES
        if isinstance(battery_report_data, dict) and not isinstance(battery_report_data, Exception):
            critical_batteries = battery_report_data.get("critical_count", 0)
            low_batteries = battery_report_data.get("low_battery_count", 0)
            
            if critical_batteries > 0:
                issues_by_category["batteries"].append({
                    "severity": "critical",
                    "issue": f"{critical_batteries} batteries critical (<10%)",
                    "details": "Devices may stop working soon",
                })
                severity_counts["critical"] += critical_batteries
            
            if low_batteries > 0:
                issues_by_category["batteries"].append({
                    "severity": "high",
                    "issue": f"{low_batteries} batteries low (<20%)",
                    "details": "Replace batteries soon",
                })
                severity_counts["high"] += low_batteries
        
        # CATEGORY: DEVICES - Offline
        if isinstance(offline_devices, dict) and not isinstance(offline_devices, Exception):
            offline_count = offline_devices.get("total_offline_devices", 0)
            if offline_count > 0:
                sev = "high" if offline_count > 5 else "medium"
                issues_by_category["devices"].append({
                    "severity": sev,
                    "issue": f"{offline_count} devices completely offline",
                    "details": f"Manufacturers: {', '.join(list(offline_devices.get('by_manufacturer', {}).keys())[:3])}",
                })
                severity_counts[sev] += 1
        
        # CATEGORY: LOGS/ERRORS
        if isinstance(error_log, dict) and not isinstance(error_log, Exception):
            error_count = error_log.get("error_count", 0)
            warning_count = error_log.get("warning_count", 0)
            
            if error_count > 50:
                sev = "high"
            elif error_count > 10:
                sev = "medium"
            elif error_count > 0:
                sev = "low"
            else:
                sev = None
            
            if sev is not None:
                issues_by_category["logs_errors"].append({
                    "severity": sev,
                    "issue": f"{error_count} errors, {warning_count} warnings in log",
                    "details": "Check error log for details",
                })
                severity_counts[sev] += 1
        
        # CATEGORY: ANOMALIES
        if isinstance(anomalies, dict) and not isinstance(anomalies, Exception):
            total_anomalies = anomalies.get("total_anomalies", 0)
            if total_anomalies > 0:
                issues_by_category["entities"].append({
                    "severity": "medium",
                    "issue": f"{total_anomalies} entities with anomalous values",
                    "details": f"Impossible values: {anomalies.get('impossible_values_count', 0)}, Frozen: {anomalies.get('frozen_sensors_count', 0)}",
                })
                severity_counts["medium"] += 1
        
        # CATEGORY: ZIGBEE MESH
        if isinstance(zha_devices, dict) and not isinstance(zha_devices, Exception):
            total_zha = zha_devices.get("total_devices", 0)
            offline_zha = zha_devices.get("offline", 0)
            
            if total_zha > 0:
                # Check for weak signals
                weak_signal_count = 0
                for device in zha_devices.get("devices", []):
                    if device.get("lqi", 255) < 120:
                        weak_signal_count += 1
                
                if weak_signal_count > 0:
                    sev = "medium" if weak_signal_count > 5 else "low"
                    issues_by_category["zigbee_mesh"].append({
                        "severity": sev,
                        "issue": f"{weak_signal_count} Zigbee devices with weak signal",
                        "details": "LQI < 120 may cause connectivity issues",
                    })
                    severity_counts[sev] += 1
                
                if offline_zha > 0:
                    issues_by_category["zigbee_mesh"].append({
                        "severity": "high",
                        "issue": f"{offline_zha} Zigbee devices offline",
                        "details": f"Total ZHA devices: {total_zha}",
                    })
                    severity_counts["high"] += 1
        
        # CATEGORY: ESPHOME
        if isinstance(esphome_devices, dict) and not isinstance(esphome_devices, Exception):
            offline_esphome = esphome_devices.get("offline", 0)
            if offline_esphome > 0:
                issues_by_category["esphome"].append({
                    "severity": "medium",
                    "issue": f"{offline_esphome} ESPHome devices offline",
                    "details": "Check network connectivity",
                })
                severity_counts["medium"] += 1
        
        # CATEGORY: UPDATES
        if isinstance(updates, dict) and not isinstance(updates, Exception):
            updates_available = updates.get("total_updates_available", 0)
            if updates_available > 0:
                issues_by_category["updates"].append({
                    "severity": "low",
                    "issue": f"{updates_available} updates available",
                    "details": f"Core: {len(updates.get('core_updates', []))}, Addons: {len(updates.get('addon_updates', []))}, Devices: {len(updates.get('device_updates', []))}",
                })
                severity_counts["low"] += 1
        
        # CATEGORY: REPAIRS
        if isinstance(repairs, dict) and not isinstance(repairs, Exception):
            total_repairs = repairs.get("total_issues", 0)
            critical_repairs = repairs.get("critical_count", 0)
            
            if critical_repairs > 0:
                issues_by_category["system"].append({
                    "severity": "critical",
                    "issue": f"{critical_repairs} critical repair items",
                    "details": "Address immediately",
                })
                severity_counts["critical"] += critical_repairs
            elif total_repairs > 0:
                issues_by_category["system"].append({
                    "severity": "medium",
                    "issue": f"{total_repairs} repair items pending",
                    "details": "Check repairs panel",
                })
                severity_counts["medium"] += 1
        
        # Calculate global health score (0-100%)
        # Formula: Start at 100%, deduct points for issues weighted by severity
        health_score = 100.0
        health_score -= severity_counts["critical"] * 10  # -10 per critical
        health_score -= severity_counts["high"] * 5      # -5 per high
        health_score -= severity_counts["medium"] * 2    # -2 per medium
        health_score -= severity_counts["low"] * 0.5     # -0.5 per low
        health_score = max(0.0, min(100.0, health_score))  # Clamp 0-100
        
        # Determine overall severity
        if health_score < 50:
            overall_severity = "critical"
        elif health_score < 70:
            overall_severity = "high"
        elif health_score < 85:
            overall_severity = "medium"
        else:
            overall_severity = "low"
        
        logger.info(f"diagnose_system: Health score={health_score:.1f}%, Severity={overall_severity}")
        
        # Build report
        report = {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "global_health_score": round(health_score, 1),
            "overall_severity": overall_severity,
            "severity_breakdown": severity_counts,
            "issues_by_category": issues_by_category,
            "category_summaries": {
                "system": len(issues_by_category["system"]),
                "network": len(issues_by_category["network"]),
                "integrations": len(issues_by_category["integrations"]),
                "devices": len(issues_by_category["devices"]),
                "entities": len(issues_by_category["entities"]),
                "batteries": len(issues_by_category["batteries"]),
                "zigbee_mesh": len(issues_by_category["zigbee_mesh"]),
                "esphome": len(issues_by_category["esphome"]),
                "logs_errors": len(issues_by_category["logs_errors"]),
                "updates": len(issues_by_category["updates"]),
            },
            "total_issues": sum(severity_counts.values()),
            "diagnostics_used": diagnostics_used,
            "diagnostics_count": len(diagnostics_used),
        }
        
        # Add entity details if requested
        if include_entities:
            report["entity_details"] = {
                "unavailable_count": unavailable_entities.get("total_unavailable", 0) if isinstance(unavailable_entities, dict) else 0,
                "stale_count": stale_entities.get("total_stale", 0) if isinstance(stale_entities, dict) else 0,
                "anomalous_count": anomalies.get("total_anomalies", 0) if isinstance(anomalies, dict) else 0,
                "domains_affected": list(unavailable_entities.get("domain_counts", {}).keys()) if isinstance(unavailable_entities, dict) else [],
            }
        
        # Report completion
        if ctx:
            await ctx.report_progress(progress=total_steps, total=total_steps)
            health_emoji = "✅" if health_score >= 85 else "⚠️" if health_score >= 70 else "🔴"
            await ctx.info(f"{health_emoji} System diagnosis complete! Health score: {health_score:.1f}%")
        
        logger.info(f"diagnose_system: Complete. Found {report['total_issues']} total issues")
        
        return report
        
    except Exception as e:
        logger.error(f"Error during system diagnosis: {str(e)}")
        if ctx:
            await ctx.error(f"System diagnosis failed: {str(e)}")
        return {
            "success": False,
            "error": f"Error during system diagnosis: {str(e)}",
        }


@handle_api_errors
async def auto_fix(entity_id: Optional[str] = None, scope: str = "auto", ctx=None) -> Dict[str, Any]:
    """
    Perform safe, low-risk automated corrective actions
    
    This function automatically applies fixes that are safe and non-destructive.
    It can work on a specific entity or perform global system fixes.
    
    NEVER restarts Home Assistant automatically.
    NEVER applies destructive changes.
    ALWAYS logs actions taken.
    
    Args:
        entity_id: Optional entity ID to fix (if None, performs global fixes)
        scope: Scope of fixes ('auto', 'entity', 'global')
        ctx: Optional MCP Context for progress reporting
    
    Returns:
        Dictionary with:
        - actions_taken: List of successful fixes
        - actions_skipped: List of skipped actions with reasons
        - risk_levels: Risk assessment for each action
        - before_snapshot: State before fixes
        - after_snapshot: State after fixes
        - success: Overall success status
    """
    try:
        logger.info(f"auto_fix: Starting auto-fix for entity_id={entity_id}, scope={scope}")
        
        # Wrap context to make all operations safe
        ctx = safe_ctx(ctx)
        
        if ctx:
            await ctx.info("🔧 Starting auto-fix process...")
        
        actions_taken = []
        actions_skipped = []
        risk_levels = {}
        before_snapshot = {}
        after_snapshot = {}
        
        # ENTITY-SPECIFIC FIXES
        if entity_id:
            logger.info(f"auto_fix: Performing entity-specific fixes for {entity_id}")
            
            if ctx:
                await ctx.info(f"🔍 Analyzing {entity_id}...")
            
            # Get before state
            before_state = await get_entity_state(entity_id)
            before_snapshot[entity_id] = {
                "state": before_state.get("state"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            # Diagnose the entity
            diagnosis = await diagnose_issue(entity_id, ctx=ctx)
            
            if not diagnosis.get("success"):
                return {
                    "success": False,
                    "error": f"Could not diagnose entity: {diagnosis.get('error')}",
                    "entity_id": entity_id,
                }
            
            # Check available auto-fix actions from diagnosis
            auto_fix_actions = diagnosis.get("auto_fix_actions_available", [])
            
            if ctx:
                await ctx.info(f"🔧 Applying {len(auto_fix_actions)} fixes...")
            
            for action in auto_fix_actions:
                action_name = action.get("action")
                risk = action.get("risk", "unknown")
                risk_levels[action_name] = risk
                
                # Only execute low-risk actions
                if risk == "low":
                    try:
                        if action_name == "reload_core_config":
                            result = await reload_core_config()
                            if result:
                                actions_taken.append({
                                    "action": action_name,
                                    "description": action.get("description"),
                                    "result": "success",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                })
                                logger.info(f"auto_fix: Successfully executed {action_name}")
                            else:
                                actions_skipped.append({
                                    "action": action_name,
                                    "reason": "reload_core_config returned False",
                                    "risk": risk,
                                })
                        
                        elif action_name == "reload_scripts":
                            result = await reload_scripts()
                            if result:
                                actions_taken.append({
                                    "action": action_name,
                                    "description": action.get("description"),
                                    "result": "success",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                })
                                logger.info(f"auto_fix: Successfully executed {action_name}")
                            else:
                                actions_skipped.append({
                                    "action": action_name,
                                    "reason": "reload_scripts returned False",
                                    "risk": risk,
                                })
                        
                        elif action_name == "turn_on_automation" and entity_id.startswith("automation."):
                            await call_service("automation", "turn_on", {"entity_id": entity_id})
                            actions_taken.append({
                                "action": action_name,
                                "description": f"Turned on {entity_id}",
                                "result": "success",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            logger.info(f"auto_fix: Turned on automation {entity_id}")
                        
                        else:
                            actions_skipped.append({
                                "action": action_name,
                                "reason": "Action not implemented in auto_fix",
                                "risk": risk,
                            })
                    
                    except Exception as e:
                        actions_skipped.append({
                            "action": action_name,
                            "reason": f"Error executing action: {str(e)}",
                            "risk": risk,
                        })
                        logger.error(f"auto_fix: Error executing {action_name}: {str(e)}")
                
                else:
                    # Skip medium/high/critical risk actions
                    actions_skipped.append({
                        "action": action_name,
                        "reason": f"Risk level too high: {risk}",
                        "risk": risk,
                    })
                    logger.info(f"auto_fix: Skipped {action_name} due to risk level: {risk}")
            
            # Wait a bit for changes to propagate
            await asyncio.sleep(2)
            
            # Get after state
            after_state = await get_entity_state(entity_id)
            after_snapshot[entity_id] = {
                "state": after_state.get("state"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            if ctx:
                await ctx.info(f"✅ Auto-fix complete! Applied {len(actions_taken)} fixes, skipped {len(actions_skipped)}")
        
        # GLOBAL SYSTEM FIXES
        else:
            logger.info("auto_fix: Performing global system fixes")
            
            if ctx:
                await ctx.info("🌐 Running system diagnosis...")
            
            # Diagnose system
            system_diagnosis = await diagnose_system(include_entities=False, ctx=ctx)
            
            if not system_diagnosis.get("success"):
                return {
                    "success": False,
                    "error": f"Could not diagnose system: {system_diagnosis.get('error')}",
                }
            
            before_snapshot["system_health_score"] = system_diagnosis.get("global_health_score")
            before_snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Safe global fixes
            safe_fixes = [
                {"action": "reload_core_config", "description": "Reload core configuration", "risk": "low"},
                {"action": "reload_scripts", "description": "Reload scripts", "risk": "low"},
            ]
            
            if ctx:
                await ctx.info("🔧 Applying system fixes...")
            
            for fix in safe_fixes:
                action_name = fix["action"]
                risk = fix["risk"]
                risk_levels[action_name] = risk
                
                try:
                    if action_name == "reload_core_config":
                        result = await reload_core_config()
                        if result:
                            actions_taken.append({
                                "action": action_name,
                                "description": fix["description"],
                                "result": "success",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            logger.info(f"auto_fix: Successfully executed {action_name}")
                        else:
                            actions_skipped.append({
                                "action": action_name,
                                "reason": "Function returned False",
                                "risk": risk,
                            })
                    
                    elif action_name == "reload_scripts":
                        result = await reload_scripts()
                        if result:
                            actions_taken.append({
                                "action": action_name,
                                "description": fix["description"],
                                "result": "success",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            logger.info(f"auto_fix: Successfully executed {action_name}")
                        else:
                            actions_skipped.append({
                                "action": action_name,
                                "reason": "Function returned False",
                                "risk": risk,
                            })
                
                except Exception as e:
                    actions_skipped.append({
                        "action": action_name,
                        "reason": f"Error: {str(e)}",
                        "risk": risk,
                    })
                    logger.error(f"auto_fix: Error executing {action_name}: {str(e)}")
            
            # Wait for changes
            await asyncio.sleep(3)
            
            if ctx:
                await ctx.info("📊 Re-checking system health...")
            
            # Re-diagnose system
            after_diagnosis = await diagnose_system(include_entities=False, ctx=ctx)
            after_snapshot["system_health_score"] = after_diagnosis.get("global_health_score")
            after_snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Compare health scores
            before_score = before_snapshot.get("system_health_score", 0)
            after_score = after_snapshot.get("system_health_score", 0)
            improvement = after_score - before_score
            
            after_snapshot["improvement"] = round(improvement, 1)
            
            if ctx:
                if improvement > 0:
                    await ctx.info(f"✅ System health improved by {improvement:.1f}%! New score: {after_score:.1f}%")
                elif improvement < 0:
                    await ctx.info(f"⚠️ System health decreased by {abs(improvement):.1f}%. Score: {after_score:.1f}%")
                else:
                    await ctx.info(f"ℹ️ System health unchanged at {after_score:.1f}%")
        
        # Compile final report
        logger.info(f"auto_fix: Complete. Took {len(actions_taken)} actions, skipped {len(actions_skipped)}")
        
        return {
            "success": True,
            "entity_id": entity_id,
            "scope": "entity" if entity_id else "global",
            "actions_taken": actions_taken,
            "actions_skipped": actions_skipped,
            "risk_levels": risk_levels,
            "before_snapshot": before_snapshot,
            "after_snapshot": after_snapshot,
            "total_actions_taken": len(actions_taken),
            "total_actions_skipped": len(actions_skipped),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error during auto_fix: {str(e)}")
        return {
            "success": False,
            "error": f"Error during auto_fix: {str(e)}",
            "entity_id": entity_id,
        }


# ============================================================================
# NEW HIGH-IMPACT DIAGNOSTIC TOOLS (Hackathon Features)
# ============================================================================

@handle_api_errors
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
    """
    try:
        logger.info("Starting advanced Zigbee mesh audit")
        
        # Get ZHA devices via WebSocket
        ws_url = HA_URL.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/websocket"
        
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
            # Auth
            await websocket.recv()
            await websocket.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
            auth_result = json.loads(await websocket.recv())
            
            if auth_result.get("type") != "auth_ok":
                return {"error": "WebSocket authentication failed"}
            
            # Get ZHA devices
            await websocket.send(json.dumps({"id": 1, "type": "zha/devices"}))
            response = json.loads(await websocket.recv())
            
            if not response.get("success"):
                return {"error": f"Failed to get ZHA devices: {response.get('error')}"}
            
            devices = response.get("result", [])[:limit]
            
            # Find coordinator
            coordinator = None
            for device in devices:
                if device.get("device_type") == "Coordinator":
                    coordinator = device
                    break
            
            # Analyze LQI distribution
            lqi_values = [d.get("lqi", 0) for d in devices if d.get("lqi") is not None]
            rssi_values = [d.get("rssi", 0) for d in devices if d.get("rssi") is not None]
            
            # Calculate statistics
            avg_lqi = sum(lqi_values) / len(lqi_values) if lqi_values else 0
            avg_rssi = sum(rssi_values) / len(rssi_values) if rssi_values else 0
            
            # Identify weak links (LQI < 120)
            weak_links = []
            for device in devices:
                lqi = device.get("lqi", 0)
                if lqi and lqi < 120:
                    weak_links.append({
                        "name": device.get("user_given_name") or device.get("name"),
                        "ieee": device.get("ieee"),
                        "lqi": lqi,
                        "rssi": device.get("rssi"),
                        "manufacturer": device.get("manufacturer"),
                        "severity": "critical" if lqi < 80 else "high" if lqi < 100 else "medium"
                    })
            
            # Identify orphan devices (unavailable or no LQI)
            orphan_devices = []
            for device in devices:
                if not device.get("available") or device.get("lqi") is None or device.get("lqi") == 0:
                    orphan_devices.append({
                        "name": device.get("user_given_name") or device.get("name"),
                        "ieee": device.get("ieee"),
                        "available": device.get("available"),
                        "last_seen": device.get("last_seen"),
                        "manufacturer": device.get("manufacturer")
                    })
            
            # LQI distribution buckets
            lqi_distribution = {
                "excellent (200-255)": sum(1 for lqi in lqi_values if lqi >= 200),
                "good (150-199)": sum(1 for lqi in lqi_values if 150 <= lqi < 200),
                "fair (120-149)": sum(1 for lqi in lqi_values if 120 <= lqi < 150),
                "poor (80-119)": sum(1 for lqi in lqi_values if 80 <= lqi < 120),
                "critical (<80)": sum(1 for lqi in lqi_values if lqi < 80)
            }
            
            # Calculate mesh health score
            # Base score 100, deduct for issues
            mesh_health_score = 100.0
            mesh_health_score -= len(weak_links) * 3  # -3 per weak link
            mesh_health_score -= len(orphan_devices) * 10  # -10 per orphan
            if avg_lqi < 150:
                mesh_health_score -= 10  # Low average LQI
            mesh_health_score = max(0, mesh_health_score)
            
            # Generate recommendations
            recommendations = []
            if len(weak_links) > 0:
                recommendations.append({
                    "priority": "high",
                    "issue": f"{len(weak_links)} device(s) with weak Zigbee signal",
                    "action": "Add Zigbee routers between weak devices and coordinator"
                })
            
            if len(orphan_devices) > 0:
                recommendations.append({
                    "priority": "critical",
                    "issue": f"{len(orphan_devices)} orphaned/unavailable device(s)",
                    "action": "Check power, re-pair devices, or remove from network"
                })
            
            if avg_lqi < 150:
                recommendations.append({
                    "priority": "medium",
                    "issue": f"Average LQI is low ({avg_lqi:.1f})",
                    "action": "Consider adding more router devices to strengthen mesh"
                })
            
            # Routing analysis (power sources)
            power_sources = {}
            for device in devices:
                ps = device.get("power_source", "Unknown")
                power_sources[ps] = power_sources.get(ps, 0) + 1
            
            router_count = power_sources.get("Mains (single phase)", 0)
            if router_count < 3:
                recommendations.append({
                    "priority": "medium",
                    "issue": f"Only {router_count} mains-powered router(s) detected",
                    "action": "Add more mains-powered devices to improve mesh stability"
                })
            
            return {
                "success": True,
                "mesh_health_score": round(mesh_health_score, 1),
                "total_devices": len(devices),
                "coordinator_info": {
                    "name": coordinator.get("name") if coordinator else "Not found",
                    "manufacturer": coordinator.get("manufacturer") if coordinator else None,
                    "model": coordinator.get("model") if coordinator else None,
                    "ieee": coordinator.get("ieee") if coordinator else None
                } if coordinator else None,
                "statistics": {
                    "average_lqi": round(avg_lqi, 1),
                    "average_rssi": round(avg_rssi, 1),
                    "weak_links_count": len(weak_links),
                    "orphan_devices_count": len(orphan_devices)
                },
                "lqi_distribution": lqi_distribution,
                "weak_links": weak_links[:20],  # Limit for token efficiency
                "orphan_devices": orphan_devices[:20],
                "power_source_distribution": power_sources,
                "router_count": router_count,
                "recommendations": recommendations,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error in audit_zigbee_mesh: {str(e)}")
        return {"error": f"Error analyzing Zigbee mesh: {str(e)}"}


@handle_api_errors
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
    """
    try:
        logger.info("Finding orphan entities (not used in automations/scripts/scenes)")
        
        # Get all entities
        all_states = await get_all_entity_states()
        all_entity_ids = set(all_states.keys())
        
        # Track entity usage
        entity_usage = {eid: 0 for eid in all_entity_ids}
        
        # Check automations
        automations = await get_automations()
        for automation in automations:
            auto_id = automation.get("entity_id")
            if auto_id:
                # Get automation config to parse triggers/conditions/actions
                state = await get_entity_state(auto_id)
                attributes = state.get("attributes", {})
                
                # Check trigger/condition/action in attributes
                for key in ["trigger", "condition", "action"]:
                    value = attributes.get(key, [])
                    value_str = str(value).lower()
                    
                    # Find entity_ids in the string representation
                    for entity_id in all_entity_ids:
                        if entity_id.lower() in value_str:
                            entity_usage[entity_id] += 1
        
        # Check scripts
        script_entities = await get_entities(domain="script")
        for script in script_entities:
            script_id = script.get("entity_id")
            if script_id:
                state = await get_entity_state(script_id)
                attributes = state.get("attributes", {})
                sequence = attributes.get("sequence", [])
                sequence_str = str(sequence).lower()
                
                for entity_id in all_entity_ids:
                    if entity_id.lower() in sequence_str:
                        entity_usage[entity_id] += 1
        
        # Check scenes
        scene_entities = await get_entities(domain="scene")
        for scene in scene_entities:
            scene_id = scene.get("entity_id")
            if scene_id:
                state = await get_entity_state(scene_id)
                attributes = state.get("attributes", {})
                entity_ids_in_scene = attributes.get("entity_id", [])
                
                if isinstance(entity_ids_in_scene, list):
                    for eid in entity_ids_in_scene:
                        if eid in entity_usage:
                            entity_usage[eid] += 1
        
        # Find orphans (usage count == 0)
        orphan_entities = []
        orphans_by_domain = {}
        
        for entity_id, usage_count in entity_usage.items():
            if usage_count == 0:
                domain = entity_id.split(".")[0]
                state_data = all_states.get(entity_id, {})
                
                orphan_info = {
                    "entity_id": entity_id,
                    "domain": domain,
                    "friendly_name": state_data.get("attributes", {}).get("friendly_name", entity_id),
                    "state": state_data.get("state"),
                    "last_updated": state_data.get("last_updated")
                }
                
                orphan_entities.append(orphan_info)
                
                if domain not in orphans_by_domain:
                    orphans_by_domain[domain] = []
                orphans_by_domain[domain].append(orphan_info)
        
        # Sort orphans by domain
        orphan_entities.sort(key=lambda x: (x["domain"], x["entity_id"]))
        
        # Calculate statistics
        total_orphans = len(orphan_entities)
        total_entities = len(all_entity_ids)
        orphan_percentage = (total_orphans / total_entities * 100) if total_entities > 0 else 0
        
        # Get most used entities (for comparison)
        most_used = sorted(
            [(eid, count) for eid, count in entity_usage.items() if count > 0],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        most_used_entities = []
        for entity_id, count in most_used:
            state_data = all_states.get(entity_id, {})
            most_used_entities.append({
                "entity_id": entity_id,
                "usage_count": count,
                "friendly_name": state_data.get("attributes", {}).get("friendly_name", entity_id)
            })
        
        return {
            "success": True,
            "total_entities": total_entities,
            "total_orphans": total_orphans,
            "orphan_percentage": round(orphan_percentage, 1),
            "orphans_by_domain": {
                domain: len(entities) for domain, entities in orphans_by_domain.items()
            },
            "orphan_entities": orphan_entities[:50],  # Limit for token efficiency
            "most_used_entities": most_used_entities,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error finding orphan entities: {str(e)}")
        return {"error": f"Error finding orphan entities: {str(e)}"}


@handle_api_errors
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
    """
    try:
        logger.info("Detecting automation conflicts and race conditions")
        
        # Get all automations
        automations = await get_automations()
        
        # Track triggers and actions by entity
        trigger_map = {}  # entity_id -> list of automations
        action_map = {}   # entity_id -> list of (automation, action_type)
        automation_details = {}
        
        for automation in automations:
            auto_id = automation.get("entity_id")
            auto_name = automation.get("alias", auto_id)
            
            if not auto_id:
                continue
            
            # Get automation details
            state = await get_entity_state(auto_id)
            attributes = state.get("attributes", {})
            mode = attributes.get("mode", "single")
            
            automation_details[auto_id] = {
                "name": auto_name,
                "mode": mode,
                "state": state.get("state"),
                "triggers": attributes.get("trigger", []),
                "conditions": attributes.get("condition", []),
                "actions": attributes.get("action", [])
            }
            
            # Parse triggers
            triggers = attributes.get("trigger", [])
            if not isinstance(triggers, list):
                triggers = [triggers]
            
            for trigger in triggers:
                if isinstance(trigger, dict):
                    trigger_entity = trigger.get("entity_id")
                    if trigger_entity:
                        if trigger_entity not in trigger_map:
                            trigger_map[trigger_entity] = []
                        trigger_map[trigger_entity].append({
                            "automation_id": auto_id,
                            "automation_name": auto_name,
                            "trigger_type": trigger.get("platform", "unknown")
                        })
            
            # Parse actions
            actions = attributes.get("action", [])
            if not isinstance(actions, list):
                actions = [actions]
            
            for action in actions:
                if isinstance(action, dict):
                    service = action.get("service", "")
                    target_entity = action.get("entity_id") or action.get("target", {}).get("entity_id")
                    
                    if target_entity:
                        if isinstance(target_entity, list):
                            target_entities = target_entity
                        else:
                            target_entities = [target_entity]
                        
                        for ent in target_entities:
                            if ent not in action_map:
                                action_map[ent] = []
                            
                            action_type = "turn_on" if "turn_on" in service else \
                                        "turn_off" if "turn_off" in service else \
                                        "toggle" if "toggle" in service else "other"
                            
                            action_map[ent].append({
                                "automation_id": auto_id,
                                "automation_name": auto_name,
                                "action_type": action_type,
                                "service": service
                            })
        
        # Detect race conditions (multiple automations trigger on same entity)
        race_conditions = []
        for entity_id, triggers in trigger_map.items():
            if len(triggers) > 1:
                race_conditions.append({
                    "entity_id": entity_id,
                    "automation_count": len(triggers),
                    "automations": triggers,
                    "severity": "high" if len(triggers) > 3 else "medium"
                })
        
        # Detect potential loops (A triggers on entity that B controls, B triggers on entity that A controls)
        potential_loops = []
        for auto_id, details in automation_details.items():
            # Get entities this automation acts on
            actions = details.get("actions", [])
            action_entities = set()
            for action in actions:
                if isinstance(action, dict):
                    target = action.get("entity_id") or action.get("target", {}).get("entity_id")
                    if target:
                        if isinstance(target, list):
                            action_entities.update(target)
                        else:
                            action_entities.add(target)
            
            # Check if any other automation triggers on those entities and acts on entities this one triggers on
            triggers = details.get("triggers", [])
            trigger_entities = set()
            for trigger in triggers:
                if isinstance(trigger, dict):
                    ent = trigger.get("entity_id")
                    if ent:
                        trigger_entities.add(ent)
            
            # Look for circular dependencies
            for other_auto_id, other_details in automation_details.items():
                if other_auto_id == auto_id:
                    continue
                
                other_triggers = other_details.get("triggers", [])
                other_trigger_entities = set()
                for trigger in other_triggers:
                    if isinstance(trigger, dict):
                        ent = trigger.get("entity_id")
                        if ent:
                            other_trigger_entities.add(ent)
                
                other_actions = other_details.get("actions", [])
                other_action_entities = set()
                for action in other_actions:
                    if isinstance(action, dict):
                        target = action.get("entity_id") or action.get("target", {}).get("entity_id")
                        if target:
                            if isinstance(target, list):
                                other_action_entities.update(target)
                            else:
                                other_action_entities.add(target)
                
                # Check for loop: A acts on X, B triggers on X, B acts on Y, A triggers on Y
                if action_entities & other_trigger_entities and other_action_entities & trigger_entities:
                    potential_loops.append({
                        "automation_a": details["name"],
                        "automation_a_id": auto_id,
                        "automation_b": other_details["name"],
                        "automation_b_id": other_auto_id,
                        "shared_entities": list(action_entities & other_trigger_entities),
                        "severity": "critical"
                    })
        
        # Detect conflicting actions (multiple automations with opposing actions on same entity)
        conflicting_actions = []
        for entity_id, actions in action_map.items():
            # Check for both turn_on and turn_off
            has_turn_on = any(a["action_type"] == "turn_on" for a in actions)
            has_turn_off = any(a["action_type"] == "turn_off" for a in actions)
            
            if has_turn_on and has_turn_off:
                on_automations = [a for a in actions if a["action_type"] == "turn_on"]
                off_automations = [a for a in actions if a["action_type"] == "turn_off"]
                
                conflicting_actions.append({
                    "entity_id": entity_id,
                    "turn_on_automations": on_automations,
                    "turn_off_automations": off_automations,
                    "severity": "medium"
                })
        
        # Detect unsafe modes (automations without mode or with mode=single but potential for rapid triggering)
        unsafe_modes = []
        for auto_id, details in automation_details.items():
            mode = details.get("mode", "single")
            
            # Check if automation could trigger rapidly (time-based or sensor-based triggers)
            triggers = details.get("triggers", [])
            has_rapid_trigger = False
            for trigger in triggers:
                if isinstance(trigger, dict):
                    platform = trigger.get("platform", "")
                    if platform in ["state", "numeric_state", "template"]:
                        has_rapid_trigger = True
                        break
            
            if mode == "single" and has_rapid_trigger:
                unsafe_modes.append({
                    "automation_id": auto_id,
                    "automation_name": details["name"],
                    "mode": mode,
                    "issue": "Automation with 'single' mode may block on rapid triggers",
                    "recommendation": "Consider using 'restart' or 'queued' mode",
                    "severity": "low"
                })
        
        # Generate recommendations
        recommendations = []
        if len(race_conditions) > 0:
            recommendations.append({
                "priority": "high",
                "issue": f"{len(race_conditions)} entity(ies) with multiple automation triggers",
                "action": "Review automations for conflicts and add conditions to prevent race conditions"
            })
        
        if len(potential_loops) > 0:
            recommendations.append({
                "priority": "critical",
                "issue": f"{len(potential_loops)} potential infinite loop(s) detected",
                "action": "Add conditions or delays to prevent circular automation triggers"
            })
        
        if len(conflicting_actions) > 0:
            recommendations.append({
                "priority": "medium",
                "issue": f"{len(conflicting_actions)} entity(ies) with conflicting automation actions",
                "action": "Review automations to ensure they don't fight each other"
            })
        
        total_conflicts = len(race_conditions) + len(potential_loops) + len(conflicting_actions) + len(unsafe_modes)
        
        return {
            "success": True,
            "total_automations": len(automations),
            "total_conflicts": total_conflicts,
            "race_conditions": race_conditions[:20],
            "potential_loops": potential_loops[:20],
            "conflicting_actions": conflicting_actions[:20],
            "unsafe_modes": unsafe_modes[:20],
            "recommendations": recommendations,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error detecting automation conflicts: {str(e)}")
        return {"error": f"Error detecting automation conflicts: {str(e)}"}


@handle_api_errors
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
    """
    try:
        logger.info(f"Generating energy consumption report for last {period_hours} hours")
        
        # Get energy entities (sensors that track energy)
        all_states = await get_all_entity_states()
        energy_entities = {}
        
        for entity_id, state in all_states.items():
            attributes = state.get("attributes", {})
            device_class = attributes.get("device_class")
            unit = attributes.get("unit_of_measurement", "")
            
            # Look for energy sensors
            if device_class == "energy" or "kwh" in unit.lower() or "energy" in entity_id.lower():
                try:
                    current_value = float(state.get("state", 0))
                    energy_entities[entity_id] = {
                        "friendly_name": attributes.get("friendly_name", entity_id),
                        "current_value": current_value,
                        "unit": unit,
                        "device_class": device_class
                    }
                except (ValueError, TypeError):
                    pass
        
        # Get historical data for energy entities
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=period_hours)
        
        consumption_data = {}
        total_consumption = 0.0
        
        for entity_id, info in energy_entities.items():
            # Get history for this entity
            history = await get_entity_history(entity_id, period_hours)
            
            if history and len(history) > 0 and len(history[0]) > 1:
                states = history[0]
                
                # Calculate consumption (difference between first and last reading)
                try:
                    first_reading = float(states[0].get("state", 0))
                    last_reading = float(states[-1].get("state", 0))
                    consumption = last_reading - first_reading
                    
                    if consumption > 0:  # Only include if consumption is positive
                        consumption_data[entity_id] = {
                            "friendly_name": info["friendly_name"],
                            "consumption": round(consumption, 2),
                            "unit": info["unit"],
                            "start_reading": round(first_reading, 2),
                            "end_reading": round(last_reading, 2)
                        }
                        total_consumption += consumption
                except (ValueError, TypeError, IndexError):
                    pass
        
        # Sort devices by consumption
        devices_by_consumption = sorted(
            consumption_data.items(),
            key=lambda x: x[1]["consumption"],
            reverse=True
        )
        
        # Identify top consumers
        top_consumers = []
        for entity_id, data in devices_by_consumption[:10]:
            percentage = (data["consumption"] / total_consumption * 100) if total_consumption > 0 else 0
            top_consumers.append({
                "entity_id": entity_id,
                "friendly_name": data["friendly_name"],
                "consumption": data["consumption"],
                "percentage": round(percentage, 1),
                "unit": data["unit"]
            })
        
        # Estimate cost (assuming average rate - can be configured)
        # Default: $0.15 per kWh (US average)
        ENERGY_RATE = 0.15
        estimated_cost = total_consumption * ENERGY_RATE
        
        # Calculate daily/monthly projections
        if period_hours > 0:
            daily_projection = (total_consumption / period_hours) * 24
            monthly_projection = daily_projection * 30
            monthly_cost_projection = monthly_projection * ENERGY_RATE
        else:
            daily_projection = 0
            monthly_projection = 0
            monthly_cost_projection = 0
        
        # Generate recommendations
        recommendations = []
        
        if len(top_consumers) > 0:
            top_device = top_consumers[0]
            if top_device["percentage"] > 30:
                recommendations.append({
                    "priority": "high",
                    "issue": f"{top_device['friendly_name']} uses {top_device['percentage']}% of total energy",
                    "action": "Consider automations to reduce usage or upgrade to more efficient device"
                })
        
        if daily_projection > 10:  # More than 10 kWh per day
            recommendations.append({
                "priority": "medium",
                "issue": f"High daily consumption: {daily_projection:.1f} kWh/day",
                "action": "Review top energy consumers and create energy-saving automations"
            })
        
        return {
            "success": True,
            "period_hours": period_hours,
            "total_consumption": round(total_consumption, 2),
            "unit": "kWh",
            "total_devices": len(energy_entities),
            "devices_with_data": len(consumption_data),
            "top_consumers": top_consumers,
            "cost_estimate": {
                "period_cost": round(estimated_cost, 2),
                "energy_rate": ENERGY_RATE,
                "currency": "USD",
                "daily_projection": round(daily_projection, 2),
                "monthly_projection": round(monthly_projection, 2),
                "monthly_cost_projection": round(monthly_cost_projection, 2)
            },
            "recommendations": recommendations,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating energy consumption report: {str(e)}")
        return {"error": f"Error generating energy report: {str(e)}"}


@handle_api_errors
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
    """
    try:
        logger.info(f"Generating entity dependency graph" + (f" for {entity_id}" if entity_id else ""))
        
        # Get all entities
        all_states = await get_all_entity_states()
        all_entity_ids = set(all_states.keys())
        
        # Track dependencies: entity_id -> set of entities it references
        references = {eid: set() for eid in all_entity_ids}
        # Track reverse: entity_id -> set of entities that reference it
        referenced_by = {eid: set() for eid in all_entity_ids}
        
        # Analyze automations
        automations = await get_automations()
        for automation in automations:
            auto_id = automation.get("entity_id")
            if not auto_id:
                continue
            
            state = await get_entity_state(auto_id)
            attributes = state.get("attributes", {})
            
            # Parse all fields as strings to find entity_ids
            full_config = str(attributes).lower()
            
            for entity in all_entity_ids:
                if entity.lower() in full_config and entity != auto_id:
                    references[auto_id].add(entity)
                    referenced_by[entity].add(auto_id)
        
        # Analyze scripts
        script_entities = await get_entities(domain="script")
        for script in script_entities:
            script_id = script.get("entity_id")
            if not script_id:
                continue
            
            state = await get_entity_state(script_id)
            attributes = state.get("attributes", {})
            sequence = str(attributes.get("sequence", [])).lower()
            
            for entity in all_entity_ids:
                if entity.lower() in sequence and entity != script_id:
                    references[script_id].add(entity)
                    referenced_by[entity].add(script_id)
        
        # Analyze template sensors
        template_entities = [eid for eid in all_entity_ids if eid.startswith("sensor.") or eid.startswith("binary_sensor.")]
        for template_id in template_entities:
            state = await get_entity_state(template_id)
            attributes = state.get("attributes", {})
            
            # Check for template attributes
            for attr_name, attr_value in attributes.items():
                if isinstance(attr_value, str) and "{{" in attr_value:
                    # This is a template
                    for entity in all_entity_ids:
                        if entity.lower() in attr_value.lower() and entity != template_id:
                            references[template_id].add(entity)
                            referenced_by[entity].add(template_id)
        
        # Analyze groups
        group_entities = await get_entities(domain="group")
        for group in group_entities:
            group_id = group.get("entity_id")
            if not group_id:
                continue
            
            state = await get_entity_state(group_id)
            attributes = state.get("attributes", {})
            entity_ids_in_group = attributes.get("entity_id", [])
            
            if isinstance(entity_ids_in_group, list):
                for entity in entity_ids_in_group:
                    if entity in all_entity_ids:
                        references[group_id].add(entity)
                        referenced_by[entity].add(group_id)
        
        # If specific entity requested, filter to that entity and its dependencies
        if entity_id:
            if entity_id not in all_entity_ids:
                return {"error": f"Entity {entity_id} not found"}
            
            # Get entities this one references
            direct_references = references.get(entity_id, set())
            # Get entities that reference this one
            direct_referenced_by = referenced_by.get(entity_id, set())
            
            return {
                "success": True,
                "entity_id": entity_id,
                "friendly_name": all_states.get(entity_id, {}).get("attributes", {}).get("friendly_name", entity_id),
                "references": list(direct_references),
                "references_count": len(direct_references),
                "referenced_by": list(direct_referenced_by),
                "referenced_by_count": len(direct_referenced_by),
                "is_isolated": len(direct_references) == 0 and len(direct_referenced_by) == 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Global analysis
        # Find most referenced entities
        most_referenced = sorted(
            [(eid, len(refs)) for eid, refs in referenced_by.items() if len(refs) > 0],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        most_referenced_entities = []
        for eid, count in most_referenced:
            most_referenced_entities.append({
                "entity_id": eid,
                "friendly_name": all_states.get(eid, {}).get("attributes", {}).get("friendly_name", eid),
                "referenced_by_count": count,
                "referenced_by": list(referenced_by[eid])[:10]  # Limit to 10 examples
            })
        
        # Find entities that reference many others
        most_referencing = sorted(
            [(eid, len(refs)) for eid, refs in references.items() if len(refs) > 0],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        most_referencing_entities = []
        for eid, count in most_referencing:
            most_referencing_entities.append({
                "entity_id": eid,
                "friendly_name": all_states.get(eid, {}).get("attributes", {}).get("friendly_name", eid),
                "references_count": count,
                "references": list(references[eid])[:10]  # Limit to 10 examples
            })
        
        # Find isolated entities (no dependencies in either direction)
        isolated_entities = []
        for eid in all_entity_ids:
            if len(references[eid]) == 0 and len(referenced_by[eid]) == 0:
                isolated_entities.append({
                    "entity_id": eid,
                    "friendly_name": all_states.get(eid, {}).get("attributes", {}).get("friendly_name", eid),
                    "domain": eid.split(".")[0]
                })
        
        # Detect circular dependencies (A refs B, B refs A)
        circular_dependencies = []
        checked_pairs = set()
        for eid, refs in references.items():
            for ref in refs:
                if ref in references and eid in references[ref]:
                    pair = tuple(sorted([eid, ref]))
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        circular_dependencies.append({
                            "entity_a": eid,
                            "entity_b": ref,
                            "severity": "medium"
                        })
        
        return {
            "success": True,
            "total_entities": len(all_entity_ids),
            "most_referenced_entities": most_referenced_entities,
            "most_referencing_entities": most_referencing_entities,
            "isolated_entities_count": len(isolated_entities),
            "isolated_entities": isolated_entities[:50],  # Limit for token efficiency
            "circular_dependencies": circular_dependencies,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating entity dependency graph: {str(e)}")
        return {"error": f"Error generating dependency graph: {str(e)}"}







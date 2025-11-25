# 🔍🐛 Home Assistant Diagnostics MCP Server 🏠🔧

**AI-Powered Home Assistant Diagnostics & Health Monitoring**

Built for Hugging Face MCP's 1st Birthday Hackathon 2025

## 🎥 Demo

<p align="center">
  <a href="https://youtu.be/VtPmbIxR7UQ">
    <img src="https://img.youtube.com/vi/VtPmbIxR7UQ/maxresdefault.jpg" alt="Demo Video" width="80%">
    <br>
    <img src="https://img.shields.io/badge/▶%20Watch%20Demo-YouTube-red?style=for-the-badge&logo=youtube">
  </a>
</p>

## 💡 Why This Exists

Home Assistant is a powerful platform for running your smart home, but when something stops working, diagnosing the issue can be a real headache. Manually checking hundreds of entities, downloading logs, or trying to explain the problem to an AI that can't see your system directly is frustrating.

This MCP server gives your favorite LLM direct access to diagnostic tools, enabling it to troubleshoot your Home Assistant instance autonomously, without you having to manually gather information.

No more debugging in the dark. Let AI be your smart home doctor.

## 🎯 What This MCP Server Does

Complete Model Context Protocol (MCP) server providing **39 specialized diagnostic tools**, **13 diagnostic resources**, and **4 AI prompts** for AI assistants to analyze, troubleshoot, and maintain Home Assistant installations.

### 🏆 Signature Features

**🌐 Advanced Network Analysis**
- Zigbee mesh health scoring (0-100) with LQI/RSSI distribution
- Weak link detection with severity levels
- Orphan device identification
- Router placement recommendations

**🧹 Smart Cleanup Intelligence**
- Orphan entity detection - Find entities not used anywhere
- Usage statistics - Most/least referenced entities
- Cleanup prioritization with safe-to-delete recommendations
- Domain-based grouping for organized cleanup

**🤖 Automation Safety Analyzer**
- Race condition detection - Multiple automations on same entity
- Infinite loop prevention - Circular automation dependencies
- Conflicting action detection
- Unsafe mode warnings

**⚡ Energy & Cost Intelligence**
- Energy consumption tracking with device-level breakdown
- Cost estimation with monthly projections
- Top consumer identification
- Energy-saving recommendations

**🔍 System-Wide Diagnostics**
- Health scoring (0-100%) with severity assessment
- Root cause analysis with confidence levels
- Real-time anomaly detection
- Automated issue classification (low/medium/high/critical)

**🔧 Safe Auto-Fix**
- Automated repairs with risk assessment
- No confirmation needed for safe operations
- Before/after snapshots for transparency

**📊 Advanced Monitoring**
- Battery health tracking with critical alerts
- Stale sensor detection
- Network connectivity monitoring
- Historical statistics with trend analysis

**🎯 Physical Device Management**
- Flash/beep device identification
- ZHA device mesh visualization
- ESPHome device monitoring
- Offline device detection by manufacturer

## 🛠️ What's Included

### 39 Diagnostic Tools

**Core Diagnostics (16 tools)**
- `get_version` - Home Assistant version info
- `battery_report` - Battery levels and critical alerts
- `find_unavailable_entities` - Detect unavailable entities
- `find_stale_entities` - Find sensors not updating
- `find_anomalous_entities` - Detect impossible/frozen values
- `offline_devices_report` - Devices completely offline
- `get_repair_items` - HA repair panel issues
- `get_error_log` - Error and warning analysis
- `diagnose_issue` - Deep entity diagnostics with AI
- `diagnose_automation` - Automation troubleshooting
- `diagnose_system` - Complete system health check
- `audit_zigbee_mesh` - Zigbee mesh health with LQI/RSSI
- `find_orphan_entities` - Detect unused entities
- `detect_automation_conflicts` - Race conditions & loops
- `energy_consumption_report` - Energy tracking & costs
- `entity_dependency_graph` - Entity relationships

**System Monitoring (6 tools)**
- `get_system_health` - CPU, memory, disk, supervisor
- `get_network_info` - Network configuration
- `get_update_status` - Available updates
- `list_integrations` - Loaded integrations
- `system_overview` - Complete system summary
- `get_zha_devices` - Zigbee devices with mesh quality

**Entity Management (13 tools)**
- `list_automations` - All automations with state
- `list_entities` - Query entities with filters
- `domain_summary_tool` - Domain statistics
- `search_entities_tool` - Semantic search
- `get_entity` - Entity state (basic/detailed)
- `get_entity_statistics` - Historical analysis
- `get_history` - State change history
- `entity_action` - Control entities (on/off/toggle)
- `call_service_tool` - Low-level HA service calls
- `identify_device` - Flash/beep device physically
- `recent_activity` - Recent events
- `get_esphome_devices` - ESPHome device status
- `get_addons` - Add-on status and updates

**Maintenance (6 tools)**
- `reload_core_config` - Reload HA configuration
- `reload_scripts` - Reload script definitions
- `auto_fix` - Automated safe repairs
- `restart_ha` - Restart Home Assistant

### 13 Diagnostic Resources

**Health & System Resources**
- `ha://diagnostics/health-score` - System health (markdown)
- `ha://diagnostics/health-score/json` - System health (JSON)
- `ha://diagnostics/system` - Full system report (markdown)
- `ha://diagnostics/system/json` - Full system report (JSON)
- `ha://diagnostics/zigbee-mesh` - Zigbee mesh health with ASCII viz
- `ha://diagnostics/system-health` - Complete health audit

**Entity Resources**
- `ha://diagnostics/entity/{entity_id}` - Entity diagnostics (markdown)
- `ha://diagnostics/entity/{entity_id}/json` - Entity diagnostics (JSON)
- `ha://entities` - All entities overview
- `ha://entities/{entity_id}` - Single entity
- `ha://entities/{entity_id}/detailed` - Entity with full attributes
- `ha://entities/domain/{domain}` - Entities by domain
- `ha://search/{query}/{limit}` - Entity search results

### 4 AI Prompts

Interactive conversation templates for diagnostic workflows:

1. **`debug_automation`** - Debug automations not triggering
2. **`troubleshoot_entity`** - Troubleshoot entity issues
3. **`automation_health_check`** - Find conflicts & improvements
4. **`diagnose_everything`** - **Ultimate diagnostic orchestrator**

#### Featured: diagnose_everything

Complete system audit in one command:
```
🏥 COMPLETE DIAGNOSTIC PROTOCOL:
├─ System-Wide Analysis (diagnose_system)
├─ Zigbee Mesh Audit (audit_zigbee_mesh)
├─ Entity Usage Analysis (find_orphan_entities)
├─ Automation Safety Check (detect_automation_conflicts)
├─ Energy Consumption (energy_consumption_report)
└─ Dependency Mapping (entity_dependency_graph)

📊 OUTPUT:
✅ Overall health score (0-100)
✅ Issues by severity & category
✅ Actionable recommendations
✅ Auto-fix opportunities
```

## 🚀 Installation

### Prerequisites
- Python 3.13+
- Home Assistant instance
- Long-lived access token

### Quick Start

```bash
# Clone repository
git clone https://github.com/burgueishon/Home-Assistant-Diagnostics-MCP-Server.git
cd Home-Assistant-Diagnostics-MCP-Server

# Install with uv (recommended)
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
uv pip install -e .

# Or use pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configuration

**Get Home Assistant Token:**
   - Go to: Home Assistant → Profile → Security
   - Create "Long-Lived Access Token"
   - Copy token (you'll use it in the next step)

## 🔧 Usage

### Claude Desktop

Edit config file:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "home-assistant-diagnostics": {
      "command": "uv",
      "args": ["run", "home-assistant-diagnostics-mcp"],
      "cwd": "/absolute/path/to/Home-Assistant-Diagnostics-MCP-Server",
      "env": {
        "HA_URL": "http://YOUR_HA_IP:8123",
        "HA_TOKEN": "your-long-lived-token"
      }
    }
  }
}
```

### Claude Code

```bash
# From project directory
claude mcp add --transport stdio \
  -e HA_URL=http://YOUR_HA_IP:8123 \
  -e HA_TOKEN=your-long-lived-token \
  -- home-assistant-diagnostics uv run home-assistant-diagnostics-mcp
```

Replace `YOUR_HA_IP` with your Home Assistant IP/hostname and `your-long-lived-token` with your actual token.

### VS Code 

Create `.vscode/mcp.json`:
```json
{
  "servers": {
    "home-assistant-diagnostics": {
      "command": "uv",
      "args": ["run", "home-assistant-diagnostics-mcp"],
      "cwd": "/absolute/path/to/Home-Assistant-Diagnostics-MCP-Server",
      "env": {
        "HA_URL": "http://YOUR_HA_IP:8123",
        "HA_TOKEN": "your-long-lived-token"
      }
    }
  }
}
```

### Cursor IDE

Add to Cursor settings:
```json
{
  "mcp": {
    "servers": {
      "home-assistant-diagnostics": {
        "command": "uv",
        "args": ["run", "home-assistant-diagnostics-mcp"],
        "cwd": "/absolute/path/to/Home-Assistant-Diagnostics-MCP-Server",
        "env": {
          "HA_URL": "http://YOUR_HA_IP:8123",
          "HA_TOKEN": "your-long-lived-token"
        }
      }
    }
  }
}
```

## 💬 Example Queries

### Complete System Health
- *"Run complete system diagnostic"*
- *"What's wrong with my Home Assistant?"*
- *"Give me a full health check"*

### Network & Zigbee Analysis
- *"Audit my Zigbee mesh network"*
- *"Show Zigbee health score and weak devices"*
- *"Which devices need better router placement?"*

### Smart Cleanup
- *"Find all orphan entities I can safely delete"*
- *"Detect automation conflicts and race conditions"*
- *"Show unused entities organized by domain"*

### Energy Monitoring
- *"Show energy consumption for last 24 hours"*
- *"Which devices consume the most energy?"*
- *"Estimate my monthly electricity cost"*

### Quick Diagnostics
- *"Which batteries are low?"*
- *"Show all unavailable entities"*
- *"Find sensors that haven't updated in 6 hours"*

### Automation Debugging
- *"Why isn't my kitchen automation working?"*
- *"Diagnose automation.low_battery_notifications"*
- *"Check for automation conflicts"*

### Device Management
- *"Identify the bedroom light"* (flashes the device)
- *"Which Zigbee devices have weak signal?"*
- *"List all offline devices"*

### System Monitoring
- *"Check error log for issues"*
- *"What integrations are loaded?"*
- *"Are there any available updates?"*

## 🏗️ Project Structure

```
Home-Assistant-Diagnostics-MCP-Server/
├── app/
│   ├── server.py       # FastMCP server (39 tools, 13 resources, 4 prompts)
│   ├── ha.py          # Home Assistant API client
│   ├── run.py         # Entry point
│   ├── config.py      # Configuration management
│   └── __main__.py    # Module execution
├── tests/
│   ├── __init__.py
│   └── test_smoke.py  # Smoke tests
├── .gitignore
├── LICENSE            # MIT License
├── README.md          # This file
├── pyproject.toml     # Project dependencies
└── uv.lock           # Dependency lock file
```

## 🤝 Contributing

Built for **Hugging Face MCP's 1st Birthday Hackathon 2025**

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Hugging Face** - MCP's 1st Birthday Hackathon 2025 organizers
- **FastMCP** - Excellent MCP framework

- **Home Assistant** - Amazing open-source home automation platform

## 🔗 Links

- [Home Assistant](https://www.home-assistant.io/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [Hugging Face Hackathon](https://huggingface.co/MCP-1st-Birthday)

---

**Built with ❤️ for the Home Assistant and AI community**

*Making smart homes smarter through AI-powered diagnostics*

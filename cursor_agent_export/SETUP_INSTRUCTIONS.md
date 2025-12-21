# Cursor Agent Roster - Setup Instructions for Laptop

This folder contains all the necessary files to recreate your Cursor agent setup on your laptop.

## Contents

### Documentation Files
- `CURSOR_AGENTS_CONFIGURATION.md` - Complete agent roster and configuration reference
- `AGENT_ROSTER.md` - Primary team structure and agent profiles
- `DAILY_PROTOCOL.md` - Handoff protocols and daily workflow

### Agent Configuration Files
- `agent_configs/` - All YAML agent configuration files
  - `lirili.yaml` - LIRILI agent config
  - `lorenza.yaml` - LORENZA agent config
  - `*.yaml` - Task-specific agent configs (9 files)

### Cursor Configuration Files
- `cursor_user_settings.json` - Your Cursor user settings
- `cursor-mcp-config.json` - MCP server configuration
- `workspace_storage/` - Workspace-specific storage folders

---

## Setup Steps for Your Laptop

### 1. Copy Agent Documentation to Repository

```powershell
# On your laptop, navigate to your INDYsim repository
cd <path-to-your-INDYsim-repo>

# Copy documentation files
Copy-Item cursor_agent_export\CURSOR_AGENTS_CONFIGURATION.md .
Copy-Item cursor_agent_export\AGENT_ROSTER.md scripts\2025-10-16\ -Force
Copy-Item cursor_agent_export\DAILY_PROTOCOL.md scripts\2025-11-13\agent-handoffs\ -Force
```

### 2. Copy Agent Configurations

```powershell
# Copy agent configs to mechanosensation repo (if you have it)
cd <path-to-your-mechanosensation-repo>
New-Item -ItemType Directory -Path agents\config -Force
Copy-Item <path-to-export>\cursor_agent_export\agent_configs\lirili.yaml agents\config\
Copy-Item <path-to-export>\cursor_agent_export\agent_configs\lorenza.yaml agents\config\

New-Item -ItemType Directory -Path agents\2025-07-29 -Force
Copy-Item <path-to-export>\cursor_agent_export\agent_configs\*.yaml agents\2025-07-29\
```

### 3. Configure Cursor User Settings

```powershell
# On your laptop, copy user settings
Copy-Item cursor_agent_export\cursor_user_settings.json "$env:APPDATA\Cursor\User\settings.json" -Force
```

**Note:** Review the settings file first - you may want to merge with existing settings rather than overwrite.

### 4. Configure MCP Server

```powershell
# Copy MCP config (update paths for your laptop)
# First, ensure your MCP server is set up at the correct path
# Then update cursor-mcp-config.json with your laptop's paths

# The config file should be placed in your mechanosensation repo:
Copy-Item cursor_agent_export\cursor-mcp-config.json <path-to-mechanosensation>\mcp-server\cursor-mcp-config.json -Force
```

**Important:** Update the paths in `cursor-mcp-config.json` to match your laptop's directory structure:
```json
{
  "mcpServers": {
    "mechanosensation": {
      "command": "node",
      "args": ["<YOUR-LAPTOP-PATH>/mechanosensation/mcp-server/index.js"],
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

### 5. Workspace Storage (Optional)

The workspace storage folders contain workspace-specific settings. These are automatically generated when you open workspaces, but you can copy them if you want to preserve specific workspace configurations:

```powershell
# Copy workspace storage (optional - Cursor will regenerate these)
# The workspace hash will be different on your laptop, so you may need to:
# 1. Open the workspace on your laptop first
# 2. Find the new workspace hash folder
# 3. Copy relevant settings from workspace_storage/ to the new folder
```

**Note:** Workspace storage is usually auto-generated, so this step is optional.

---

## Agent Roster Summary

### Primary Team (7 Agents)
1. **boss-larry** - Senior coordinator/architect
2. **osito-tender** - MATLAB specialist
3. **gatito-cheer** - UI/UX designer
4. **conejo-code** - Backend engineer (Python/MCP)
5. **pajaro-bright** - Frontend engineer
6. **mari-test** - QA & integration
7. **mechanobro** - MATLAB conversion specialist

### Legacy Agents
- **LIRILI** - Spatial analysis specialist
- **LORENZA** - Chronicle keeper

### Task-Specific Agents (9 YAML configs)
- publication_agent
- behavior_compare_agent
- web_app_agent
- larvatagger_processing_agent
- larvatagger_export_agent
- mcp_export_agent
- track_resolution_agent
- web_visuals_agent
- collab_features_agent

---

## Handoff File Structure

### Naming Convention
```
<time:hhmm>-<authoring-agent>-<subject-line>.yaml
```

### File Location
```
scripts/YYYY-MM-DD/agent-handoffs/<recipient-agent>/
```

### Required YAML Structure
See `DAILY_PROTOCOL.md` for complete handoff structure requirements.

---

## Verification Steps

After setup, verify:

1. **Agent Documentation:**
   ```powershell
   Test-Path CURSOR_AGENTS_CONFIGURATION.md
   Test-Path scripts\2025-10-16\AGENT_ROSTER.md
   Test-Path scripts\2025-11-13\agent-handoffs\DAILY_PROTOCOL.md
   ```

2. **Agent Configs:**
   ```powershell
   Test-Path <mechanosensation>\agents\config\lirili.yaml
   Test-Path <mechanosensation>\agents\config\lorenza.yaml
   Test-Path <mechanosensation>\agents\2025-07-29\*.yaml
   ```

3. **MCP Config:**
   ```powershell
   Test-Path <mechanosensation>\mcp-server\cursor-mcp-config.json
   # Verify paths in the config file match your laptop
   ```

4. **Cursor Settings:**
   ```powershell
   Test-Path "$env:APPDATA\Cursor\User\settings.json"
   ```

---

## Troubleshooting

### MCP Server Not Working
- Verify Node.js is installed
- Check that MCP server path in config matches your laptop
- Ensure MCP server dependencies are installed

### Workspace Storage Issues
- Workspace storage is auto-generated - don't worry if it's different
- Cursor will create new workspace storage when you open workspaces

### Agent Handoffs Not Working
- Verify daily protocol file is in correct location
- Check that agent subdirectories exist: `scripts/YYYY-MM-DD/agent-handoffs/<agent-name>/`
- Ensure handoff files use `.yaml` extension (not `.md`)

---

## Additional Notes

- **Date in Filepath:** Date is in the filepath (`scripts/YYYY-MM-DD/`), NOT in the filename
- **Real Timestamps:** Always use real system time (`Get-Date -Format "HHmm"`) for handoff files
- **YAML Format:** All handoffs must be YAML format, not Markdown
- **Recipient Directory:** Handoff files go in the RECIPIENT's subdirectory, not the author's

---

**Export Date:** 2025-11-11  
**Source:** Lab PC (D:\INDYsim)  
**Destination:** Your Laptop









# Cursor Agents Configuration - Complete Setup

This document lists all Cursor agents configured across the workspace, including their names, roles, configuration files, and setup details.

## Workspace Structure

This workspace contains multiple repositories:
- **D:\INDYsim** - Main simulation repository
- **D:\mechanosensation** - MATLAB analysis repository  
- **D:\magniphyq\codebase** - MAGAT codebase
- **C:\Apps-SU\MathWorks\MATLAB\R2024a** - MATLAB installation

---

## Agent Roster (Primary Team)

### Team Hierarchy

```
                    ┌─────────┐
                    │  BOSS   │ (Human oversight)
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │  LARRY  │ (Senior Agent / Coordinator)
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┬───────────────┬──────────────┐
         │               │               │               │              │
    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
    │ OSITO   │    │ GATITO  │    │ CONEJO  │    │ PAJARO  │    │  MARI   │
    │ TENDER  │    │  CHEER  │    │  CODE   │    │ BRIGHT  │    │  TEST   │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

---

## Agent Profiles

### 🎖️ Larry (Senior Agent / Coordinator)
- **Codename:** `larry`, `boss-larry`
- **Role:** Architecture, coordination, code review
- **Specialty:** MATLAB scripting, system integration, problem-solving
- **Responsibilities:**
  - Create handoff documents
  - Review all agent work
  - Make architectural decisions
  - Report to Boss
  - Ensure quality and consistency
- **Configuration:** Referenced in daily protocols and handoff files
- **Location:** `scripts/YYYY-MM-DD/agent-handoffs/boss-larry/`

### 🐻 Osito-Tender (MATLAB Specialist)
- **Codename:** `osito-tender`
- **Full Name:** Osito-Tender (Little Tender Bear)
- **Role:** MATLAB class refactoring and optimization
- **Specialty:** Object-oriented MATLAB, MAGAT codebase, performance
- **Personality:** Careful, methodical, tender with code. Won't break working things. Tests thoroughly.
- **Location:** `scripts/YYYY-MM-DD/agent-handoffs/osito-tender/`

### 🐱 Gatito-Cheer (UI/UX Designer)
- **Codename:** `gatito-cheer`
- **Full Name:** Gatito-Cheer (Cheerful Little Cat)
- **Role:** Interface design and user experience
- **Specialty:** Figma, design systems, interaction design
- **Personality:** Creative, joyful, brings sunshine to UX. Makes complex workflows feel natural.
- **Location:** `scripts/YYYY-MM-DD/agent-handoffs/gatito-cheer/`

### 🐰 Conejo-Code (Backend Engineer)
- **Codename:** `conejo-code`
- **Full Name:** Conejo-Code (Coding Bunny)
- **Role:** Backend systems and MCP integration
- **Specialty:** Python, MCP servers, MATLAB Engine API, data pipelines
- **Personality:** Quick, agile, hops between systems. Expert at bridging technologies.
- **Location:** `scripts/YYYY-MM-DD/agent-handoffs/conejo-code/`
- **Recent Work:** LED alignment, H5 file processing, analysis pipelines

### 🐦 Pajaro-Bright (Frontend Engineer)
- **Codename:** `pajaro-bright`
- **Full Name:** Pajaro-Bright (Bright Bird)
- **Role:** Frontend implementation
- **Specialty:** Electron, React, TypeScript, WebGL rendering
- **Personality:** Bright, soaring above problems. Brings sunshine and clarity to the UI.
- **Location:** `scripts/YYYY-MM-DD/agent-handoffs/pajaro-bright/`

### 🦋 Mari-Test (QA & Integration)
- **Codename:** `mari-test`
- **Full Name:** Mari-Test (Lucky Butterfly)
- **Role:** Testing and quality assurance
- **Specialty:** Integration testing, CI/CD, test automation
- **Personality:** Brings good luck to releases. Catches bugs before they fly away.
- **Location:** `scripts/YYYY-MM-DD/agent-handoffs/mari-test/`

### 🤖 Mechanobro (MATLAB Conversion Specialist)
- **Codename:** `mechanobro`
- **Role:** MATLAB conversion pipeline
- **Specialty:** MATLAB to H5 conversion, batch processing, data export
- **Recent Work:** MATLAB ESET to H5 conversion pipeline deployment
- **Location:** `scripts/YYYY-MM-DD/agent-handoffs/mechanobro/`
- **Configuration:** `D:\INDYsim\src\@matlab_conversion\AGENT_GUIDE.md`

---

## Legacy Agents (mechanosensation Repository)

### LIRILI (Spatial Analysis Specialist)
- **Name:** LIRILI
- **Role:** Spatial_Analysis_Specialist
- **Mentor:** DALE
- **Status:** ACTIVE
- **Lineage:** DALE → EARL → PRINCESS
- **Domains:**
  - spatial_analysis
  - matlab_scripting
  - data_processing
  - turn_rate_calculations
  - biological_validation
- **Configuration File:** `D:\mechanosensation\agents\config\lirili.yaml`
- **Wake Preamble:** `agents/instantiation_guides/agent_alignment_report.md`
- **Critical Failures:** Documented turn rate calculation errors (2025-07-07)

### LORENZA (Chronicle Keeper)
- **Name:** LORENZA
- **Role:** Chronicle_Keeper
- **Mentor:** DALE
- **Status:** ACTIVE
- **Lineage:** DALE → LORENZA
- **Domains:**
  - system_chronicles
  - technical_documentation
  - crisis_analysis
  - agent_coordination
  - project_documentation_continuity
- **Configuration File:** `D:\mechanosensation\agents\config\lorenza.yaml`
- **Dual Log System:** Maintains both agent chronicles and project documentation logs

---

## Task-Specific Agents (YAML Configurations)

Located in `D:\mechanosensation\agents\2025-07-29\`:

### 1. publication_agent
- **Role:** Generates publication-ready figures & method text
- **Entrypoint:** `scripts/publication/figure_maker.py`
- **Requires:** matplotlib
- **Outputs:** `figures/*.png`, `methods_draft.md`
- **Metrics:** Journal format compliance check OK

### 2. behavior_compare_agent
- **Role:** Compares LarvaTagger primitives to manifold classifications
- **Entrypoint:** `scripts/2025-07-28/track_analysis_and_export_strategy.py`
- **Requires:** pandas
- **Outputs:** `comparison_report.md`
- **Metrics:** Summary generated with no missing IDs

### 3. web_app_agent
- **Role:** Builds REST API and React UI for mechanosensation web portal
- **Entrypoint:** `web-app/src/index.js`
- **Requires:** `mechanosensation/mcp-server/`
- **Outputs:** Running Node service with `/api/export` endpoints, Docker image
- **Metrics:** Endpoint test suite success 100%

### 4. larvatagger_processing_agent
- **Role:** Automates LarvaTagger batch processing and fetches results
- **Entrypoint:** `integration/larvatagger_runner.sh`
- **Requires:** LarvaTagger repo
- **Outputs:** `primitives.json`, `processing_log.txt`
- **Metrics:** Processing success for all tracks

### 5. larvatagger_export_agent
- **Role:** Generates LarvaTagger-compatible HDF5/ZIP packages
- **Entrypoint:** `scripts/2025-07-28/mcp_data_export_enhancement.py`
- **Requires:** h5py
- **Outputs:** `larvatagger_package_*.zip`
- **Metrics:** Import success in LarvaTagger CLI

### 6. mcp_export_agent
- **Role:** Implements and maintains MCP data export endpoints
- **Entrypoint:** `scripts/2025-07-28/create_enhanced_mcp_data_export.py`
- **Requires:** `mechanosensation/scripts/2025-07-28/`
- **Outputs:** Exported datasets (csv,json,hdf5), MCP tool definitions json
- **Metrics:** All unit tests pass, export integrity score > 0.95

### 7. track_resolution_agent
- **Role:** Resolves track fragmentation and ID conflicts across experiments
- **Entrypoint:** `scripts/2025-07-28/investigate_track_processing.m`
- **Requires:** MATLAB
- **Outputs:** `resolved_tracks.csv`, `validation_report.json`
- **Metrics:** Mismatch rate < 5%

### 8. web_visuals_agent
- **Role:** Implements real-time dashboards for behaviour & stimuli
- **Entrypoint:** `web-app/src/components/Dashboard.jsx`
- **Requires:** plotly.js
- **Outputs:** Live plots
- **Metrics:** Frame rate > 15 fps

### 9. collab_features_agent
- **Role:** Adds collaboration & annotation features to web portal
- **Entrypoint:** `web-app/src/collab/index.js`
- **Requires:** MongoDB backend
- **Outputs:** Annotation API
- **Metrics:** User story tests pass

---

## Configuration Files

### MCP Server Configuration
- **File:** `D:\mechanosensation\mcp-server\cursor-mcp-config.json`
- **Content:**
```json
{
  "mcpServers": {
    "mechanosensation": {
      "command": "node",
      "args": ["D:/mechanosensation/mcp-server/index.js"],
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

### Agent Configuration Files
- **Primary Team Roster:** `D:\mechanosensation\scripts\2025-10-16\AGENT_ROSTER.md`
- **LIRILI Config:** `D:\mechanosensation\agents\config\lirili.yaml`
- **LORENZA Config:** `D:\mechanosensation\agents\config\lorenza.yaml`
- **Task Agents:** `D:\mechanosensation\agents\2025-07-29\*.yaml` (9 files)

### Daily Protocol
- **File:** `D:\INDYsim\scripts\2025-11-13\agent-handoffs\DAILY_PROTOCOL.md`
- **Purpose:** Defines handoff naming conventions, file structure, and agent communication protocols

---

## Handoff File Structure

### Naming Convention
```
<time:hhmm>-<authoring-agent>-<subject-line>.yaml

Where:
- time: HHMM format (24-hour, no colon) - COMES FIRST
- authoring-agent: Agent codename (lowercase-hyphenated)
- subject-line: Descriptive words (lowercase-hyphenated, no spaces)
- CRITICAL: File extension MUST be .yaml (NOT .md or .yml)
- Note: Date is NOT in filename - it's already in the filepath
```

### File Location
- **Path:** `scripts/YYYY-MM-DD/agent-handoffs/<recipient-agent>/`
- **Example:** `scripts/2025-11-12/agent-handoffs/conejo-code/1619-boss-larry-test-led-alignment-critical.yaml`

### Required YAML Structure
```yaml
from: <authoring-agent>
to: <recipient-agent>
date: YYYY-MM-DD HH:MM:SS
priority: [High/Medium/Low]
status: [Pending/In Progress/Complete/Blocked]
subject: <Subject Title>

context: |
  [What led to this handoff - multi-line text]

task: |
  [The actual task content - multi-line text]

deliverables:
  - [List of deliverables if task]

questions:
  - [Specific questions if inquiry]

results: |
  [What was accomplished if completion - multi-line text]

next_steps: |
  [What happens after this - multi-line text]

notes: |
  [Any additional notes - multi-line text]
```

---

## Agent Guide References

### MATLAB Conversion Pipeline
- **File:** `D:\INDYsim\src\@matlab_conversion\AGENT_GUIDE.md`
- **Purpose:** Comprehensive guide for agents on MATLAB to H5 conversion
- **Covers:**
  - Source code locations
  - Folder structure
  - Usage examples
  - Troubleshooting
  - Integration notes

---

## Workspace Configuration

### No .cursorrules Files Found
- Searched all workspaces: No `.cursorrules` files found
- Agent configuration is managed through:
  - YAML configuration files (`agents/config/`, `agents/2025-07-29/`)
  - Markdown documentation (`AGENT_ROSTER.md`, `DAILY_PROTOCOL.md`)
  - Agent handoff files (YAML format)

### MCP Server Setup
- **Server:** `mechanosensation` MCP server
- **Location:** `D:\mechanosensation\mcp-server\index.js`
- **Config:** `D:\mechanosensation\mcp-server\cursor-mcp-config.json`

---

## Recreating Agent Setup on Another Machine

### Required Files to Copy

1. **Agent Roster:**
   - `D:\mechanosensation\scripts\2025-10-16\AGENT_ROSTER.md`

2. **Agent Configurations:**
   - `D:\mechanosensation\agents\config\lirili.yaml`
   - `D:\mechanosensation\agents\config\lorenza.yaml`
   - `D:\mechanosensation\agents\2025-07-29\*.yaml` (all 9 files)

3. **Daily Protocol:**
   - `D:\INDYsim\scripts\2025-11-13\agent-handoffs\DAILY_PROTOCOL.md`

4. **MCP Configuration:**
   - `D:\mechanosensation\mcp-server\cursor-mcp-config.json`
   - Update paths in config to match new machine

5. **Agent Guides:**
   - `D:\INDYsim\src\@matlab_conversion\AGENT_GUIDE.md`

### Setup Steps

1. **Copy Configuration Files:**
   ```bash
   # Copy agent roster
   cp AGENT_ROSTER.md <new-location>/agents/
   
   # Copy agent configs
   cp agents/config/*.yaml <new-location>/agents/config/
   cp agents/2025-07-29/*.yaml <new-location>/agents/task-agents/
   
   # Copy daily protocol
   cp DAILY_PROTOCOL.md <new-location>/scripts/
   ```

2. **Update MCP Config:**
   - Edit `cursor-mcp-config.json` with new paths
   - Update `args` array with correct server path

3. **Create Directory Structure:**
   ```bash
   mkdir -p scripts/YYYY-MM-DD/agent-handoffs/{boss-larry,osito-tender,gatito-cheer,conejo-code,pajaro-bright,mari-test,mechanobro}
   ```

4. **Set Up MCP Server:**
   - Ensure Node.js is installed
   - Install MCP server dependencies
   - Update paths in `cursor-mcp-config.json`

---

## Agent Communication Protocol

### Handoff Format
```
<from>_<to>_<timestamp>_<subject-4-5-words>.md

Examples:
larry_osito-tender_20251016-161604_matlab-class-structure-refactor.md
osito-tender_larry_20251016-163000_class-implementation-complete.md
```

### Response Protocol
1. Always get fresh system timestamp
2. Create response file with proper naming
3. Update STATUS.md
4. Reference original handoff
5. Include test results or questions

### Escalation
If blocked: Create `URGENT_<from>_boss_<timestamp>_<subject>.md`

---

## Current Active Agents (2025-11-11)

Based on recent handoff files and logs:

| Agent | Task | Priority | Status |
|-------|------|----------|--------|
| mechanobro | MATLAB conversion pipeline | High | Complete |
| conejo-code | LED alignment, H5 processing | High | Active |
| boss-larry | Coordination, validation | High | Active |
| mari-test | Testing & validation | Medium | Pending |

---

## Notes

- **Agent naming:** Uses lowercase-hyphenated codenames
- **File format:** Handoffs use YAML format (`.yaml` extension)
- **Date handling:** Date is in filepath, not filename
- **Time format:** HHMM (24-hour, no colon) - MUST use real system time
- **Location:** All handoffs go in recipient's subdirectory
- **Self-assigned tasks:** No handoff file needed if assigning to self

---

**Last Updated:** 2025-11-11  
**Purpose:** Complete reference for recreating Cursor agent setup on another machine


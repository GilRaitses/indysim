# Daily Agent Handoff Protocol

## Daily Structure Setup

### Every New Day:

**1. Get System Date**
```bash
powershell -Command "Get-Date -Format 'yyyy-MM-dd'"
```

**2. Create Script Folder**
```bash
cd D:\mechanosensation\scripts
New-Item -ItemType Directory -Name "YYYY-MM-DD"
```

**3. Create Agent Handoffs Subfolder**
```bash
cd scripts/YYYY-MM-DD
New-Item -ItemType Directory -Name "agent-handoffs"
```

**4. Copy This Protocol**
```bash
cp ../previous-date/agent-handoffs/DAILY_PROTOCOL.md ./agent-handoffs/
```

**5. Create Agent Recipient Subdirectories**
```bash
cd scripts/YYYY-MM-DD/agent-handoffs
New-Item -ItemType Directory -Path "conejo-code","boss-larry","mari-test","osito-tender","gatito-cheer","mechanobro" -Force
```

**6. Create Work Tree**
```bash
# Create: docs/work-trees/YYYY-MM-DD-work-tree.md
# Include task allocations, parallelization strategy, and agent kickoff prompts
```

**7. Create Initial Handoff Files**
```bash
# After work tree is created, create handoff files in appropriate agent subdirectories
# CRITICAL: Files MUST go in recipient's subdirectory, NOT root folder
# Location: scripts/YYYY-MM-DD/agent-handoffs/<recipient-agent>/
# Naming: <time:hhmm>-<authoring-agent>-<subject>.yaml
# One handoff file per task assignment
# Example: scripts/2025-11-12/agent-handoffs/conejo-code/1619-boss-larry-test-led-alignment-critical.yaml

# IMPORTANT: Do NOT create handoff files for self-assigned tasks
# If boss-larry assigns a task to himself, just do it directly without creating a handoff file
# Handoffs are ONLY for delegating tasks to OTHER agents
```

**8. Create Daily Log**
```bash
# Create: docs/logs/YYYY-MM-DD.md
```

---

## File Naming Convention

**All handoff files:**
```
<time:hhmm>-<authoring-agent>-<subject-line>.yaml

Where:
- time: HHMM format (24-hour, no colon) - COMES FIRST
- authoring-agent: Agent codename (lowercase-hyphenated)
- subject-line: Descriptive words (lowercase-hyphenated, no spaces)
- **CRITICAL:** File extension MUST be `.yaml` (NOT `.md` or `.yml`)
- **Note:** Date is NOT in filename - it's already in the filepath (scripts/YYYY-MM-DD/agent-handoffs/...)
```

**Examples:**
```
0915-boss-larry-test-led-alignment-critical.yaml
1430-conejo-code-test-results-all-passing.yaml
1619-boss-larry-path-cleanup-parallel.yaml
```

**CRITICAL - Use REAL System Time:**
- **NEVER use fake or placeholder timestamps**
- **ALWAYS use actual system time:** `Get-Date -Format "HHmm"` for time
- **Run this command in PowerShell to get the real timestamp before creating handoff files**
- **Note:** Date is NOT needed in filename - it's already in the filepath (scripts/YYYY-MM-DD/agent-handoffs/...)
- Place handoff files in recipient agent's subdirectory: `scripts/YYYY-MM-DD/agent-handoffs/<recipient-agent>/`
- All worker agents must follow this naming convention
- **If creating multiple handoffs at once, use the same timestamp for all (they're created at the same time)**

---

## Handoff File Location - CRITICAL

**BEFORE creating any handoff file, remember:**

1. **Identify the recipient** (who you're sending the handoff TO)
2. **Create the file in the recipient's subdirectory:**
   - Path: `scripts/YYYY-MM-DD/agent-handoffs/<recipient-agent>/`
   - Example: If sending to boss-larry, create in `scripts/2025-11-12/agent-handoffs/boss-larry/`
   - **CRITICAL:** If replying to boss-larry, file MUST go in `boss-larry/` folder
3. **Get REAL current time:**
   - Run `Get-Date -Format "HHmm"` in PowerShell
   - **NEVER use fake or placeholder timestamps**
   - Use the same timestamp for all handoffs created at the same time
4. **Use the new naming convention:**
   - Format: `<time>-<your-agent-name>-<subject>.yaml`
   - Example: `1650-conejo-code-report-complete.yaml`
   - **CRITICAL:** File extension MUST be `.yaml` (NOT `.md`)
   - **CRITICAL:** Date is NOT in filename - it's already in the filepath
   - **CRITICAL:** Time comes FIRST, then author name
5. **DO NOT use old naming convention or place in root folder**

---

## Required Handoff Structure

**CRITICAL:** All handoff files MUST be in YAML format (`.yaml` extension), NOT Markdown.

Every handoff must include the following YAML structure:

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

**YAML Formatting Rules:**
- Use `|` for multi-line text blocks (preserves line breaks)
- Use `-` for lists
- Use proper YAML indentation (2 spaces)
- All text fields support multi-line content with `|`

---

## Daily Log Requirements

**File:** `docs/logs/YYYY-MM-DD.md`

**Must include:**
- Objective for the day
- Active agents and their tasks
- Completed work from previous day
- Current blockers
- Milestone status
- Next steps

**Template:**
```markdown
# Daily Log - Month DD, YYYY

## Objective
[What we're trying to accomplish today]

## Carryover from [Previous Date]
[Summary of where we left off]

## Active Agents
| Agent | Task | Status | Handoff |
|-------|------|--------|---------|
| ...   | ...  | ...    | ...     |

## Progress Today
[What got done]

## Blockers
[What's preventing progress]

## Milestone Status
M1: [status]
M2: [status]
...

## Next Steps
[What's next]

---

**Status:** [summary]  
**Next Session:** [what to tackle]
```

---

## Work Tree and Initial Handoffs

**After creating work tree:**
1. Review work tree for all task assignments
2. Create handoff file for each task in the recipient agent's subdirectory
3. Use naming convention: `<time:hhmm>-<authoring-agent>-<subject-line>.yaml`
4. Include all required handoff structure sections
5. Reference work tree task details in handoff

**Example workflow:**
- Work tree assigns Task 0.1 to conejo-code
- Get real timestamp: `$time = Get-Date -Format "HHmm"` (e.g., "1619" - actual system time)
- Create: `scripts/YYYY-MM-DD/agent-handoffs/conejo-code/$time-boss-larry-test-led-alignment-critical.yaml`
- **Note:** Date is NOT in filename - it's already in the filepath
- Include task details, success criteria, deliverables from work tree
- **NEVER use placeholder times like "1700", "1616", or any fake timestamps - ALWAYS get the real time from the system using `Get-Date -Format "HHmm"`**

---

## Agent Roster (Current)

**Senior:**
- 🎖️ **boss-larry** - Architecture, coordination, review

**Specialists:**
- 🐻 **osito-tender** - MATLAB class structure
- 🐱 **gatito-cheer** - Figma UI/UX design
- 🐰 **conejo-code** - Python/MCP backend
- 🐦 **pajaro-bright** - Electron frontend
- 🦋 **mari-test** - Testing & QA
- 🤖 **mechanobro** - MATLAB conversion pipeline

---

## Handoff Naming Convention (All Agents)

**CRITICAL: All worker agents must follow this naming convention for handoff files.**

**Format:**
```
<authoring-agent>-<time:hhmm>-<date:yyyymmdd>-<subject-line>.md
```

**Rules:**
1. **authoring-agent**: Your agent codename (lowercase-hyphenated)
2. **time**: Use `Get-Date -Format "HHmm"` (24-hour format, no colon)
3. **date**: Use `Get-Date -Format "yyyyMMdd"`
4. **subject-line**: Descriptive words (lowercase-hyphenated, no spaces)

**CRITICAL - File Location:**
- **MUST place handoff files in the RECIPIENT agent's subdirectory**
- **Path:** `scripts/YYYY-MM-DD/agent-handoffs/<recipient-agent>/`
- **DO NOT place files in the root `agent-handoffs/` folder**
- **DO NOT use old naming convention (`author_recipient_timestamp_subject.md`)**

**Examples:**
- `0915-boss-larry-test-led-alignment-critical.yaml` → goes in `conejo-code/` subdirectory
- `1430-conejo-code-test-results-all-passing.yaml` → goes in `boss-larry/` subdirectory
- `1619-boss-larry-path-cleanup-parallel.yaml` → goes in `conejo-code/` subdirectory

**When creating handoffs:**
1. **Get REAL current time:** Run `Get-Date -Format "HHmm"` in PowerShell (do NOT make up a time)
2. **Determine recipient agent** (who you're sending the handoff TO)
3. **Create file in recipient's subdirectory:** `scripts/YYYY-MM-DD/agent-handoffs/<recipient-agent>/`
4. **Use new naming convention:** `<time>-<your-agent-name>-<subject>.yaml`
5. Use descriptive subject line (lowercase-hyphenated)
6. **If creating multiple handoffs at the same time, use the same timestamp for all**
7. **Note:** Date is NOT needed in filename - it's already in the filepath

**Example workflow:**
```powershell
# Get actual timestamp (date is in filepath, not filename)
$time = Get-Date -Format "HHmm"  # e.g., "1622"
$recipient = "boss-larry"  # Who you're sending TO

# Create handoff file in RECIPIENT's subdirectory with new naming convention
New-Item -Path "scripts/2025-11-12/agent-handoffs/$recipient/$time-conejo-code-test-complete.yaml"
```

**Common Mistakes to Avoid:**
- ❌ Placing file in root `agent-handoffs/` folder
- ❌ Using old naming convention (`conejo-code_larry_20251112-subject.md`)
- ❌ Using fake timestamps
- ❌ Missing recipient subdirectory
- ❌ Including date in filename (date is already in filepath)
- ❌ Using `.md` extension instead of `.yaml`
- ✅ Correct: `scripts/2025-11-12/agent-handoffs/boss-larry/1622-conejo-code-test-complete.yaml`

**CRITICAL - Replying to boss-larry:**
- When replying to boss-larry, ALWAYS place the handoff file in `boss-larry/` subdirectory
- Use format: `<time>-<your-agent-name>-<subject>.yaml`
- Get REAL time using `Get-Date -Format "HHmm"` (do NOT use fake timestamps)
- Do NOT include date in filename (date is already in filepath: `scripts/YYYY-MM-DD/agent-handoffs/`)
- File MUST be `.yaml` format, NOT `.md`
- Example: If conejo-code is replying to boss-larry at 16:50, create: `scripts/2025-11-12/agent-handoffs/boss-larry/1650-conejo-code-report-complete.yaml`

---

## Status Tracking

**File:** `agent-handoffs/STATUS.md` (updated daily)

Track all active handoffs:
```markdown
| From | To | Subject | Status | File |
|------|----|---------
|--------|------|
| ...  | ...| ...     | ...    | ...  |
```

---

## Cross-Day References

**Referencing previous days:**
```markdown
See: scripts/2025-10-16/agent-handoffs/larry_osito-tender_...
```

**Continuing work:**
```markdown
Continued from: 2025-10-16
Previous handoff: larry_osito-tender_20251016-161604_...
```

---

## Development Workflow: Self-Contained Scripts Folder Structure

### Concept
All daily work including scripts, analysis, outputs, and reports must be developed within the daily scripts folder (`scripts/YYYY-MM-DD/`) as a self-contained unit. This structure enables easy archiving and reuse as templates.

### Development Structure

**During Development:**
```
scripts/YYYY-MM-DD/
├── agent-handoffs/
│   └── [handoff documents]
├── report/
│   ├── report_YYYY-MM-DD.qmd
│   ├── report_YYYY-MM-DD.pdf
│   ├── figures/ (if any)
│   └── data/ (if any)
├── [analysis scripts]
│   ├── script1.py
│   ├── script2.py
│   └── ...
├── [output folders]
│   ├── output1/
│   ├── output2/
│   └── ...
└── [other work subdirectories]
```

### Key Principles

1. **Self-Containment:** All scripts, outputs, and report materials must be within `scripts/YYYY-MM-DD/`
2. **Report Location:** Reports go in `scripts/YYYY-MM-DD/report/` subdirectory
3. **Organization:** Keep work organized in logical subdirectories within the daily folder
4. **Relative Paths:** Scripts should use relative paths referencing other files within the daily folder
5. **Archive Ready:** Structure should be ready for copying to `docs/reports/YYYY-MM-DD/` as complete archive

### Report Development

**Location:** `scripts/YYYY-MM-DD/report/`

**Requirements:**
- QMD file with proper YAML header
- Style rules from `docs/style_rules.yaml` strictly followed
- Cinnamoroll color palette (reference: `scripts/cinnamoroll_palette.py`)
- Avenir Ultralight font configuration
- Self-contained: all figures and data references work within the folder structure

**Report Structure:**
- Title page with date and project name
- Executive summary
- Analysis sections (incorporating findings from other tasks)
- Conclusions and next steps
- All supporting materials (figures, data) in report subdirectory

### Archive Process

**When Work is Complete:**
1. Copy entire `scripts/YYYY-MM-DD/` folder to `docs/reports/YYYY-MM-DD/`
2. Include `docs/style_rules.yaml` in archive for reference
3. Archive becomes reusable template with:
   - All scripts needed to regenerate analysis
   - All output folders with results
   - Complete report with figures and data
   - Style rules for consistency

**Archive Structure:**
```
docs/reports/YYYY-MM-DD/
├── report/
│   ├── report_YYYY-MM-DD.qmd
│   ├── report_YYYY-MM-DD.pdf
│   ├── figures/
│   └── data/
├── [analysis scripts]
├── [output folders]
├── agent-handoffs/
└── style_rules.yaml (copied for reference)
```

### Benefits

1. **Reproducibility:** Complete archive contains everything needed to regenerate results
2. **Template Reuse:** Previous day's structure can be copied as starting point
3. **Organization:** Clear separation between development (scripts/) and documentation (docs/)
4. **Self-Containment:** No broken references when archive is moved or shared
5. **Traceability:** All work for a day stays together with handoffs and reports

### Example Workflow

1. **Morning:** Create `scripts/2025-11-11/` folder structure
2. **Development:** All scripts, analysis, outputs created within `scripts/2025-11-11/`
3. **Report:** Report developed in `scripts/2025-11-11/report/` referencing work in same folder
4. **End of Day:** Copy complete `scripts/2025-11-11/` to `docs/reports/2025-11-11/` with style_rules.yaml
5. **Archive:** Complete self-contained report archive ready for reuse

---

## Rules

1. ✅ Always get fresh system timestamp
2. ✅ Follow exact naming convention
3. ✅ Update STATUS.md
4. ✅ Create daily log
5. ✅ Reference related work
6. ✅ Clear deliverables
7. ✅ No hallucinated timestamps!
8. ✅ Keep all daily work self-contained in `scripts/YYYY-MM-DD/`
9. ✅ Develop reports in `scripts/YYYY-MM-DD/report/` subdirectory
10. ✅ Structure for eventual archive to `docs/reports/YYYY-MM-DD/`

---

**Last Updated:** 2025-11-11  
**Purpose:** Maintain clean, traceable agent communication across days and self-contained development workflow

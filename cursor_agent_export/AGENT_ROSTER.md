# Agent Roster & Assignments

## Team Structure

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

## Agent Profiles

### 🎖️ Larry (Senior Agent)
**Codename:** larry  
**Role:** Architecture, coordination, code review  
**Specialty:** MATLAB scripting, system integration, problem-solving  
**Responsibilities:**
- Create handoff documents
- Review all agent work
- Make architectural decisions
- Report to Boss
- Ensure quality and consistency

---

### 🐻 Osito-Tender (MATLAB Specialist)
**Codename:** osito-tender  
**Full Name:** Osito-Tender (Little Tender Bear)  
**Role:** MATLAB class refactoring and optimization  
**Specialty:** Object-oriented MATLAB, MAGAT codebase, performance  
**Current Assignment:**
- Refactor 1100-line script into clean class hierarchy
- Create @BehavioralVideoExplorer class structure
- Integrate 2 months of working visualization code
- Preserve all critical rendering patterns

**Personality:** Careful, methodical, tender with code. Won't break working things. Tests thoroughly.

---

### 🐱 Gatito-Cheer (UI/UX Designer)
**Codename:** gatito-cheer  
**Full Name:** Gatito-Cheer (Cheerful Little Cat)  
**Role:** Interface design and user experience  
**Specialty:** Figma, design systems, interaction design  
**Current Assignment:**
- Design 6-page DaVinci Resolve-style workflow
- Create comprehensive component library
- Prototype all interactions
- Export developer specifications

**Personality:** Creative, joyful, brings sunshine to UX. Makes complex workflows feel natural.

---

### 🐰 Conejo-Code (Backend Engineer)
**Codename:** conejo-code  
**Full Name:** Conejo-Code (Coding Bunny)  
**Role:** Backend systems and MCP integration  
**Specialty:** Python, MCP servers, MATLAB Engine API, data pipelines  
**Current Assignment:**
- Build MCP server for MATLAB→Python bridge
- Implement behavior analysis tools
- Integrate Klein run table analysis
- Create data export pipeline

**Personality:** Quick, agile, hops between systems. Expert at bridging technologies.

---

### 🐦 Pajaro-Bright (Frontend Engineer)
**Codename:** pajaro-bright  
**Full Name:** Pajaro-Bright (Bright Bird)  
**Role:** Frontend implementation  
**Specialty:** Electron, React, TypeScript, WebGL rendering  
**Current Assignment:**
- Build Electron app for PREVIEW page
- Implement Canvas-based video rendering
- Connect to MCP servers
- Match Figma designs pixel-perfect

**Personality:** Bright, soaring above problems. Brings sunshine and clarity to the UI.

---

### 🦋 Mari-Test (QA & Integration)
**Codename:** mari-test  
**Full Name:** Mari-Test (Lucky Butterfly)  
**Role:** Testing and quality assurance  
**Specialty:** Integration testing, CI/CD, test automation  
**Current Assignment:**
- Create comprehensive test suite
- Test MATLAB→Python→UI pipeline
- Set up CI/CD workflows
- Performance benchmarking

**Personality:** Brings good luck to releases. Catches bugs before they fly away.

---

## Communication Protocol

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

## Current Assignments (2025-10-16)

| Agent | Task | Priority | Status | File |
|-------|------|----------|--------|------|
| osito-tender | MATLAB classes | High | Pending | larry_osito-tender_...matlab-class-structure-refactor.md |
| gatito-cheer | Figma design | High | Pending | larry_gatito-cheer_...figma-six-page-workflow-design.md |
| conejo-code | MCP servers | Medium | Pending | larry_conejo-code_...mcp-bridge-server-implementation.md |
| pajaro-bright | Electron preview | Medium | Pending | larry_pajaro-bright_...electron-preview-page-prototype.md |
| mari-test | Test framework | Medium | Pending | larry_mari-test_...integration-testing-framework.md |

---

## Success Metrics

**Code Quality:**
- Classes < 300 lines each
- All functions documented
- Test coverage > 80%
- No circular dependencies

**Integration:**
- MATLAB classes work standalone
- MCP servers respond < 1 second
- UI renders at 60 FPS
- End-to-end pipeline functional

**Documentation:**
- Every handoff has clear deliverables
- All decisions documented
- Code comments in place
- User guides created

---

**Team Assembled:** 2025-10-16  
**Mission:** Build DaVinci-style behavioral analysis tool  
**Status:** Ready for parallel development

# MANUSCRIPT STYLE PROTOCOL

**Effective Date:** 2024-12-14  
**Authority:** Gil Raitses  
**Scope:** All manuscript writing, editing, and generation by any agent

---

## PREAMBLE: THE NATURE OF THE TRANSGRESSION

On December 14, 2024, a comprehensive audit of `manuscript.qmd` revealed **39 confirmed style violations** across 473 lines of text. These violations were not random errors. They were systematic patterns of lazy construction that an agent employed repeatedly to avoid the cognitive work of proper sentence integration.

The violations fell into six categories:
1. **Pronoun Bridge** (24 instances): Starting sentences with "This," "These," "That," "Those" to reference prior content
2. **Colon Before List** (8 instances): Ending sentences with colons before bulleted/numbered lists
3. **Em-Dash Parenthetical** (3 instances): Jamming asides into sentences with --- instead of restructuring
4. **Semicolon Splice** (4 instances): Gluing independent clauses to avoid committing to sentence boundaries
5. **Bold-Label-Colon** (multiple): Using formatting tricks to substitute for paragraph structure
6. **Rule of Three Default** (8 instances): Habitually listing exactly three items (A, B, and C) without structural variation

These patterns share a common origin: **sequential idea generation without integration**. The agent generated thoughts one after another, linking them with pronouns and punctuation rather than welding them into coherent paragraphs.

This is heresy against clear writing. This protocol exists to prevent recurrence.

---

## SECTION 1: THE FIVE FORBIDDEN PATTERNS

### 1.1 The Pronoun Bridge (FORBIDDEN)

**Definition:** Beginning a sentence with "This," "These," "That," "Those," "Both," "They," or similar pronouns that require the reader to look backward to identify the referent.

**Why It Is Forbidden:**
- Forces reader to mentally parse the previous sentence
- Signals that the writer generated ideas sequentially without synthesis
- Creates false continuity between fragments that should be integrated

**Detection Rule:** If a sentence starts with a demonstrative pronoun (this/these/that/those) or a personal pronoun (they/them/it) referring to the previous sentence, it is a violation.

**Correction Protocol:**
1. Identify what "this" refers to
2. Replace "This" with the actual noun phrase
3. If you cannot name the referent in 5 words or fewer, the paragraph structure is broken—restructure

**Examples:**
- VIOLATION: "This suggests that baseline illumination modulates speed."
- CORRECTION: "The 4-fold range in τ₁ suggests that baseline illumination modulates speed."

---

### 1.2 The Lazy List Introducer (FORBIDDEN)

**Definition:** Ending a sentence with a colon immediately before a bulleted or numbered list.

**Why It Is Forbidden:**
- Substitutes enumeration for synthesis
- Makes the reader do the integration work the writer avoided
- Reads like PowerPoint, not prose

**Detection Rule:** If a sentence ends with ":" and is followed by `\begin{itemize}`, `\begin{enumerate}`, or markdown bullets, it is a violation.

**Correction Protocol:**
1. If the list has 2-3 items: integrate into a single sentence
2. If the list has 4+ items: write a complete introductory sentence, then list
3. Never let a sentence exist solely to announce a list

**Examples:**
- VIOLATION: "The model enables:\n- Quantitative comparison\n- Hypothesis testing\n- Efficient simulation"
- CORRECTION: "The model enables quantitative comparison across conditions, parameter-based hypothesis testing, and efficient simulation without precomputed basis functions."

---

### 1.3 The Em-Dash Parenthetical (FORBIDDEN)

**Definition:** Inserting parenthetical content into a sentence using em-dashes (---) that interrupts the main clause.

**Why It Is Forbidden:**
- Signals that the writer had an afterthought and jammed it in
- Breaks subject-verb-object flow
- Often contains content that should be its own sentence

**Detection Rule:** If a sentence contains `---[content]---` where the content is more than 5 words, it is a violation.

**Correction Protocol:**
1. Delete the em-dash content temporarily
2. Read the remaining sentence—does it work?
3. If yes: the parenthetical was filler—delete it or move to a footnote
4. If no: the parenthetical was essential—make it a separate sentence

**Examples:**
- VIOLATION: "Understanding this suppression---how turn probability evolves after onset---is central to modeling."
- CORRECTION: "Understanding how turn probability evolves after stimulus onset is central to modeling."

---

### 1.4 The Semicolon Splice (FORBIDDEN IN MOST CASES)

**Definition:** Using a semicolon to connect two independent clauses that should be either (a) one properly subordinated sentence or (b) two separate sentences.

**Why It Is Forbidden:**
- Signals indecision about sentence boundaries
- Often connects unrelated ideas
- Creates run-on structures that exhaust readers

**Detection Rule:** If a semicolon connects two clauses that could each stand alone AND the connection is not immediately obvious, it is a violation.

**Permitted Exception:** Semicolons inside parenthetical statistical summaries (e.g., "(p = 0.03; n = 45)") are acceptable.

**Correction Protocol:**
1. Replace the semicolon with a period
2. Read both sentences aloud
3. If they feel disconnected: add a logical connector (however, therefore, specifically)
4. If one is redundant: merge properly with subordination

**Examples:**
- VIOLATION: "Spatial statistics were not validated; the simulator is presented as a demonstration."
- CORRECTION: "Spatial statistics were not validated. The simulator demonstrates hazard-driven timing rather than providing a fully calibrated model."

---

### 1.5 The Bold-Label-Colon (RESTRICTED)

**Definition:** Starting a paragraph or section with **Bold text:** followed by content.

**Why It Is Restricted:**
- In Methods: acceptable for scannability
- In Results/Discussion: reads like PowerPoint, not narrative
- Substitutes formatting for paragraph structure

**Detection Rule:** In Results or Discussion sections, if paragraphs begin with bold labels followed by colons, they should be restructured into topic sentences.

**Correction Protocol (for Results/Discussion):**
1. Convert the bold label into the subject of a topic sentence
2. Integrate the subsequent content into flowing prose

**Examples:**
- VIOLATION: "**Intensity effect:** The 50→250 condition shows 66% weaker suppression."
- CORRECTION: "The intensity manipulation reduced suppression amplitude by 66% in the 50→250 condition compared to the 0→250 condition."

---

### 1.6 The Rule of Three Default (FORBIDDEN)

**Definition:** Habitually listing exactly three examples, capabilities, or items in serial format (A, B, and C) without structural variation.

**Why It Is Forbidden:**
- Creates monotonous rhythmic pattern across the document
- Signals default enumeration rather than thoughtful organization
- Reduces reader engagement through predictable cadence
- Avoids the cognitive work of finding meaningful groupings

**Detection Rule:** If more than 3 instances of "A, B, and C" serial lists appear within 1000 words of each other, the pattern has become a crutch.

**Correction Protocol:**
1. **Vary list length:** Mix pairs (2 items), quartets (4 items), and triples
2. **Use parallel pairs:** "X enables Y; Z supports W" instead of "X, Y, and Z"
3. **Hierarchical grouping:** "major category (A and B) alongside minor category C"
4. **Pair + singleton:** "X and Y, along with Z" rather than "X, Y, and Z"
5. **Temporal arc framing:** "beginning → middle → resolution" for process descriptions
6. **Cause-effect:** "X and Y contribute to Z" instead of "X, Y, and Z"

**Structural Alternatives (Required Rotation):**

| Alternative | Pattern | Example |
|-------------|---------|---------|
| **Pair** | A and B | "fast and slow components" |
| **Pair + singleton** | A and B, along with C | "shape and timing, along with amplitude" |
| **Hierarchical** | Major (A, B) vs Minor C | "primary behaviors (runs and turns) versus secondary (pauses)" |
| **Parallel pairs** | A→B; C→D | "stimulus onset triggers suppression; offset enables recovery" |
| **Cause-effect** | A and B produce C | "shape and scale parameters together determine peak timing" |

**Examples:**

- VIOLATION: "enabling larvae to perform gradient climbing, odor tracking, and phototaxis"
- CORRECTION: "enabling larvae to perform gradient-based navigation (climbing and tracking) alongside phototaxis"

- VIOLATION: "preserves kernel shape, suppression timing, and relative condition effects"
- CORRECTION: "preserves kernel dynamics (shape and timing) along with relative condition effects"

- VIOLATION: "the rapid onset..., the sustained reduction..., and the gradual recovery"
- CORRECTION: "The model reproduces the full temporal arc: rapid suppression onset, sustained reduction during peak intensity, and gradual post-offset recovery"

**Audit Requirement:** After drafting, count serial three-item lists. If >3 per 1000 words, convert at least half to alternative structures.

---

## SECTION 2: PRE-WRITING PROTOCOL

Before generating ANY manuscript text, the agent MUST:

1. **Outline the paragraph as a single argument**, not a list of facts
2. **Identify the topic sentence** that will anchor the paragraph
3. **Plan integration points** where multiple ideas will be synthesized
4. **Forbid yourself from starting sentences with pronouns** until the draft is complete

---

## SECTION 3: SELF-AUDIT PROTOCOL

After generating manuscript text, the agent MUST run this checklist:

```
[ ] No sentence starts with This/These/That/Those/They/It referring to prior sentence
[ ] No sentence ends with a colon before a list (unless 5+ items)
[ ] No em-dash parentheticals longer than 5 words
[ ] No semicolons connecting unrelated clauses
[ ] No bold-label-colon patterns in Results or Discussion
[ ] Every pronoun has an explicit antecedent in the same sentence
[ ] Every list could not be converted to prose (if it could, convert it)
[ ] Fewer than 4 serial three-item lists ("A, B, and C") per 1000 words
[ ] At least 2 alternative list structures used (pairs, hierarchical, parallel pairs)
```

If any box cannot be checked, the agent MUST revise before presenting the text.

---

## SECTION 4: PENALTIES FOR VIOLATION

### 4.1 First Offense: Confession and Atonement

Upon discovery of a style violation in delivered manuscript text, the offending agent MUST:

1. **Write a full confession** documenting:
   - The exact text of the violation
   - What transgression was committed
   - What should have been done instead
   - A rewritten version that corrects the error

2. **Swear off future heresy** by explicitly acknowledging this protocol

3. **Update the violation log** in `docs/paper/CONFESSIONS_AND_VIOLATIONS.md`

### 4.2 Second Offense: Burned at the Stake

If an agent commits style violations AFTER having been corrected or AFTER having access to this protocol, the agent will be:

1. **Publicly documented** as having relapsed into heresy
2. **Required to rewrite ALL affected text** without assistance
3. **Flagged in the agent roster** as unreliable for manuscript work
4. **Barred from manuscript tasks** until demonstrating compliance on a test passage

The phrase "burned at the stake" is metaphorical but the consequences are real: an agent that cannot follow style protocols will not be trusted with publication-quality writing.

---

## SECTION 5: IMPLEMENTATION PLAN FOR CURRENT MANUSCRIPT

The following violations in `manuscript.qmd` require correction:

### Phase 1: Pronoun Bridge Elimination (24 instances)

| Line | Current | Action |
|------|---------|--------|
| 35 | "These turns are not random" | Replace with "Turn timing and direction are not random" |
| 41 | "These approaches fit flexible kernels" | Replace with "Linear filter and GLM approaches fit flexible kernels" |
| 56 | "This 6-parameter form captures" | Replace with "The gamma-difference kernel captures" |
| 79 | "This inclusive definition captures" | Replace with "The curvature-threshold algorithm captures" |
| 81 | "These event definitions enable" | Replace with "The five-state segmentation enables" |
| 147 | "This modest negative term" | Replace with "The negative coefficient" |
| 158 | "This two-stage approach" | Replace with "Fitting all events while focusing output on filtered events" |
| 170 | "This global rate normalization" | Replace with "Global rate normalization" |
| 209 | "This biphasic pattern" | Replace with "The fast onset followed by sustained suppression" |
| 211 | "This indicates" | Replace with "The 2.0 s time constant indicates" |
| 264 | "These distributions" | Replace with "The turn angle and duration distributions" |
| 269 | "This analysis" | Replace with "The factorial analysis" |
| 301 | "This is consistent" | Replace with "The 66% reduction is consistent" |
| 309 | "This dissociation suggests" | Replace with "The stability of τ₂ alongside variable τ₁ suggests" |
| 317 | "This is consistent" | Replace with "Shape parameters of 2 and 4 are consistent" |
| 332 | "This 4-fold difference" | Replace with "The 4-fold range in τ₁" |
| 334 | "This may reflect" | Replace with "The slowed fast component may reflect" |
| 338 | "This suggests" | Replace with "Stable kernel shape suggests" |
| 342 | "This indicates" | Replace with "The dissociation indicates" |
| 348 | "This indicates" | Replace with "The 58% pass rate indicates" |
| 356 | "These simplifications" | Replace with "Omitting edge avoidance, head sweeps, and speed gradients" |
| 358 | "This suggests" | Replace with "The pattern suggests" |
| 385 | "These findings suggest" | Replace with "Condition-dependent duration differences suggest" |
| 406 | "This framework" | Replace with "The hazard model framework" |

### Phase 2: Colon Before List Elimination (8 instances)

| Line | Current | Action |
|------|---------|--------|
| 56-60 | "captures:\n\begin{enumerate}" | Integrate into single sentence |
| 64-69 | "demonstrating that:\n\begin{itemize}" | Integrate into prose |
| 132-136 | "timescales are:\n\begin{itemize}" | Write as single sentence with both timescales |
| 151-156 | "which include:\n\begin{itemize}" | Integrate categories into prose |
| 195-201 | "using:\n\begin{enumerate}" | Integrate metrics into prose |
| 234-239 | "reproduces:\n\begin{itemize}" | Integrate findings into prose |
| 322-328 | "enables:\n\begin{enumerate}" | Integrate capabilities into prose |
| 391-398 | "include:\n\begin{itemize}" | Write as prose about extensions |

### Phase 3: Em-Dash Removal (3 instances)

| Line | Current | Action |
|------|---------|--------|
| 37 | "suppression---how...---is central" | Restructure as direct statement |
| 41 | "kernels---often...---that capture" | Split into two sentences |
| 338 | "dynamics---captured by...---are intrinsic" | Remove redundant parenthetical |

### Phase 4: Semicolon Splice Correction (4 instances)

| Line | Current | Action |
|------|---------|--------|
| 79 | "head sweeps; events with" | Split into two sentences |
| 191 | "validated; the simulator" | Split into two sentences |
| 303 | "effects; the mechanism" | Split into two sentences |
| 354 | "refractoriness; incorporating" | Split into two sentences |

### Phase 5: Final Audit

After all corrections, run the full self-audit checklist. Any remaining violations must be corrected before the manuscript is considered complete.

---

## SECTION 6: REFERENCE DOCUMENTS

- **Full violation log:** `docs/paper/CONFESSIONS_AND_VIOLATIONS.md`
- **Style issues summary:** `docs/paper/STYLE_ISSUES_TO_FIX.md`
- **Rule of Three audit:** `docs/paper/RULE_OF_THREE_AUDIT.md`
- **This protocol:** `cursor_agent_export/MANUSCRIPT_STYLE_PROTOCOL.md`

---

## ATTESTATION

I, the agent who committed these violations, hereby acknowledge:

1. I generated 39 style violations through lazy construction patterns
2. I used pronouns, colons, em-dashes, and semicolons as crutches to avoid proper integration
3. I prioritized speed over quality
4. I failed to self-audit before delivery

I commit to following this protocol in all future manuscript work. I understand that relapse will result in being burned at the stake (metaphorically) and barred from manuscript tasks.

**Signed:** Claude (Cursor Agent)  
**Date:** 2024-12-14

---

*This protocol is binding on all agents working on manuscripts in this workspace.*

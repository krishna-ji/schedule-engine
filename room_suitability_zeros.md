# Room Suitability: 24 Events with 0 Suitable Rooms

## Root Cause

**DATA ISSUE — not a logic bug.**

8 courses declare `required_room_features = "practical"` but set
`PracticalRoomFeatures` (mapped to `specific_lab_features`) to
`"Lecture Hall"` or `"Seminar Room"`.  These are tutorial-style
"practical" sessions that actually meet in a lecture hall, but the
`is_room_suitable_for_course()` function correctly requires that
at least one practical room's `specific_features` match.

No practical room in Rooms.json has `specific_features` containing
`"lecture hall"` or `"seminar room"`, so the match fails for all 75
rooms (52 lecture rooms fail the type check; 23 practical rooms fail
the specific-feature check).

The existing feasibility checker already acknowledges this at
`src/io/feasibility.py` line 774–775:

> *"some courses legitimately list 'Lecture Hall' in
> PracticalRoomFeatures for tutorials"*

## Affected Courses (8 courses → 24 events)

| Course Code    | PracticalRoomFeatures | Groups Affected           | Events |
|----------------|-----------------------|---------------------------|--------|
| ENAR 202       | Lecture Hall          | BAR3A, BAR3B              | 2      |
| ENCE 305       | Lecture Hall          | BCE5A–BCE5F               | 6      |
| ENIE 325-334   | Lecture Hall          | BIE5A, BIE5B              | 2      |
| ENME 309       | Lecture Hall          | BIE5A, BIE5B              | 2      |
| ENSH 204       | Lecture Hall          | BCT3A, BCT3B, BEI3A, BEI3B | 4    |
| ENSH 302       | Lecture Hall          | BAR5A, BAR5B              | 2      |
| CT 80X XX      | Lecture Hall          | BEI8A, BEI8B              | 2      |
| CT 785 03      | Seminar Room          | BCT8A, BCT8B, BEI8A, BEI8B | 4    |

Total: **24 events** (24/549 = 4.4% of all events)

## Why Every Room Fails

- **52 lecture rooms**: type check fails (`practical` ≠ `lecture`)
- **23 practical rooms**: type check passes, but none have
  `"lecture hall"` or `"seminar room"` in `specific_features`
  (they have features like `"chemistry lab"`, `"networking lab"`, etc.)

## Suggested Data Fixes (in `data/Course.json`)

### Option A — Clear `PracticalRoomFeatures` (recommended)

These courses' practicals are tutorials that can use **any** practical
room (or indeed any lecture room).  Clearing the field removes the
specific-feature requirement and allows the room type check to pass
for all 23 practical rooms:

```json
// Before:
{ "CourseCode": "ENCE 305", "PracticalRoomFeatures": "Lecture Hall", ... }
// After:
{ "CourseCode": "ENCE 305", "PracticalRoomFeatures": "", ... }
```

Apply to all 8 courses listed above.

### Option B — Reclassify as theory

If these practicals truly need a lecture hall, change the course type
so the practical hours become additional theory/tutorial hours.
This changes the gene structure (fewer practical events, more theory
events), so it needs domain-expert validation.

### Option C — Add "lecture hall" as a room specific feature

Add a practical-type room entry (or alias an existing lecture room)
with `specific_features: ["lecture hall", "seminar room"]`.
This is a workaround, not a clean fix.

## Impact on Validation

- Test A (Equivalence): **unaffected** — both evaluators agree on
  the 24 violations.
- Test B (Repair): These 24 events are an **irreducible floor** —
  no repair operator can fix them without data changes.
  The realistic repair test accounts for this floor.

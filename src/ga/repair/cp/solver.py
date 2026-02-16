"""CP-SAT Solver: Core constraint-programming model for scheduling subproblems.

Builds an OR-Tools CP-SAT model from a set of ``SessionGene`` objects and
solves for feasible ``(instructor_id, room_id, start_quanta)`` assignments
that satisfy all hard constraints.

Used by both the Global Phase (bridge genes) and the Cluster Phase (per-cluster
genes).  Frozen assignments (from a previous phase) are injected as additional
fixed intervals so the new genes cannot conflict with them.

Hard constraints modelled:
    HC1  StudentGroupExclusivity   -- NoOverlap per group-family member
    HC2  InstructorExclusivity     -- optional-interval NoOverlap per instructor
    HC3  RoomExclusivity           -- optional-interval NoOverlap per room
    HC4  InstructorQualifications  -- domain restriction on instr_idx variable
    HC5  RoomSuitability           -- domain restriction on room_idx variable
    HC6  InstructorTimeAvailability -- conditional domain restriction on start
    HC7  RoomTimeAvailability      -- built into start domain (operating hours)
    HC8  CourseCompleteness        -- structural (gene count fixed by GA)

Soft objectives (when ``soft_objective=True``):
    SC1  GroupCompactness          -- minimise time span per student group
    SC2  InstructorCompactness     -- minimise time span per instructor
    These are added as weighted terms alongside the warm-start deviation.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext

__all__ = ["CPSATSolver", "CPSolveResult", "FrozenAssignment"]

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class FrozenAssignment:
    """A previously decided gene assignment that must be respected.

    Used to pass Global Phase results into the Cluster Phase.  The solver
    adds these as fixed intervals to the relevant NoOverlap constraints
    so that no new gene can conflict with them.
    """

    gene_index: int
    course_id: str
    course_type: str
    instructor_id: str
    group_ids: tuple[str, ...]
    room_id: str
    start_quanta: int
    num_quanta: int

    @classmethod
    def from_gene(cls, idx: int, gene: SessionGene) -> FrozenAssignment:
        return cls(
            gene_index=idx,
            course_id=gene.course_id,
            course_type=gene.course_type,
            instructor_id=gene.instructor_id,
            group_ids=tuple(gene.group_ids),
            room_id=gene.room_id,
            start_quanta=gene.start_quanta,
            num_quanta=gene.num_quanta,
        )


@dataclass
class CPSolveResult:
    """Result of a CP-SAT solve.

    Attributes
    ----------
    success : bool
        True if a feasible (or optimal) solution was found.
    assignments : dict[int, tuple[str, str, int]]
        ``{gene_index: (instructor_id, room_id, start_quanta)}``.
    status : str
        OR-Tools solver status string.
    wall_time : float
        Solve wall time in seconds.
    """

    success: bool = False
    assignments: dict[int, tuple[str, str, int]] = field(default_factory=dict)
    status: str = "UNKNOWN"
    wall_time: float = 0.0


# ── Solver ────────────────────────────────────────────────────────────────


class CPSATSolver:
    """Build and solve a CP-SAT model for a subset of scheduling genes.

    Parameters
    ----------
    ctx : SchedulingContext
        Fully linked scheduling context.
    family_map : dict[str, set[str]]
        Group family map (group -> set of all related groups incl. self).
    timeout_seconds : float
        Maximum solver wall-clock time.  Default 60.
    num_workers : int
        CP-SAT solver parallelism.  Default 4.
    soft_objective : bool
        When True, add soft-constraint objectives (group and instructor
        compactness) to the minimisation target alongside warm-start
        deviation.  Default False (backward-compatible).
    deviation_weight : int
        Weight for warm-start deviation terms.  Default 1.
    compactness_weight : int
        Weight for compactness (span minimisation) terms.  Default 2.
    """

    def __init__(
        self,
        ctx: SchedulingContext,
        family_map: dict[str, set[str]],
        *,
        timeout_seconds: float = 60.0,
        num_workers: int = 4,
        soft_objective: bool = False,
        deviation_weight: int = 1,
        compactness_weight: int = 2,
    ) -> None:
        self.ctx = ctx
        self.family_map = family_map
        self.timeout_seconds = timeout_seconds
        self.num_workers = num_workers
        self.soft_objective = soft_objective
        self.deviation_weight = deviation_weight
        self.compactness_weight = compactness_weight

        # Pre-compute per-course suitable rooms + qualified instructors
        self._suitable_rooms: dict[tuple[str, str], list[str]] = {}
        self._qual_instrs: dict[tuple[str, str], list[str]] = {}
        self._precompute_compatibility()

        # Pre-compute day boundaries for day-boundary validation
        self._day_ranges: list[tuple[int, int]] = []
        self._precompute_days()

        # Cache valid-starts by duration to avoid recomputation
        self._valid_starts_cache: dict[int, list[int]] = {}

    # ── Pre-computation helpers ──────────────────────────────────────

    def _precompute_compatibility(self) -> None:
        from src.utils.room_compatibility import is_room_suitable_for_course

        for key, course in self.ctx.courses.items():
            req = (
                str(getattr(course, "required_room_features", "lecture"))
                .lower()
                .strip()
            )
            lab = getattr(course, "specific_lab_features", None)
            suitable: list[str] = []
            for room in self.ctx.rooms.values():
                rt = str(getattr(room, "room_features", "lecture")).lower().strip()
                rf = getattr(room, "specific_features", None)
                if is_room_suitable_for_course(req, rt, lab, rf):
                    suitable.append(room.room_id)
            self._suitable_rooms[key] = suitable
            self._qual_instrs[key] = list(course.qualified_instructor_ids)

    def _precompute_days(self) -> None:
        from src.domain.gene import get_time_system

        qts = get_time_system()
        if qts is None:
            return
        for d_name in qts.DAY_NAMES:
            d_off = qts.day_quanta_offset.get(d_name)
            d_cnt = qts.day_quanta_count.get(d_name, 0)
            if d_off is not None and d_cnt > 0:
                self._day_ranges.append((d_off, d_off + d_cnt))

    def _valid_starts(self, dur: int) -> list[int]:
        """Return start quanta where a *dur*-length session can be placed
        without crossing a day boundary."""
        if dur in self._valid_starts_cache:
            return self._valid_starts_cache[dur]

        avail_set = set(self.ctx.available_quanta)
        result: list[int] = []
        for sq in sorted(avail_set):
            # Every quantum in [sq..sq+dur) must be a valid operating quantum
            if not all((sq + d) in avail_set for d in range(dur)):
                continue
            # Must stay inside one operating day
            if self._day_ranges:
                ok = False
                for ds, de in self._day_ranges:
                    if ds <= sq < de and sq + dur <= de:
                        ok = True
                        break
                if not ok:
                    continue
            result.append(sq)

        self._valid_starts_cache[dur] = result
        return result

    # ── Public API ───────────────────────────────────────────────────

    def solve(
        self,
        genes: list[SessionGene],
        gene_indices: list[int],
        *,
        frozen: list[FrozenAssignment] | None = None,
        warm_start: bool = True,
    ) -> CPSolveResult:
        """Build and solve a CP-SAT model for the given gene subset.

        Parameters
        ----------
        genes : list[SessionGene]
            The full chromosome.
        gene_indices : list[int]
            Indices into *genes* to optimise.
        frozen : list[FrozenAssignment] | None
            Assignments from a prior phase that must not be violated.
        warm_start : bool
            Seed solver with current gene values (default True).

        Returns
        -------
        CPSolveResult
        """
        from ortools.sat.python import cp_model

        if not gene_indices:
            return CPSolveResult(success=True, status="TRIVIAL")

        t0 = _time.monotonic()
        model = cp_model.CpModel()
        frozen = frozen or []

        avail_quanta = sorted(self.ctx.available_quanta)
        all_rooms = list(self.ctx.rooms.keys())
        all_instrs = list(self.ctx.instructors.keys())

        # ── 1. Decision variables ────────────────────────────────────

        starts: dict[int, Any] = {}
        room_idxs: dict[int, Any] = {}
        instr_idxs: dict[int, Any] = {}
        intervals: dict[int, Any] = {}
        dur_map: dict[int, int] = {}
        rooms_for: dict[int, list[str]] = {}
        instrs_for: dict[int, list[str]] = {}

        for gi in gene_indices:
            g = genes[gi]
            ckey = (g.course_id, g.course_type)
            dur = g.num_quanta
            dur_map[gi] = dur

            # HC5: Room suitability → restrict room domain
            sr = self._suitable_rooms.get(ckey, all_rooms) or all_rooms
            rooms_for[gi] = sr

            # HC4: Instructor qualification → restrict instr domain
            qi = self._qual_instrs.get(ckey, []) or all_instrs
            instrs_for[gi] = qi

            # HC7: Valid time slots
            vs = self._valid_starts(dur) or avail_quanta
            starts[gi] = model.new_int_var_from_domain(
                cp_model.Domain.from_values(vs), f"s{gi}"
            )
            room_idxs[gi] = model.new_int_var(0, len(sr) - 1, f"r{gi}")
            instr_idxs[gi] = model.new_int_var(0, len(qi) - 1, f"i{gi}")
            intervals[gi] = model.new_fixed_size_interval_var(
                starts[gi], dur, f"iv{gi}"
            )

            # Warm-start hints
            if warm_start:
                model.add_hint(starts[gi], g.start_quanta)
                if g.room_id in sr:
                    model.add_hint(room_idxs[gi], sr.index(g.room_id))
                if g.instructor_id in qi:
                    model.add_hint(instr_idxs[gi], qi.index(g.instructor_id))

        # ── 2. HC1 — Student Group Exclusivity ──────────────────────
        #    For every group-family member, all intervals mapped to it
        #    must not overlap (including frozen intervals).

        group_ivs: dict[str, list[Any]] = {}

        def _add_group_iv(gids: tuple[str, ...] | list[str], iv: Any) -> None:
            seen: set[str] = set()
            for gid in gids:
                for fam in self.family_map.get(gid, {gid}):
                    if fam not in seen:
                        seen.add(fam)
                        group_ivs.setdefault(fam, []).append(iv)

        for gi in gene_indices:
            _add_group_iv(genes[gi].group_ids, intervals[gi])

        for fa in frozen:
            fiv = model.new_fixed_size_interval_var(
                fa.start_quanta, fa.num_quanta, f"fg{fa.gene_index}"
            )
            _add_group_iv(fa.group_ids, fiv)

        for ivs in group_ivs.values():
            if len(ivs) >= 2:
                model.add_no_overlap(ivs)

        # ── 3. HC2 — Instructor Exclusivity ─────────────────────────
        #    Per instructor, collect optional intervals (active when
        #    that instructor is selected) + fixed frozen intervals.

        instr_opt_ivs: dict[str, list[Any]] = {}

        for gi in gene_indices:
            for li, iid in enumerate(instrs_for[gi]):
                b = model.new_bool_var(f"is{gi}i{li}")
                model.add(instr_idxs[gi] == li).only_enforce_if(b)
                model.add(instr_idxs[gi] != li).only_enforce_if(b.negated())
                oiv = model.new_optional_fixed_size_interval_var(
                    starts[gi], dur_map[gi], b, f"oi{gi}i{li}"
                )
                instr_opt_ivs.setdefault(iid, []).append(oiv)

        for fa in frozen:
            fiv = model.new_fixed_size_interval_var(
                fa.start_quanta, fa.num_quanta, f"fi{fa.gene_index}"
            )
            instr_opt_ivs.setdefault(fa.instructor_id, []).append(fiv)

        for ivs in instr_opt_ivs.values():
            if len(ivs) >= 2:
                model.add_no_overlap(ivs)

        # ── 4. HC3 — Room Exclusivity ───────────────────────────────

        room_opt_ivs: dict[str, list[Any]] = {}

        for gi in gene_indices:
            for li, rid in enumerate(rooms_for[gi]):
                b = model.new_bool_var(f"rs{gi}r{li}")
                model.add(room_idxs[gi] == li).only_enforce_if(b)
                model.add(room_idxs[gi] != li).only_enforce_if(b.negated())
                oiv = model.new_optional_fixed_size_interval_var(
                    starts[gi], dur_map[gi], b, f"ro{gi}r{li}"
                )
                room_opt_ivs.setdefault(rid, []).append(oiv)

        for fa in frozen:
            fiv = model.new_fixed_size_interval_var(
                fa.start_quanta, fa.num_quanta, f"fr{fa.gene_index}"
            )
            room_opt_ivs.setdefault(fa.room_id, []).append(fiv)

        for ivs in room_opt_ivs.values():
            if len(ivs) >= 2:
                model.add_no_overlap(ivs)

        # ── 5. HC6 — Part-time Instructor Availability ──────────────

        for gi in gene_indices:
            dur = dur_map[gi]
            for li, iid in enumerate(instrs_for[gi]):
                instr = self.ctx.instructors.get(iid)
                if not instr or instr.is_full_time:
                    continue
                ok_starts = [
                    sq
                    for sq in self._valid_starts(dur)
                    if all((sq + d) in instr.available_quanta for d in range(dur))
                ]
                if not ok_starts:
                    model.add(instr_idxs[gi] != li)
                else:
                    b = model.new_bool_var(f"av{gi}i{li}")
                    model.add(instr_idxs[gi] == li).only_enforce_if(b)
                    model.add(instr_idxs[gi] != li).only_enforce_if(b.negated())
                    model.add_linear_expression_in_domain(
                        starts[gi],
                        cp_model.Domain.from_values(ok_starts),
                    ).only_enforce_if(b)

        # ── 6. Objective ─────────────────────────────────────────────
        #    When soft_objective=True, minimise a weighted combination of
        #    deviation from warm-start AND soft-constraint proxies
        #    (group/instructor schedule compactness).
        #    Otherwise, just minimise deviation (preserves GA's soft work).

        obj_terms: list[Any] = []

        # 6a. Deviation from warm start
        if warm_start and gene_indices:
            ub = max(avail_quanta) + 1 if avail_quanta else 35
            for gi in gene_indices:
                d = model.new_int_var(0, ub, f"dv{gi}")
                model.add_abs_equality(d, starts[gi] - genes[gi].start_quanta)
                obj_terms.append(self.deviation_weight * d)

        # 6b. Soft-constraint objectives (compactness)
        if self.soft_objective and gene_indices:
            max_q = max(avail_quanta) + 1 if avail_quanta else 35

            # -- Group compactness: minimise span per group --
            # For each group, span = max(end) - min(start) across all
            # its sessions (both variable genes and frozen).  Minimising
            # span encourages compact schedules with fewer idle gaps.
            group_gene_map: dict[str, list[int]] = {}
            group_frozen_bounds: dict[str, tuple[int, int]] = {}

            for gi in gene_indices:
                for gid in genes[gi].group_ids:
                    for fam in self.family_map.get(gid, {gid}):
                        group_gene_map.setdefault(fam, []).append(gi)

            for fa in frozen:
                for gid in fa.group_ids:
                    for fam in self.family_map.get(gid, {gid}):
                        lo, hi = group_frozen_bounds.get(fam, (max_q, 0))
                        group_frozen_bounds[fam] = (
                            min(lo, fa.start_quanta),
                            max(hi, fa.start_quanta + fa.num_quanta),
                        )

            for gid, gis in group_gene_map.items():
                if len(gis) < 2 and gid not in group_frozen_bounds:
                    continue  # single session, no compactness to optimise

                # Build min/max variables across gene starts/ends for this group
                ends = [starts[gi] + dur_map[gi] for gi in gis]
                start_vars = [starts[gi] for gi in gis]

                g_min = model.new_int_var(0, max_q, f"gmin_{gid}")
                g_max = model.new_int_var(0, max_q, f"gmax_{gid}")
                model.add_min_equality(g_min, start_vars)
                model.add_max_equality(g_max, ends)

                # If frozen sessions exist, constrain the bounds further
                if gid in group_frozen_bounds:
                    flo, fhi = group_frozen_bounds[gid]
                    model.add(g_min <= flo)
                    model.add(g_max >= fhi)

                span = model.new_int_var(0, max_q, f"gspan_{gid}")
                model.add(span == g_max - g_min)
                obj_terms.append(self.compactness_weight * span)

            # -- Instructor compactness: minimise span per instructor --
            instr_gene_map: dict[str, list[int]] = {}
            instr_frozen_bounds: dict[str, tuple[int, int]] = {}

            for gi in gene_indices:
                for li, iid in enumerate(instrs_for[gi]):
                    instr_gene_map.setdefault(iid, []).append(gi)

            for fa in frozen:
                lo, hi = instr_frozen_bounds.get(fa.instructor_id, (max_q, 0))
                instr_frozen_bounds[fa.instructor_id] = (
                    min(lo, fa.start_quanta),
                    max(hi, fa.start_quanta + fa.num_quanta),
                )

            for iid, gis in instr_gene_map.items():
                if len(gis) < 2 and iid not in instr_frozen_bounds:
                    continue
                ends = [starts[gi] + dur_map[gi] for gi in gis]
                start_vars = [starts[gi] for gi in gis]

                i_min = model.new_int_var(0, max_q, f"imin_{iid}")
                i_max = model.new_int_var(0, max_q, f"imax_{iid}")
                model.add_min_equality(i_min, start_vars)
                model.add_max_equality(i_max, ends)

                if iid in instr_frozen_bounds:
                    flo, fhi = instr_frozen_bounds[iid]
                    model.add(i_min <= flo)
                    model.add(i_max >= fhi)

                span = model.new_int_var(0, max_q, f"ispan_{iid}")
                model.add(span == i_max - i_min)
                obj_terms.append(self.compactness_weight * span)

        if obj_terms:
            model.minimize(sum(obj_terms))

        # ── 7. Solve ────────────────────────────────────────────────

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout_seconds
        solver.parameters.num_workers = self.num_workers
        solver.parameters.log_search_progress = False

        status = solver.solve(model)
        wall = _time.monotonic() - t0
        sname = solver.status_name(status)

        logger.info(
            "CP-SAT: status=%s  wall=%.1fs  genes=%d  frozen=%d",
            sname,
            wall,
            len(gene_indices),
            len(frozen),
        )

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            asgn: dict[int, tuple[str, str, int]] = {}
            for gi in gene_indices:
                sq = solver.value(starts[gi])
                ri = solver.value(room_idxs[gi])
                ii = solver.value(instr_idxs[gi])
                asgn[gi] = (instrs_for[gi][ii], rooms_for[gi][ri], sq)
            return CPSolveResult(
                success=True, assignments=asgn, status=sname, wall_time=wall
            )

        logger.warning("CP-SAT failed: %s", sname)
        return CPSolveResult(success=False, status=sname, wall_time=wall)

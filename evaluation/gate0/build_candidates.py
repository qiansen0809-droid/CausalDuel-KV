from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Unit:
    layer: int
    head: int


@dataclass(frozen=True)
class SwapCandidate:
    donor: Unit
    receiver: Unit
    amount: int

    @property
    def name(self) -> str:
        return (
            f"L{self.donor.layer}H{self.donor.head}"
            f"_to_L{self.receiver.layer}H{self.receiver.head}"
            f"_q{self.amount}"
        )


def clone_budget(budget: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(map(int, row)) for row in budget]


def total_budget(budget: Sequence[Sequence[int]]) -> int:
    return sum(sum(int(v) for v in row) for row in budget)


def apply_swap(
    baseline: Sequence[Sequence[int]],
    candidate: SwapCandidate,
    min_keep: int = 1,
    max_keep: int | None = None,
) -> list[list[int]]:
    out = clone_budget(baseline)
    before = total_budget(out)

    d = candidate.donor
    r = candidate.receiver
    q = int(candidate.amount)

    if q <= 0:
        raise ValueError("swap amount must be positive")
    if d == r:
        raise ValueError("donor and receiver must be different units")

    try:
        donor_value = out[d.layer][d.head]
        receiver_value = out[r.layer][r.head]
    except IndexError as e:
        raise ValueError("swap references an invalid layer/head") from e

    if donor_value - q < min_keep:
        raise ValueError(
            f"donor {d} would keep {donor_value - q}, below min_keep={min_keep}"
        )
    if max_keep is not None and receiver_value + q > max_keep:
        raise ValueError(
            f"receiver {r} would keep {receiver_value + q}, above max_keep={max_keep}"
        )

    out[d.layer][d.head] -= q
    out[r.layer][r.head] += q

    after = total_budget(out)
    if after != before:
        raise AssertionError(f"budget changed from {before} to {after}")

    return out


def budget_to_override(
    baseline: Sequence[Sequence[int]],
    candidate: SwapCandidate | None,
    min_keep: int = 1,
    max_keep: int | None = None,
) -> dict[int, list[int]]:
    budget = (
        clone_budget(baseline)
        if candidate is None
        else apply_swap(baseline, candidate, min_keep=min_keep, max_keep=max_keep)
    )
    return {layer: row for layer, row in enumerate(budget)}


def _flatten_units(budget: Sequence[Sequence[int]]) -> list[Unit]:
    return [
        Unit(layer=layer_idx, head=head_idx)
        for layer_idx, row in enumerate(budget)
        for head_idx, _ in enumerate(row)
    ]


def generate_poc_swaps(
    baseline: Sequence[Sequence[int]],
    amount: int,
    num_candidates: int = 3,
    min_keep: int = 1,
    max_keep: int | None = None,
    donor_order: Iterable[Unit] | None = None,
    receiver_order: Iterable[Unit] | None = None,
) -> list[SwapCandidate]:
    """
    Generate deterministic budget-preserving swaps for the engineering POC.

    If explicit donor/receiver rankings are supplied, they are respected. Otherwise
    donors are sorted by larger current budgets and receivers by smaller budgets.

    Gate 0 MiniGate should later replace this fallback ordering with LU-KV marginal
    remove-cost / next-page-gain rankings.
    """
    units = _flatten_units(baseline)

    if donor_order is None:
        donors = sorted(
            units,
            key=lambda u: baseline[u.layer][u.head],
            reverse=True,
        )
    else:
        donors = list(donor_order)

    if receiver_order is None:
        receivers = sorted(
            units,
            key=lambda u: baseline[u.layer][u.head],
        )
    else:
        receivers = list(receiver_order)

    candidates: list[SwapCandidate] = []
    seen: set[tuple[Unit, Unit]] = set()

    for donor in donors:
        if baseline[donor.layer][donor.head] - amount < min_keep:
            continue

        for receiver in receivers:
            if donor == receiver:
                continue
            if max_keep is not None and baseline[receiver.layer][receiver.head] + amount > max_keep:
                continue

            key = (donor, receiver)
            if key in seen:
                continue

            cand = SwapCandidate(donor=donor, receiver=receiver, amount=amount)
            apply_swap(
                baseline,
                cand,
                min_keep=min_keep,
                max_keep=max_keep,
            )
            candidates.append(cand)
            seen.add(key)

            if len(candidates) >= num_candidates:
                return candidates

    return candidates

from evaluation.gate0.build_candidates import (
    SwapCandidate,
    Unit,
    apply_swap,
    generate_poc_swaps,
    total_budget,
)


def test_apply_swap_preserves_budget():
    baseline = [[10, 20], [30, 40]]
    candidate = SwapCandidate(Unit(1, 1), Unit(0, 0), 5)

    swapped = apply_swap(baseline, candidate, min_keep=1, max_keep=100)

    assert total_budget(swapped) == total_budget(baseline)
    assert swapped[1][1] == 35
    assert swapped[0][0] == 15


def test_apply_swap_rejects_min_keep_violation():
    baseline = [[8, 20]]
    candidate = SwapCandidate(Unit(0, 0), Unit(0, 1), 4)

    try:
        apply_swap(baseline, candidate, min_keep=5, max_keep=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected min_keep violation")


def test_poc_candidates_are_budget_preserving():
    baseline = [[20, 40], [30, 10]]
    candidates = generate_poc_swaps(
        baseline,
        amount=5,
        num_candidates=3,
        min_keep=5,
        max_keep=50,
    )

    assert len(candidates) == 3
    for candidate in candidates:
        swapped = apply_swap(
            baseline,
            candidate,
            min_keep=5,
            max_keep=50,
        )
        assert total_budget(swapped) == total_budget(baseline)

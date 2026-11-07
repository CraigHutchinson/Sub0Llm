"""Not trusting one measurement: confirming regressions, and re-measuring what was inconclusive."""

from .common import group_of
from .judge import names_with


def whole_groups(names, evaluations):
    """Expands names to every benchmark in their groups, so pair members are re-measured side by side."""
    groups = {group_of(n) for n in names}
    every = set()
    for rows, pair_rows, env in evaluations:
        every |= {r[0] for r in rows}
    return {n for n in every if group_of(n) in groups} | set(names)


def regressions_in(evaluations):
    """From evaluate() results [(rows, pair_rows, env)]: (every regressed name, the names to re-measure).

    Covers benchmarks, pairs and counter rows. Re-measuring a pair needs its reference too.
    """
    flagged, remeasure = set(), set()
    for rows, pair_rows, env in evaluations:
        flagged |= set(names_with(rows, pair_rows, env, "REGRESSED"))
        remeasure |= flagged
        for p in pair_rows:
            if p[4] == "REGRESSED":
                remeasure.add(p[1]["reference"])
    return flagged, whole_groups(remeasure, evaluations)


def confirm_regressions(evaluate_all, remeasure_rounds, confirm_rounds, more_for=None):
    """Judges, and when something regressed re-measures just that and judges again with all rounds.

    `evaluate_all()` returns the evaluate() results to inspect; `remeasure_rounds(names, n)` adds rounds.
    `more_for(names)`, when given, says how many rounds the names need for the two-level (bimodal) test to be possible
    (see stats.rounds_needed): the confirmation pass is at least that long, because a benchmark that sits at one of two
    levels from process to process cannot be told from a regression with fewer.
    Returns (first-pass regressions, regressions that persist).
    """
    first, remeasure = regressions_in(evaluate_all())
    if not first or confirm_rounds <= 0:
        return first, first
    confirm_rounds = max(confirm_rounds, more_for(remeasure) if more_for else 0)
    print(f"sub0perf: {len(first)} regression(s) on the first pass ({', '.join(sorted(first))});"
          f" confirming with {confirm_rounds} more round(s) for {len(remeasure)} benchmark(s)", flush=True)
    remeasure_rounds(remeasure, confirm_rounds)
    final, _ = regressions_in(evaluate_all())
    print(f"sub0perf: confirmed: {', '.join(sorted(first & final)) or 'none'}")
    print(f"sub0perf: not confirmed: {', '.join(sorted(first - final)) or 'none'}")
    if final - first:
        print(f"sub0perf: also regressed after confirmation: {', '.join(sorted(final - first))}")
    return first, final


def noisy_names(rows, pair_rows):
    """Names judged NOISY (benchmarks or pairs), the references of noisy pairs, and their whole groups."""
    noisy = {r[0] for r in rows if r[5] == "NOISY"} | {p[0] for p in pair_rows if p[4] == "NOISY"}
    noisy |= {p[1]["reference"] for p in pair_rows if p[4] == "NOISY"}
    return whole_groups(noisy, [(rows, pair_rows, None)])


def inconclusive_in(evaluations):
    """From evaluate() results: (every NOISY or ENVIRONMENT name, the names worth re-measuring).

    A method mismatch (epochs, pinned CPU) makes absolute times inconclusive however many rounds are added, so
    those are reported but not re-measured.
    """
    names, remeasure = set(), set()
    for rows, pair_rows, env in evaluations:
        retry = not env.get("method")
        for r in rows:
            if r[5] in ("NOISY", "ENVIRONMENT"):
                names.add(r[0])
                if retry:
                    remeasure.add(r[0])
        for p in pair_rows:
            if p[4] == "NOISY":
                names.add(p[0])
                if retry:
                    remeasure |= {p[0], p[1]["reference"]}
    return names, (whole_groups(remeasure, evaluations) if remeasure else set())


def retry_inconclusive(evaluate_all, remeasure_rounds, extra_rounds, more_for=None, affordable=None):
    """When anything is inconclusive, re-measures its groups once with `extra_rounds` more rounds.

    `affordable(names, rounds)`, when given, says whether the retry fits the time budget (a quick gate); if not, it is skipped,
    since an inconclusive result never blocks. Returns (inconclusive before, inconclusive after). What is still inconclusive
    is the caller's to report.
    """
    first, remeasure = inconclusive_in(evaluate_all())
    if not first or not remeasure or extra_rounds <= 0:
        return first, first
    extra_rounds = max(extra_rounds, more_for(remeasure) if more_for else 0)
    if affordable is not None and not affordable(remeasure, extra_rounds):
        print(f"sub0perf: {len(first)} inconclusive result(s) ({', '.join(sorted(first))}); not re-measured: "
              f"{extra_rounds} more round(s) of {len(remeasure)} benchmark(s) do not fit the remaining time budget", flush=True)
        return first, first
    print(f"sub0perf: {len(first)} inconclusive result(s) ({', '.join(sorted(first))});"
          f" re-measuring {len(remeasure)} benchmark(s) once with {extra_rounds} more round(s)", flush=True)
    remeasure_rounds(remeasure, extra_rounds)
    final, _ = inconclusive_in(evaluate_all())
    print(f"sub0perf: still inconclusive: {', '.join(sorted(final)) or 'none'}")
    return first, final

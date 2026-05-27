#!/usr/bin/env python3
"""Generate Ch21 arithmetic dataset (~225k examples, 5 tiers)."""

import json
import random
import argparse
import sys


# ── Tier 1: Router discrimination ────────────────────────────────────────────

def make_tier1(rng, n=50000):
    """Pure numeric tokens vs pure text tokens — router discrimination test."""
    samples = []
    words = [
        "the", "cat", "sat", "on", "mat", "a", "dog", "ran", "fast", "slow",
        "blue", "red", "big", "small", "tall", "old", "new", "good", "bad",
        "hello", "world", "one", "two", "three", "four", "five",
    ]
    for _ in range(n // 2):
        val = rng.randint(-32768, 32767)
        samples.append({
            "input": str(val),
            "output": "numeric",
            "tier": 1,
            "op": "classify",
        })
    for _ in range(n // 2):
        phrase = " ".join(rng.choices(words, k=rng.randint(1, 4)))
        samples.append({
            "input": phrase,
            "output": "text",
            "tier": 1,
            "op": "classify",
        })
    return samples


# ── Tier 2: Single-operation unit tests ──────────────────────────────────────

def make_tier2(rng, n_per_op=25000):
    """Single-operation arithmetic: A op B = C."""
    samples = []

    def clamp_int16(x):
        return -32768 <= x <= 32767

    for _ in range(n_per_op):
        a = rng.randint(-100, 100)
        b = rng.randint(-100, 100)
        c = a + b
        samples.append({
            "input": f"{a} + {b} =",
            "output": str(c) if clamp_int16(c) else "overflow",
            "tier": 2,
            "op": "add",
        })

    for _ in range(n_per_op):
        a = rng.randint(-100, 100)
        b = rng.randint(-100, 100)
        c = a - b
        samples.append({
            "input": f"{a} - {b} =",
            "output": str(c) if clamp_int16(c) else "overflow",
            "tier": 2,
            "op": "sub",
        })

    for _ in range(n_per_op):
        a = rng.randint(-50, 50)
        b = rng.randint(-50, 50)
        c = a * b
        samples.append({
            "input": f"{a} * {b} =",
            "output": str(c) if clamp_int16(c) else "overflow",
            "tier": 2,
            "op": "mul",
        })

    for _ in range(n_per_op):
        a = rng.randint(-1000, 1000)
        b = rng.randint(1, 100)  # avoid div by zero in basic tier
        c = round(a / b)
        samples.append({
            "input": f"{a} / {b} =",
            "output": str(c) if clamp_int16(c) else "overflow",
            "tier": 2,
            "op": "div",
        })

    # Division by zero
    for _ in range(min(n_per_op // 10, 1000)):
        a = rng.randint(-1000, 1000)
        samples.append({
            "input": f"{a} / 0 =",
            "output": "nan",
            "tier": 2,
            "op": "div_zero",
        })

    for _ in range(n_per_op):
        a = rng.randint(-1000, 1000)
        b = rng.randint(-1000, 1000)
        samples.append({
            "input": f"{a} < {b} ?",
            "output": "1" if a < b else "0",
            "tier": 2,
            "op": "compare",
        })

    return samples


# ── Tier 3: Arithmetic word problems ─────────────────────────────────────────

def make_tier3(rng, n=50000):
    """Natural language arithmetic word problems."""
    templates_add = [
        ("Alice has {a} apples and Bob has {b} apples . Together they have", "{c}"),
        ("A store sold {a} items in the morning and {b} in the afternoon . Total sold :", "{c}"),
        ("{a} plus {b} equals", "{c}"),
        ("The sum of {a} and {b} is", "{c}"),
        ("Add {a} to {b} to get", "{c}"),
    ]
    templates_sub = [
        ("There were {a} birds on a wire . {b} flew away . Now there are", "{c}"),
        ("{a} minus {b} equals", "{c}"),
        ("Subtract {b} from {a} to get", "{c}"),
        ("The difference between {a} and {b} is", "{c}"),
    ]
    templates_mul = [
        ("{a} groups of {b} objects each gives", "{c}"),
        ("{a} times {b} equals", "{c}"),
        ("The product of {a} and {b} is", "{c}"),
    ]
    templates_div = [
        ("{a} items split evenly into {b} groups gives", "{c}"),
        ("{a} divided by {b} is", "{c}"),
        ("Divide {a} by {b} to get", "{c}"),
    ]

    def clamp_int16(x):
        return -32768 <= x <= 32767

    samples = []
    all_templates = [
        (templates_add, "add"),
        (templates_sub, "sub"),
        (templates_mul, "mul"),
        (templates_div, "div"),
    ]

    per_category = n // len(all_templates)
    for templates, op in all_templates:
        for _ in range(per_category):
            if op == "add":
                a = rng.randint(0, 200)
                b = rng.randint(0, 200)
                c = a + b
            elif op == "sub":
                a = rng.randint(0, 200)
                b = rng.randint(0, a)
                c = a - b
            elif op == "mul":
                a = rng.randint(1, 30)
                b = rng.randint(1, 30)
                c = a * b
            else:  # div
                b = rng.randint(1, 20)
                c = rng.randint(1, 100)
                a = b * c  # ensure exact division

            tmpl_in, tmpl_out = rng.choice(templates)
            inp = tmpl_in.format(a=a, b=b, c=c)
            out = tmpl_out.format(a=a, b=b, c=c)
            samples.append({
                "input": inp,
                "output": out if clamp_int16(c) else "overflow",
                "tier": 3,
                "op": op,
            })
    return samples


# ── Tier 4: Multi-step chained calculations ───────────────────────────────────

def make_tier4(rng, n=20000):
    """Multi-step chained calculations: (A op B) op C."""
    samples = []
    ops = ["+", "-"]

    def clamp_int16(x):
        return -32768 <= x <= 32767

    for _ in range(n):
        a = rng.randint(0, 50)
        b = rng.randint(0, 50)
        c = rng.randint(0, 50)
        op1 = rng.choice(ops)
        op2 = rng.choice(ops)

        step1 = a + b if op1 == "+" else a - b
        result = step1 + c if op2 == "+" else step1 - c

        samples.append({
            "input": f"{a} {op1} {b} {op2} {c} =",
            "output": str(result) if clamp_int16(result) else "overflow",
            "tier": 4,
            "op": f"{op1}{op2}",
        })
    return samples


# ── Tier 5: Adversarial/boundary cases ───────────────────────────────────────

def make_tier5():
    """Boundary and adversarial examples."""
    samples = []

    # Int16 boundary values
    boundaries = [-32768, -32767, -1, 0, 1, 32766, 32767]
    for a in boundaries:
        for b in boundaries:
            c = a + b
            ok = -32768 <= c <= 32767
            samples.append({
                "input": f"{a} + {b} =",
                "output": str(c) if ok else "overflow",
                "tier": 5,
                "op": "add_boundary",
            })

    # Division by zero for multiple numerators
    for a in [-32768, -1, 0, 1, 100, 32767]:
        samples.append({
            "input": f"{a} / 0 =",
            "output": "nan",
            "tier": 5,
            "op": "div_zero",
        })

    # Compare equality
    for v in [-10, -1, 0, 1, 10]:
        samples.append({
            "input": f"{v} < {v} ?",
            "output": "0",
            "tier": 5,
            "op": "compare_eq",
        })

    # Large multiplication overflow
    for a in [200, 300, 1000]:
        for b in [200, 300, 1000]:
            c = a * b
            ok = -32768 <= c <= 32767
            samples.append({
                "input": f"{a} * {b} =",
                "output": str(c) if ok else "overflow",
                "tier": 5,
                "op": "mul_overflow",
            })

    # Mixed text-number tokens
    mixed = [
        ("add 5 and 3", "8"),
        ("subtract 2 from 10", "8"),
        ("multiply 4 by 7", "28"),
        ("what is 100 divided by 4", "25"),
        ("the value 42 plus 0 is", "42"),
        ("negative 5 plus 3 equals", "-2"),
    ]
    for inp, out in mixed:
        samples.append({
            "input": inp,
            "output": out,
            "tier": 5,
            "op": "mixed_text_number",
        })

    return samples


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Ch21 arithmetic dataset (~225k examples, 5 tiers).")
    parser.add_argument("--out", default="arithmetic_dataset.json",
                        help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--tier", type=int, default=0,
                        help="Generate only this tier (0 = all)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Generating arithmetic dataset...", file=sys.stderr)

    all_samples = []

    if args.tier == 0 or args.tier == 1:
        t1 = make_tier1(rng)
        all_samples.extend(t1)
        print(f"  Tier 1 (router discrimination): {len(t1)} examples", file=sys.stderr)

    if args.tier == 0 or args.tier == 2:
        t2 = make_tier2(rng)
        all_samples.extend(t2)
        print(f"  Tier 2 (single operations):      {len(t2)} examples", file=sys.stderr)

    if args.tier == 0 or args.tier == 3:
        t3 = make_tier3(rng)
        all_samples.extend(t3)
        print(f"  Tier 3 (word problems):          {len(t3)} examples", file=sys.stderr)

    if args.tier == 0 or args.tier == 4:
        t4 = make_tier4(rng)
        all_samples.extend(t4)
        print(f"  Tier 4 (chained calculations):   {len(t4)} examples", file=sys.stderr)

    if args.tier == 0 or args.tier == 5:
        t5 = make_tier5()
        all_samples.extend(t5)
        print(f"  Tier 5 (adversarial/boundary):   {len(t5)} examples", file=sys.stderr)

    rng.shuffle(all_samples)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_samples)} examples → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

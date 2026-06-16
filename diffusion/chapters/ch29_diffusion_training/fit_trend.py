#!/usr/bin/env python3
# Fit power-law trends to a ch29 training run's (step, recall, NELBO) series and extrapolate.
#   NELBO(s)  = L_inf + a * s^(-b)     (standard LLM loss-vs-step power law)
#   recall(s) = R_inf - c * s^(-d)     (saturating approach to an asymptote)
# Pure Python (no numpy/scipy): grid-search the nonlinear params (asymptote, exponent),
# closed-form least-squares for the linear amplitude at each grid point.
# Usage: python fit_trend.py /tmp/trend.dat [steps_per_sec]
import sys, math

path = sys.argv[1] if len(sys.argv) > 1 else "/d/tmp/trend.dat"
sps  = float(sys.argv[2]) if len(sys.argv) > 2 else None

steps, rec, nel = [], [], []
for line in open(path):
    p = line.split()
    if len(p) != 3: continue
    steps.append(float(p[0])); rec.append(float(p[1])); nel.append(float(p[2]))
n = len(steps)
print(f"{n} points, step {steps[0]:.0f}..{steps[-1]:.0f}, "
      f"recall {rec[0]:.1f}%->{rec[-1]:.1f}%, NELBO {nel[0]:.3f}->{nel[-1]:.3f}")

def fit_pow(xs, ys, asym_lo, asym_hi, sign, b_lo=0.04, b_hi=0.8):
    # y = asym + sign*a*x^(-b); grid over (asym,b), LS for a>=0; return (asym,a,b,rmse)
    best = None
    A = [asym_lo + (asym_hi-asym_lo)*i/120 for i in range(121)]
    B = [b_lo + (b_hi-b_lo)*j/120 for j in range(121)]
    for asym in A:
        for b in B:
            # fit a: minimize sum((ys - asym - sign*a*x^-b)^2) -> a = <(ys-asym), g> / <g,g>
            sg = sgg = 0.0
            for x, y in zip(xs, ys):
                g = x**(-b)
                sg  += (y - asym) * g
                sgg += g * g
            a = (sg / sgg) / sign if sgg > 0 else 0.0
            if a < 0: continue            # amplitude must keep the sign sensible
            e = 0.0
            for x, y in zip(xs, ys):
                pred = asym + sign * a * x**(-b)
                e += (y - pred)**2
            rmse = math.sqrt(e / len(xs))
            if best is None or rmse < best[3]:
                best = (asym, a, b, rmse)
    return best

# Fit on the steady tail (drop the first ~15% noisy warmup so the asymptote isn't biased).
i0 = max(1, int(0.15*n))
xs, ysL, ysR = steps[i0:], nel[i0:], rec[i0:]

Lf = fit_pow(xs, ysL, asym_lo=2.4, asym_hi=ysL[-1]-0.005, sign=+1)
Rf = fit_pow(xs, ysR, asym_lo=ysR[-1]+0.05, asym_hi=ysR[-1]+25, sign=-1)
print(f"NELBO  = {Lf[0]:.3f} + {Lf[1]:.3g}*s^(-{Lf[2]:.3f})   rmse={Lf[3]:.4f}   asymptote L_inf={Lf[0]:.3f}")
print(f"recall = {Rf[0]:.2f} - {Rf[1]:.3g}*s^(-{Rf[2]:.3f})   rmse={Rf[3]:.3f}   asymptote R_inf={Rf[0]:.1f}%")

def predL(s): return Lf[0] + Lf[1]*s**(-Lf[2])
def predR(s): return Rf[0] - Rf[1]*s**(-Rf[2])

cur = steps[-1]
print(f"\ncurrent step {cur:.0f}: NELBO {nel[-1]:.3f} (fit {predL(cur):.3f}), recall {rec[-1]:.1f}% (fit {predR(cur):.1f}%)")
for mult, label in [(2, "2x steps"), (3, "3x steps"), (5, "5x steps"), (10, "10x steps")]:
    s = cur*mult
    print(f"  {label:9s} (step {s:>9.0f}): NELBO ~{predL(s):.3f}, recall ~{predR(s):.1f}%  "
          f"(+{predR(s)-rec[-1]:.1f}pt)")
if sps:
    for hrs in [6, 12, 24, 48]:
        s = cur + sps*3600*hrs
        print(f"  +{hrs:>2}h  (step {s:>9.0f}): NELBO ~{predL(s):.3f}, recall ~{predR(s):.1f}%")

# Diminishing-returns read: recent local rate vs early rate.
def rate(a, b, k=15):
    ds = steps[b]-steps[a];
    return (rec[b]-rec[a])/ds*10000, (nel[a]-nel[b])/ds*10000   # per 10k steps
rr_e, rn_e = rate(i0, i0+15)
rr_l, rn_l = rate(n-16, n-1)
print(f"\nrate per 10k steps  recall: early {rr_e:+.2f}pt -> recent {rr_l:+.2f}pt   "
      f"NELBO: early {rn_e:+.3f} -> recent {rn_l:+.3f}")
# Steps to capture 90% of the remaining recall gain to the asymptote.
rem = Rf[0]-rec[-1]
if Rf[2] > 0 and rem > 0:
    s90 = (Rf[1]/(0.1*rem))**(1.0/Rf[2])
    print(f"remaining recall to asymptote: {rem:.1f}pt; ~90% of it by step {s90:.0f} "
          f"({s90/cur:.1f}x current)")

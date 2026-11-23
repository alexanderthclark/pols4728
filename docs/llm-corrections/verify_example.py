"""Reproduce the original notebook's 40-post point estimates using stdlib only.

The fixed audit IDs are the original NumPy default_rng(2026) draw of 16 posts.
The repeated-audit summaries are retained notebook outputs, not rerun here.
Run from any directory; writes example-results.json next to this script.
"""
import json
from pathlib import Path
from statistics import mean, variance

audit_ids = {1, 3, 5, 8, 11, 12, 13, 15, 18, 22, 27, 29, 30, 33, 35, 39}
rows = []
for i in range(40):
    x = i // 8 + 1
    z = -1 if i % 8 < 4 else 1
    noise = [-3, -1, 1, 3][i % 4]
    rows.append(dict(post=i + 1, x=x, q=x + 0.5 * z,
                     y=20 + 2 * x + 6 * z + noise, z=z,
                     audited=i + 1 in audit_ids))
audit = [r for r in rows if r['audited']]

def avg(data, fn):
    return mean(fn(r) for r in data)

def fit(data, key):
    mx = avg(data, lambda r: r[key])
    my = avg(data, lambda r: r['y'])
    vx = avg(data, lambda r: r[key] ** 2) - mx ** 2
    b = (avg(data, lambda r: r[key] * r['y']) - mx * my) / vx
    return dict(intercept=my - b * mx, slope=b, mean_x=mx, variance_x=vx)

my = avg(rows, lambda r: r['y'])
moments = []
for name, fn in [('X', lambda x, y: x), ('X2', lambda x, y: x*x),
                 ('XY', lambda x, y: x*y)]:
    auto = avg(rows, lambda r: fn(r['q'], r['y']))
    correction = avg(audit, lambda r: fn(r['x'], r['y']) - fn(r['q'], r['y']))
    oracle = avg(rows, lambda r: fn(r['x'], r['y']))
    moments.append(dict(moment=name, automated=auto, correction=correction,
                        corrected=auto+correction, oracle=oracle))
mx, mx2, mxy = [r['corrected'] for r in moments]
vx = mx2 - mx*mx
beta = (mxy - mx*my) / vx
alpha = my - beta*mx
fits = dict(oracle=fit(rows, 'x'), automated=fit(rows, 'q'),
            expert_only=fit(audit, 'x'),
            dsl=dict(intercept=alpha, slope=beta, mean_x=mx, variance_x=vx))
hx = [(r['x']-mx)*(r['y']-alpha-beta*r['x'])/vx for r in audit]
hq = [(r['q']-mx)*(r['y']-alpha-beta*r['q'])/vx for r in audit]
audit_variance = (1/16-1/40)*variance([a-b for a,b in zip(hx,hq)])
dsl_population_se = (variance(hx)/40+audit_variance)**0.5
expert = fits['expert_only']
he = [(r['x']-expert['mean_x'])*(r['y']-expert['intercept']-expert['slope']*r['x'])/expert['variance_x'] for r in audit]
expert_population_se = (variance(he)/16)**0.5
assert abs(beta-68/33)<1e-12
assert abs(fits['oracle']['slope']-2)<1e-12
assert abs(fits['automated']['slope']-28/9)<1e-12
result = dict(source='dsl_biased_regressor.ipynb, current source read 2026-11-16',
              description='Synthetic teaching example, not empirical paper results.',
              rows=rows, moments=moments, fits=fits, mean_y=my,
              population_se=dict(dsl=dsl_population_se, expert_only=expert_population_se),
              audit_only_se=dict(dsl=audit_variance**0.5),
              archived_5000_audit_summary=dict(
                  note='Original notebook outputs, fixed corpus; not population regression SEs.',
                  expert_only=dict(mean_slope=1.993, sd=0.979, rmse=0.979),
                  dsl=dict(mean_slope=2.031, sd=0.191, rmse=0.194)))
Path(__file__).with_name('example-results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(fits=fits, population_se=result['population_se'], moments=moments), indent=2))

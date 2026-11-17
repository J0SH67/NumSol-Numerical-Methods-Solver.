"""Chart helper: builds a Vega-Lite spec for st.vega_lite_chart.

Uses only numpy, so it adds no new dependency (Streamlit renders Vega-Lite
natively).
"""

import math

import numpy as np

# colours chosen to read well on both light and dark backgrounds
CURVE, BAND, POINT, ROOT, GREY = "#1F5FA8", "#1F5FA8", "#E07B00", "#C62828", "#8A8F98"


def _x_enc(domain=None):
    enc = {"field": "x", "type": "quantitative", "title": "x"}
    if domain:
        enc["scale"] = {"domain": domain, "nice": False}
    return enc


def function_spec(f, x_min, x_max, iterates=None, roots=None, band=None,
                  samples=600, height=380):
    """Return a Vega-Lite spec dict plotting f on [x_min, x_max].

    iterates: list of (x, y, label) drawn as white dots on the curve
    roots   : list of x values drawn as red diamonds on the x-axis
    band    : (a, b) shaded vertical band (the starting bracket)
    """
    if not (x_max > x_min):
        raise ValueError("Graph range: the maximum must be greater than the minimum.")
    xs = np.linspace(x_min, x_max, samples)
    ys = []
    for x in xs:
        try:
            y = f(float(x))
            ys.append(y if math.isfinite(y) else None)
        except (ValueError, ZeroDivisionError, OverflowError):
            ys.append(None)
    finite = np.array([y for y in ys if y is not None], dtype=float)
    if finite.size == 0:
        raise ValueError("f(x) is undefined everywhere in this graph range.")

    lo, hi = np.percentile(finite, [2, 98])
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    y_domain = [float(lo - pad), float(hi + pad)]
    # break the line at points far outside the view (e.g. tan(x) asymptotes)
    span = y_domain[1] - y_domain[0]
    ys = [y if (y is not None and y_domain[0] - 3 * span <= y <= y_domain[1] + 3 * span)
          else None for y in ys]

    layers = []
    if band is not None:
        layers.append({
            "data": {"values": [{"x1": float(band[0]), "x2": float(band[1])}]},
            "mark": {"type": "rect", "color": BAND, "opacity": 0.10},
            "encoding": {"x": {"field": "x1", "type": "quantitative", "title": "x"},
                         "x2": {"field": "x2"}},
        })
    layers.append({  # zero line
        "data": {"values": [{"y": 0}]},
        "mark": {"type": "rule", "color": GREY, "strokeDash": [4, 4], "opacity": 0.7},
        "encoding": {"y": {"field": "y", "type": "quantitative", "title": "f(x)"}},
    })
    layers.append({  # the curve
        "data": {"values": [{"x": float(x), "y": y} for x, y in zip(xs, ys)]},
        "mark": {"type": "line", "color": CURVE, "strokeWidth": 2.2,
                 "clip": True, "invalid": None},
        "encoding": {
            "x": _x_enc([float(x_min), float(x_max)]),
            "y": {"field": "y", "type": "quantitative", "title": "f(x)",
                  "scale": {"domain": y_domain, "nice": False}},
        },
    })
    if iterates:
        pts = [{"x": float(x), "y": float(y), "label": str(l)}
               for x, y, l in iterates if math.isfinite(x) and math.isfinite(y)]
        layers.append({
            "data": {"values": pts},
            "mark": {"type": "circle", "color": POINT, "size": 50, "opacity": 0.9,
                     "clip": True},
            "encoding": {
                "x": _x_enc(), "y": {"field": "y", "type": "quantitative", "title": "f(x)"},
                "tooltip": [{"field": "label", "title": "point"},
                            {"field": "x", "type": "quantitative", "format": ".6f"},
                            {"field": "y", "type": "quantitative", "format": ".3e",
                             "title": "f(x)"}],
            },
        })
    if roots:
        pts = [{"x": float(r), "y": 0.0, "label": f"root {i + 1}"}
               for i, r in enumerate(roots) if math.isfinite(r)]
        layers.append({
            "data": {"values": pts},
            "mark": {"type": "point", "shape": "diamond", "filled": True,
                     "size": 200, "color": ROOT, "opacity": 1, "clip": True},
            "encoding": {
                "x": _x_enc(), "y": {"field": "y", "type": "quantitative", "title": "f(x)"},
                "tooltip": [{"field": "label", "title": "point"},
                            {"field": "x", "type": "quantitative", "format": ".8f",
                             "title": "x"}],
            },
        })

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": height,
        "layer": layers,
        "config": {
            "background": "transparent",
            "view": {"stroke": None},
        },
    }

"""Pull Tables 3 and A2 out of the article PDF, to check the reimplementation
against.

Table 3 is typeset rotated 90 degrees, so a printed "column" is a horizontal
band of words sharing a baseline (y1), and within a band it's x0 that orders the
table rows. The band positions below were found by trial and error. Table A2 is
ordinary two-block text.
"""
import csv
import re
import sys
from pathlib import Path

import fitz

PDF = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/Users/zilervarls/Downloads/1-s2.0-S0303264720301088-main.pdf")
DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "table3_published.csv"
OUT_A2 = DATA / "tableA2_published.csv"
PAGE = 6   # 0-based; Table 3 is on printed page 7
PAGE_A2 = 10  # Table A2 is on printed page 11

# baseline y1 -> field, left half is NO 1-30 and right half 31-62
LEFT = {"date": 712.6, "no": 680.5, "t": 662.8, "m": 635.7, "e": 618.2,
        "P": 581.5, "N": 537.4, "u_sign": 488.2, "v_sign": 466.2,
        "delta_sign": 444.3, "region": 422.3}
RIGHT = {"date": 371.3, "no": 339.2, "t": 321.5, "m": 294.4, "e": 276.9,
         "P": 240.2, "N": 196.2, "u_sign": 147.0, "v_sign": 125.0,
         "delta_sign": 103.0, "region": 81.1}
YEARS = {"left": {1: 1997, 10: 1998, 18: 1999}, "right": {31: 2000, 42: 2001, 56: 2002}}
HEADER_JUNK = {"(100", "stems)", "(PU/100", "Natural", "population", "Aphid",
               "enemies", "Time", "(day)", "Survey", "dates", "Region", "Year"}


def band(words, y1_target, tol=0.6):
    """Words on one baseline, left to right, which is table order here."""
    hits = [(x0, w) for x0, _y0, _x1, y1, w in words if abs(y1 - y1_target) < tol]
    hits.sort()
    # some glyph runs are drawn twice, so drop repeated x positions
    out, seen = [], set()
    for x0, w in hits:
        key = round(x0, 0)
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return [w for w in out if w not in HEADER_JUNK]


def extract_a2(doc):
    page = doc[PAGE_A2]
    lines = {}
    for x0, y0, _x1, _y1, w, *_ in page.get_text("words"):
        lines.setdefault(round(y0, 0), []).append((x0, w))

    num = re.compile(r"^-?\d+\.\d+$")
    rows, year_l, year_r = [], None, None
    for y in sorted(lines):
        toks = [w.replace("\u2212", "-") for _x, w in sorted(lines[y])]
        # up to two records per line: [year] NO u v Delta, side by side
        i = 0
        while i < len(toks):
            if toks[i].isdigit() and len(toks[i]) == 4:       # year label
                yr = int(toks[i]); i += 1
                if i < len(toks) and toks[i].isdigit():
                    left = int(toks[i]) <= 30
                    if left:
                        year_l = yr
                    else:
                        year_r = yr
                continue
            if (toks[i].isdigit() and i + 3 < len(toks)
                    and all(num.match(t) for t in toks[i + 1:i + 4])):
                no = int(toks[i])
                rows.append(dict(year=year_l if no <= 30 else year_r, no=no,
                                 u_e12=float(toks[i + 1]), v_e18=float(toks[i + 2]),
                                 delta_e36=float(toks[i + 3])))
                i += 4
                continue
            i += 1

    rows.sort(key=lambda r: r["no"])
    with OUT_A2.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT_A2}")


def main():
    doc = fitz.open(PDF)
    page = doc[PAGE]
    words = [(x0, y0, x1, y1, w) for x0, y0, x1, y1, w, *_ in page.get_text("words")
             if x0 > 150]

    rows = []
    for half, bands in (("left", LEFT), ("right", RIGHT)):
        cols = {name: band(words, y1) for name, y1 in bands.items()}
        n = len(cols["no"])
        assert all(len(v) == n for v in cols.values()), \
            {k: len(v) for k, v in cols.items()}
        year = None
        for i in range(n):
            no = int(cols["no"][i])
            year = YEARS[half].get(no, year)
            rows.append(dict(
                year=year, no=no, date=cols["date"][i], t=int(cols["t"][i]),
                m=float(cols["m"][i]), e=float(cols["e"][i].replace("−", "-")),
                P=float(cols["P"][i]), N=float(cols["N"][i]),
                u_sign=cols["u_sign"][i].replace("–", "-"),
                v_sign=cols["v_sign"][i].replace("–", "-"),
                delta_sign=cols["delta_sign"][i].replace("–", "-"),
                region=cols["region"][i]))

    rows.sort(key=lambda r: r["no"])
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")
    extract_a2(doc)


if __name__ == "__main__":
    main()

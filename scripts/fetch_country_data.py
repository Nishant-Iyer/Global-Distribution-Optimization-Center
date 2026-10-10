"""
Download country indicators from the World Bank API for every country in the dataset.

Writes gdoc_opt/country_indicators.csv with, per country:
  gdp_per_capita   GDP per capita, current US$ (NY.GDP.PCAP.CD), most recent year available
  lpi              Logistics Performance Index, overall score 1-5 (LP.LPI.OVRL.XQ), most recent edition
  price_level      Price level ratio of PPP conversion factor to market exchange rate (PA.NUS.PPPC.RF),
                   i.e. how expensive the country is relative to the US (US = 1). Used as a site running-cost index.
and the year of each value. A value the World Bank doesn't publish is imputed with the median of
countries in the same income group, and flagged in the *_source column.

Run: python scripts/fetch_country_data.py      (needs internet access)
"""
import csv
import json
import os
import statistics
import urllib.request

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
API = "https://api.worldbank.org/v2"
INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "lpi": "LP.LPI.OVRL.XQ",
    "price_level": "PA.NUS.PPPC.RF",
}
# Dataset country names that differ from Natural Earth / ISO naming
MANUAL_ISO3 = {
    "Korea, South": "KOR", "Korea, North": "PRK", "Congo (Kinshasa)": "COD", "Congo (Brazzaville)": "COG",
    "Hong Kong": "HKG", "Singapore": "SGP", "Burma": "MMR", "Côte d’Ivoire": "CIV", "Czechia": "CZE",
    "Taiwan": "TWN",
}
# Not a World Bank economy. IMF World Economic Outlook, 2023 GDP per capita (current US$).
MANUAL_VALUES = {"TWN": {"gdp_per_capita": (32339.0, 2023, "IMF WEO")}}
MANUAL_INCOME = {"TWN": "High income"}


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def country_iso3():
    df = pd.read_csv(os.path.join(ROOT, "final_dataset.csv"))
    names = sorted(df["country"].dropna().unique())
    geo = json.load(open(os.path.join(ROOT, "gdoc_opt", "countries.geojson")))
    index = {}
    for f in geo["features"]:
        p = f["properties"]
        code = p["iso_a3"] if p["iso_a3"] != "-99" else p["adm0_a3"]
        for k in ("name", "name_long", "admin", "formal_en", "sovereignt", "geounit", "brk_name", "name_sort"):
            if p.get(k):
                index[p[k].lower()] = code
    out = {}
    for n in names:
        code = MANUAL_ISO3.get(n) or index.get(n.lower())
        if code:
            out[n] = code
    return out


def main():
    iso = country_iso3()
    codes = sorted(set(iso.values()))
    meta = {}
    res = get(f"{API}/country/all?format=json&per_page=1000")
    for c in res[1]:
        meta[c["id"]] = {"income": c["incomeLevel"]["value"], "region": c["region"]["value"]}
    values = {c: {} for c in codes}
    for col, ind in INDICATORS.items():
        res = get(f"{API}/country/all/indicator/{ind}?format=json&mrnev=1&per_page=20000")
        if len(res) < 2:
            raise SystemExit(f"World Bank API error for {ind}: {res}")
        for row in res[1] or []:
            code = row["countryiso3code"]
            if code in values and row["value"] is not None:
                values[code][col] = (float(row["value"]), int(row["date"]), "World Bank")
    for code, vals in MANUAL_VALUES.items():
        for col, v in vals.items():
            values[code].setdefault(col, v)

    income = {c: (meta.get(c, {}).get("income") or MANUAL_INCOME.get(c, "Unknown")) for c in codes}
    medians = {}
    for col in INDICATORS:
        for grp in set(income.values()):
            vs = [values[c][col][0] for c in codes if income[c] == grp and col in values[c]]
            medians[(col, grp)] = statistics.median(vs) if vs else None
        allv = [values[c][col][0] for c in codes if col in values[c]]
        medians[(col, "Unknown")] = medians.get((col, "Unknown")) or statistics.median(allv)

    rows = []
    for name, code in sorted(iso.items()):
        row = {"country": name, "iso3": code, "region": meta.get(code, {}).get("region", ""), "income_group": income[code]}
        for col in INDICATORS:
            if col in values[code]:
                v, year, src = values[code][col]
            else:
                v, year, src = medians[(col, income[code])], "", f"imputed: median of {income[code]} countries"
            row[col] = round(v, 4) if col != "gdp_per_capita" else round(v, 1)
            row[f"{col}_year"] = year
            row[f"{col}_source"] = src
        rows.append(row)

    path = os.path.join(ROOT, "gdoc_opt", "country_indicators.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for col in INDICATORS:
        n = sum(1 for r in rows if not r[f"{col}_source"].startswith("imputed"))
        print(f"{col}: {n} of {len(rows)} countries from source data")
    missing = sorted(set(pd.read_csv(os.path.join(ROOT, "final_dataset.csv"))["country"].dropna()) - set(iso))
    print(f"wrote {path}; {len(missing)} dataset countries without an ISO code: {missing[:20]}")


if __name__ == "__main__":
    main()

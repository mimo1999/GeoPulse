"""
Gleditsch-Ward country number -> ISO 3166-1 alpha-3.

UCDP keys countries by Gleditsch-Ward number (`ged_events.country_id`); the
rest of this repo keys by FIPS 10-4 two-letter codes (GDELT's `ActionGeo_*`
fields). The chain is:

    UCDP country_id (GW) --[this module]--> ISO3 --[data/iso3_to_fips]--> FIPS2

Both hops are enumerated, never guessed. This repo has already shipped one
FIPS/ISO-3 mismatch that rendered zero countries on the choropleth, so
`build_gw_map()` raises on any GW number it does not recognise rather than
dropping the row. A country that genuinely has no FIPS equivalent must be
listed in UNMAPPED with a reason.

Covers all 125 countries appearing in UCDP GED v26.1 (1989-2025).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ISO3_TO_FIPS_PATH = Path(__file__).resolve().parent / "iso3_to_fips.py"
_spec = importlib.util.spec_from_file_location("_iso3_to_fips", _ISO3_TO_FIPS_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ISO3_TO_FIPS: dict[str, str | None] = _mod.ISO3_TO_FIPS


# GW number -> ISO3. Names in comments are UCDP's own `country` strings, which
# sometimes carry a historical alias in brackets (GW numbers are continuous
# across state successions; the ISO3 is the present-day successor).
GW_TO_ISO3: dict[int, str] = {
    # --- Americas ---
    2:   "USA",  # United States of America
    20:  "CAN",  # Canada
    41:  "HTI",  # Haiti
    51:  "JAM",  # Jamaica
    52:  "TTO",  # Trinidad and Tobago
    70:  "MEX",  # Mexico
    90:  "GTM",  # Guatemala
    91:  "HND",  # Honduras
    92:  "SLV",  # El Salvador
    93:  "NIC",  # Nicaragua
    94:  "CRI",  # Costa Rica
    95:  "PAN",  # Panama
    100: "COL",  # Colombia
    101: "VEN",  # Venezuela
    110: "GUY",  # Guyana
    130: "ECU",  # Ecuador
    135: "PER",  # Peru
    140: "BRA",  # Brazil
    145: "BOL",  # Bolivia
    150: "PRY",  # Paraguay
    160: "ARG",  # Argentina

    # --- Europe ---
    200: "GBR",  # United Kingdom
    210: "NLD",  # Netherlands
    211: "BEL",  # Belgium
    220: "FRA",  # France
    230: "ESP",  # Spain
    260: "DEU",  # Germany
    290: "POL",  # Poland
    305: "AUT",  # Austria
    325: "ITA",  # Italy
    338: "MLT",  # Malta
    339: "ALB",  # Albania
    343: "MKD",  # North Macedonia
    344: "HRV",  # Croatia
    345: "SRB",  # Serbia (Yugoslavia) -- successor state is Serbia
    346: "BIH",  # Bosnia-Herzegovina
    352: "CYP",  # Cyprus
    359: "MDA",  # Moldova
    360: "ROU",  # Romania
    365: "RUS",  # Russia (Soviet Union)
    369: "UKR",  # Ukraine
    371: "ARM",  # Armenia
    372: "GEO",  # Georgia
    373: "AZE",  # Azerbaijan
    380: "SWE",  # Sweden

    # --- Africa ---
    404: "GNB",  # Guinea-Bissau
    420: "GMB",  # Gambia
    432: "MLI",  # Mali
    433: "SEN",  # Senegal
    434: "BEN",  # Benin
    435: "MRT",  # Mauritania
    436: "NER",  # Niger
    437: "CIV",  # Ivory Coast
    438: "GIN",  # Guinea
    439: "BFA",  # Burkina Faso
    450: "LBR",  # Liberia
    451: "SLE",  # Sierra Leone
    452: "GHA",  # Ghana
    461: "TGO",  # Togo
    471: "CMR",  # Cameroon
    475: "NGA",  # Nigeria
    482: "CAF",  # Central African Republic
    483: "TCD",  # Chad
    484: "COG",  # Congo (Brazzaville)
    490: "COD",  # DR Congo (Zaire)
    500: "UGA",  # Uganda
    501: "KEN",  # Kenya
    510: "TZA",  # Tanzania
    516: "BDI",  # Burundi
    517: "RWA",  # Rwanda
    520: "SOM",  # Somalia
    522: "DJI",  # Djibouti
    530: "ETH",  # Ethiopia
    531: "ERI",  # Eritrea
    540: "AGO",  # Angola
    541: "MOZ",  # Mozambique
    551: "ZMB",  # Zambia
    552: "ZWE",  # Zimbabwe (Rhodesia)
    560: "ZAF",  # South Africa
    565: "NAM",  # Namibia
    570: "LSO",  # Lesotho
    571: "BWA",  # Botswana
    572: "SWZ",  # Kingdom of eSwatini (Swaziland)
    580: "MDG",  # Madagascar (Malagasy)
    581: "COM",  # Comoros
    600: "MAR",  # Morocco
    615: "DZA",  # Algeria
    616: "TUN",  # Tunisia
    620: "LBY",  # Libya
    625: "SDN",  # Sudan
    626: "SSD",  # South Sudan

    # --- Middle East ---
    630: "IRN",  # Iran
    640: "TUR",  # Turkey
    645: "IRQ",  # Iraq
    651: "EGY",  # Egypt
    652: "SYR",  # Syria
    660: "LBN",  # Lebanon
    663: "JOR",  # Jordan
    666: "ISR",  # Israel
    670: "SAU",  # Saudi Arabia
    678: "YEM",  # Yemen (North Yemen)
    690: "KWT",  # Kuwait
    692: "BHR",  # Bahrain
    694: "QAT",  # Qatar
    696: "ARE",  # United Arab Emirates

    # --- Asia / Pacific ---
    700: "AFG",  # Afghanistan
    702: "TJK",  # Tajikistan
    703: "KGZ",  # Kyrgyzstan
    704: "UZB",  # Uzbekistan
    710: "CHN",  # China
    750: "IND",  # India
    760: "BTN",  # Bhutan
    770: "PAK",  # Pakistan
    771: "BGD",  # Bangladesh
    775: "MMR",  # Myanmar (Burma)
    780: "LKA",  # Sri Lanka
    790: "NPL",  # Nepal
    800: "THA",  # Thailand
    811: "KHM",  # Cambodia (Kampuchea)
    812: "LAO",  # Laos
    820: "MYS",  # Malaysia
    840: "PHL",  # Philippines
    850: "IDN",  # Indonesia
    900: "AUS",  # Australia
    910: "PNG",  # Papua New Guinea
    940: "SLB",  # Solomon Islands
}

# GW numbers deliberately excluded from the FIPS panel, with the reason.
# Empty today -- every GED country resolves. Kept so a future GED release that
# introduces an unmappable entity has a documented place to go instead of
# being silently dropped.
UNMAPPED: dict[int, str] = {}

# Overrides for data/iso3_to_fips.py. Empty -- the three entries that needed
# fixing (COD, COG, SSD) were wrong at the source and were corrected there
# instead, since every other consumer of that module was wrong too.
FIPS_PATCH: dict[str, str] = {}


def gw_to_fips(gw_id: int) -> str | None:
    """GW number -> FIPS-2, or None if the country has no FIPS equivalent.

    Raises KeyError on an unknown GW number -- an unrecognised country must
    surface loudly, not vanish from the panel.
    """
    if gw_id in UNMAPPED:
        return None
    iso3 = GW_TO_ISO3[gw_id]
    return FIPS_PATCH.get(iso3) or ISO3_TO_FIPS.get(iso3)


def gw_to_iso3(gw_id: int) -> str:
    return GW_TO_ISO3[gw_id]


def build_gw_map(rows: list[tuple[int, str]]) -> list[tuple[int, str, str | None, str | None, str | None]]:
    """Build ucdp.gw_country_map rows from (country_id, country_name) pairs.

    Raises on any GW number that is neither in GW_TO_ISO3 nor UNMAPPED, and on
    any ISO3 that resolves to no FIPS code without an explicit UNMAPPED entry.
    """
    out = []
    unknown_gw, unknown_fips = [], []
    for gw_id, name in rows:
        if gw_id in UNMAPPED:
            out.append((gw_id, name, GW_TO_ISO3.get(gw_id), None, UNMAPPED[gw_id]))
            continue
        if gw_id not in GW_TO_ISO3:
            unknown_gw.append((gw_id, name))
            continue
        iso3 = GW_TO_ISO3[gw_id]
        fips = FIPS_PATCH.get(iso3) or ISO3_TO_FIPS.get(iso3)
        if not fips:
            unknown_fips.append((gw_id, name, iso3))
            continue
        out.append((gw_id, name, iso3, fips, None))

    if unknown_gw or unknown_fips:
        raise KeyError(
            "Unresolved UCDP countries -- add to GW_TO_ISO3/FIPS_PATCH, or to "
            "UNMAPPED with a reason. Never drop silently.\n"
            f"  unknown GW numbers: {unknown_gw}\n"
            f"  ISO3 with no FIPS:  {unknown_fips}"
        )
    return out

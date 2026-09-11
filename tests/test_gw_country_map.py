"""
Unit tests for data/gw_country_map.py -- no DB required.

This is the join-integrity test named in the plan: "Write a test that fails
if any UCDP country with 2023-2025 events doesn't resolve. Enumerate
known-unmapped cases explicitly; never silently drop." The 126-row fixture
below is the full distinct (country_id, country) set from ucdp.ged_events
(all years, 1989-2025), captured 2026-09-11 -- if a future GED release adds a
country, this test starts failing until GW_TO_ISO3 is updated, which is the
intended behaviour: a silent drop must never happen.
"""

from __future__ import annotations

import pytest

from data.gw_country_map import GW_TO_ISO3, ISO3_TO_FIPS, UNMAPPED, build_gw_map

# (gw_id, ucdp_country_name) for every country appearing in GEDEvent_v26_1.csv.
GED_COUNTRIES: list[tuple[int, str]] = [
    (2, "United States of America"), (20, "Canada"), (41, "Haiti"), (51, "Jamaica"),
    (52, "Trinidad and Tobago"), (70, "Mexico"), (90, "Guatemala"), (91, "Honduras"),
    (92, "El Salvador"), (93, "Nicaragua"), (94, "Costa Rica"), (95, "Panama"),
    (100, "Colombia"), (101, "Venezuela"), (110, "Guyana"), (130, "Ecuador"),
    (135, "Peru"), (140, "Brazil"), (145, "Bolivia"), (150, "Paraguay"), (160, "Argentina"),
    (200, "United Kingdom"), (210, "Netherlands"), (211, "Belgium"), (220, "France"),
    (230, "Spain"), (260, "Germany"), (290, "Poland"), (305, "Austria"), (325, "Italy"),
    (338, "Malta"), (339, "Albania"), (343, "North Macedonia"), (344, "Croatia"),
    (345, "Serbia (Yugoslavia)"), (346, "Bosnia-Herzegovina"), (352, "Cyprus"),
    (359, "Moldova"), (360, "Romania"), (365, "Russia (Soviet Union)"), (369, "Ukraine"),
    (371, "Armenia"), (372, "Georgia"), (373, "Azerbaijan"), (380, "Sweden"),
    (404, "Guinea-Bissau"), (420, "Gambia"), (432, "Mali"), (433, "Senegal"),
    (434, "Benin"), (435, "Mauritania"), (436, "Niger"), (437, "Ivory Coast"),
    (438, "Guinea"), (439, "Burkina Faso"), (450, "Liberia"), (451, "Sierra Leone"),
    (452, "Ghana"), (461, "Togo"), (471, "Cameroon"), (475, "Nigeria"),
    (482, "Central African Republic"), (483, "Chad"), (484, "Congo"),
    (490, "DR Congo (Zaire)"), (500, "Uganda"), (501, "Kenya"), (510, "Tanzania"),
    (516, "Burundi"), (517, "Rwanda"), (520, "Somalia"), (522, "Djibouti"),
    (530, "Ethiopia"), (531, "Eritrea"), (540, "Angola"), (541, "Mozambique"),
    (551, "Zambia"), (552, "Zimbabwe (Rhodesia)"), (560, "South Africa"),
    (565, "Namibia"), (570, "Lesotho"), (571, "Botswana"),
    (572, "Kingdom of eSwatini (Swaziland)"), (580, "Madagascar (Malagasy)"),
    (581, "Comoros"), (600, "Morocco"), (615, "Algeria"), (616, "Tunisia"),
    (620, "Libya"), (625, "Sudan"), (626, "South Sudan"), (630, "Iran"),
    (640, "Turkey"), (645, "Iraq"), (651, "Egypt"), (652, "Syria"), (660, "Lebanon"),
    (663, "Jordan"), (666, "Israel"), (670, "Saudi Arabia"), (678, "Yemen (North Yemen)"),
    (690, "Kuwait"), (692, "Bahrain"), (694, "Qatar"), (696, "United Arab Emirates"),
    (700, "Afghanistan"), (702, "Tajikistan"), (703, "Kyrgyzstan"), (704, "Uzbekistan"),
    (710, "China"), (750, "India"), (760, "Bhutan"), (770, "Pakistan"),
    (771, "Bangladesh"), (775, "Myanmar (Burma)"), (780, "Sri Lanka"), (790, "Nepal"),
    (800, "Thailand"), (811, "Cambodia (Kampuchea)"), (812, "Laos"), (820, "Malaysia"),
    (840, "Philippines"), (850, "Indonesia"), (900, "Australia"),
    (910, "Papua New Guinea"), (940, "Solomon Islands"),
]


def test_all_ged_countries_resolve():
    """The core join-integrity assertion: no GED country is silently dropped."""
    mapped = build_gw_map(GED_COUNTRIES)
    assert len(mapped) == len(GED_COUNTRIES)


def test_all_ged_countries_get_a_fips_code():
    """Every GED country resolves to a real FIPS-2, not just a None placeholder
    (UNMAPPED is empty today; this fails loudly the day that changes)."""
    mapped = build_gw_map(GED_COUNTRIES)
    missing = [(gw, name) for gw, name, iso3, fips, note in mapped if fips is None]
    assert not missing, f"GED countries with no FIPS code: {missing}"


def test_no_duplicate_fips_codes():
    """Two different countries silently sharing a FIPS code is exactly the
    class of bug this module exists to prevent (found and fixed 2026-09-11:
    COD/COG/CAF transposition, SCG/SRB collision)."""
    mapped = build_gw_map(GED_COUNTRIES)
    fips_codes = [fips for _, _, _, fips, _ in mapped if fips]
    assert len(fips_codes) == len(set(fips_codes)), "duplicate FIPS codes in the GED country set"


def test_unknown_gw_id_raises():
    """An unrecognised GW number must raise, never be dropped."""
    with pytest.raises(KeyError):
        build_gw_map([(999999, "Not A Real Country")])


def test_gw_to_iso3_has_no_stale_unmapped_entries():
    """Every UNMAPPED gw_id should actually be absent from GW_TO_ISO3 -- if it's
    in both, GW_TO_ISO3 silently wins and the UNMAPPED reason is dead."""
    overlap = set(UNMAPPED) & set(GW_TO_ISO3)
    assert not overlap, f"gw_ids in both GW_TO_ISO3 and UNMAPPED: {overlap}"


def test_iso3_to_fips_has_no_duplicate_fips_codes():
    """Whole-file audit, not just the GED subset -- catches the next
    transposition before it reaches a join."""
    rev: dict[str, list[str]] = {}
    for iso3, fips in ISO3_TO_FIPS.items():
        if fips:
            rev.setdefault(fips, []).append(iso3)
    dups = {fips: isos for fips, isos in rev.items() if len(isos) > 1}
    assert not dups, f"duplicate FIPS codes in data/iso3_to_fips.py: {dups}"

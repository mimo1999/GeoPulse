"""
CAMEO Actor Type ("role") codes -- the well-known 3-letter institutional
role codes GDELT uses when an actor has no country attached (e.g. a bare
"GOV" or a concatenated "GOVMIL"). Used by ingestion/graph_builder.py to
decide when a country can legitimately be *inferred* from the event's own
location, versus when an actor code is something else entirely and must be
left unresolved rather than guessed at.

Not exhaustive: CAMEO also defines much larger ethnic and religious
sub-code tables, which are not enumerated here. An actor code containing
one of those is correctly left un-inferred by this module, not misclassified.
"""

from __future__ import annotations

CAMEO_ACTOR_TYPE_CODES: dict[str, str] = {
    "AGR": "Agriculture",
    "BUS": "Business",
    "COP": "Police Forces",
    "CRM": "Criminal",
    "CVL": "Civilian",
    "DEV": "Development",
    "EDU": "Education",
    "ELI": "Elite",
    "ENV": "Environmental",
    "GOV": "Government",
    "HLH": "Health",
    "HRI": "Human Rights",
    "IGO": "Intergovernmental Organization",
    "INS": "Insurgents",
    "JUD": "Judicial",
    "LAB": "Labor",
    "LEG": "Legislature",
    "MED": "Media",
    "MIL": "Military",
    "MNC": "Multinational Corporation",
    "NGM": "Non-Governmental Movement",
    "NGO": "Non-Governmental Organization",
    "OPP": "Political Opposition",
    "PTY": "Political Party",
    "REB": "Rebel",
    "REF": "Refugees",
    "SEP": "Separatist",
    "SPY": "State Intelligence",
    "UAF": "Unaligned Armed Forces",
    "UIF": "Unidentified Forces",
}


def is_bare_role_code(code: str | None) -> bool:
    """True if `code` is composed entirely of known CAMEO Actor Type codes
    chunked in 3 characters (e.g. "GOV", "GOVMIL", "GOVCOPMIL") -- i.e. an
    actor identity carrying no country information at all, whose country
    can only ever come from context (the event's own location), never from
    the actor code itself.
    """
    if not code or len(code) % 3 != 0:
        return False
    chunks = [code[i:i + 3] for i in range(0, len(code), 3)]
    return all(chunk in CAMEO_ACTOR_TYPE_CODES for chunk in chunks)

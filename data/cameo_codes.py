"""
CAMEO event-code reference and interaction-type classification.

CAMEO (Conflict and Mediation Event Observations) codes every GDELT event
with a 2-digit root code (1-20), 3-4 digit base code, and a `quad_class`
(1=Verbal Cooperation, 2=Material Cooperation, 3=Verbal Conflict,
4=Material Conflict) that GDELT itself derives from the root code.

This module maps CAMEO codes to interpretable interaction types
(cooperation, consultation, conflict). `consultation` (root 04) is broken out of `cooperation`
specifically because it is a distinct category in the target taxonomy, not because
CAMEO treats it differently from the rest of quad_class 1 -- it doesn't.
"""

from __future__ import annotations

# Official CAMEO root-code names (01-20).
CAMEO_ROOT_NAMES: dict[int, str] = {
    1: "Make Public Statement",
    2: "Appeal",
    3: "Express Intent to Cooperate",
    4: "Consult",
    5: "Engage in Diplomatic Cooperation",
    6: "Engage in Material Cooperation",
    7: "Provide Aid",
    8: "Yield",
    9: "Investigate",
    10: "Demand",
    11: "Disapprove",
    12: "Reject",
    13: "Threaten",
    14: "Protest",
    15: "Exhibit Military Posture",
    16: "Reduce Relations",
    17: "Coerce",
    18: "Assault",
    19: "Fight",
    20: "Engage in Unconventional Mass Violence",
}

# GDELT's own quad_class, 1:1 with root code (kept here so the interaction-type
# mapping can be derived from quad_class instead of hardcoding the same root
# ranges twice, and so any consumer that only has quad_class can still classify).
ROOT_TO_QUAD_CLASS: dict[int, int] = {
    **{r: 1 for r in range(1, 6)},    # 01-05: Verbal Cooperation
    **{r: 2 for r in range(6, 10)},   # 06-09: Material Cooperation
    **{r: 3 for r in range(10, 14)},  # 10-13: Verbal Conflict
    **{r: 4 for r in range(14, 21)},  # 14-20: Material Conflict
}

CONSULT_ROOT_CODE = 4

INTERACTION_TYPES = ("cooperation", "consultation", "conflict")


def interaction_type(root_code: int | None, quad_class: int | None = None) -> str | None:
    """Map a CAMEO root code (and/or quad_class) to one of the three
    interaction types. root_code takes precedence when both are given, since
    it's the only way to isolate `consultation`; quad_class alone can still
    resolve cooperation-vs-conflict, just never consultation specifically.

    Returns None for an unrecognised code -- callers must not guess.
    """
    if root_code is not None:
        if root_code not in CAMEO_ROOT_NAMES:
            return None
        if root_code == CONSULT_ROOT_CODE:
            return "consultation"
        quad = ROOT_TO_QUAD_CLASS[root_code]
    elif quad_class is not None:
        quad = quad_class
    else:
        return None

    if quad in (1, 2):
        return "cooperation"
    if quad in (3, 4):
        return "conflict"
    return None

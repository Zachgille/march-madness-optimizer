from __future__ import annotations

import re
from typing import Optional


def slugify(value: str) -> str:
    s = value.lower().strip()
    s = s.replace("&", "and")
    s = s.replace("'", "")
    s = s.replace(".", "")
    s = s.replace("/", " ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_team_name(name: str) -> str:
    if name is None:
        return ""

    s = str(name).lower().strip()

    replacements = {
        "&": " and ",
        "st.": "state",
        "saint": "st",
        "uc santa barbara": "ucsb",
        "uconn": "connecticut",
        "ole miss": "mississippi",
        "miami (fl)": "miami fl",
        "miami (oh)": "miami oh",
        "nc state": "north carolina state",
        "unc": "north carolina",
        "usc": "southern california",
        "lsu": "louisiana state",
        "byu": "brigham young",
        "smu": "southern methodist",
        "vcu": "virginia commonwealth",
        "utep": "texas el paso",
        "utsa": "texas san antonio",
        "uab": "alabama birmingham",
        "umass": "massachusetts",
        "siu": "southern illinois",
        "uic": "illinois chicago",
        "uc irvine": "california irvine",
        "uc davis": "california davis",
        "uc san diego": "california san diego",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = s.replace("(", " ")
    s = s.replace(")", " ")
    s = s.replace("'", "")
    s = s.replace(".", "")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\bthe\b", " ", s)
    s = re.sub(r"\buniversity of\b", " ", s)
    s = re.sub(r"\buniv\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_team_key(season: int, team_name: str) -> str:
    return f"{season}_{slugify(normalize_team_name(team_name))}"


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def parse_seed_label(seed_label: str) -> tuple[int, bool, Optional[str]]:
    """
    Examples:
      '1'   -> (1, False, None)
      '11a' -> (11, True, '11')
      '16b' -> (16, True, '16')
    """
    s = str(seed_label).strip().lower()
    match = re.fullmatch(r"(\d{1,2})([ab])?", s)
    if not match:
        raise ValueError(f"Invalid seed_label: {seed_label}")

    seed_num = int(match.group(1))
    suffix = match.group(2)
    is_play_in = suffix is not None
    play_in_group = str(seed_num) if is_play_in else None
    return seed_num, is_play_in, play_in_group
"""MLB constants: teams, divisions, enums."""

from enum import StrEnum


class GameStatus(StrEnum):
    SCHEDULED = "scheduled"
    PREGAME = "pregame"
    LIVE = "live"
    FINAL = "final"
    POSTPONED = "postponed"
    SUSPENDED = "suspended"


class MarketType(StrEnum):
    H2H = "h2h"
    TOTALS = "totals"
    SPREADS = "spreads"


class ConfidenceTier(StrEnum):
    A = "A"  # 5%+ edge, high confidence
    B = "B"  # 3-5% edge
    C = "C"  # 1-3% edge, marginal
    D = "D"  # no actionable edge


class RoofType(StrEnum):
    OPEN = "open"
    RETRACTABLE = "retractable"
    DOME = "dome"


# All 30 MLB teams: abbreviation -> (full name, league, division, MLB Stats API team ID)
MLB_TEAMS: dict[str, tuple[str, str, str, int]] = {
    "AZ":  ("Arizona Diamondbacks",    "NL", "NL West",    109),
    "ATL": ("Atlanta Braves",          "NL", "NL East",    144),
    "BAL": ("Baltimore Orioles",       "AL", "AL East",    110),
    "BOS": ("Boston Red Sox",          "AL", "AL East",    111),
    "CHC": ("Chicago Cubs",            "NL", "NL Central", 112),
    "CWS": ("Chicago White Sox",       "AL", "AL Central", 145),
    "CIN": ("Cincinnati Reds",         "NL", "NL Central", 113),
    "CLE": ("Cleveland Guardians",     "AL", "AL Central", 114),
    "COL": ("Colorado Rockies",        "NL", "NL West",    115),
    "DET": ("Detroit Tigers",          "AL", "AL Central", 116),
    "HOU": ("Houston Astros",          "AL", "AL West",    117),
    "KC":  ("Kansas City Royals",      "AL", "AL Central", 118),
    "LAA": ("Los Angeles Angels",      "AL", "AL West",    108),
    "LAD": ("Los Angeles Dodgers",     "NL", "NL West",    119),
    "MIA": ("Miami Marlins",           "NL", "NL East",    146),
    "MIL": ("Milwaukee Brewers",       "NL", "NL Central", 158),
    "MIN": ("Minnesota Twins",         "AL", "AL Central", 142),
    "NYM": ("New York Mets",           "NL", "NL East",    121),
    "NYY": ("New York Yankees",        "AL", "AL East",    147),
    "OAK": ("Oakland Athletics",       "AL", "AL West",    133),
    "PHI": ("Philadelphia Phillies",   "NL", "NL East",    143),
    "PIT": ("Pittsburgh Pirates",      "NL", "NL Central", 134),
    "SD":  ("San Diego Padres",        "NL", "NL West",    135),
    "SF":  ("San Francisco Giants",    "NL", "NL West",    137),
    "SEA": ("Seattle Mariners",        "AL", "AL West",    136),
    "STL": ("St. Louis Cardinals",     "NL", "NL Central", 138),
    "TB":  ("Tampa Bay Rays",          "AL", "AL East",    139),
    "TEX": ("Texas Rangers",           "AL", "AL West",    140),
    "TOR": ("Toronto Blue Jays",       "AL", "AL East",    141),
    "WSH": ("Washington Nationals",    "NL", "NL East",    120),
}

# Reverse lookup: MLB API team ID -> abbreviation
MLB_ID_TO_ABBR: dict[int, str] = {v[3]: k for k, v in MLB_TEAMS.items()}

# Park data: abbreviation -> (venue name, MLB venue ID, roof type, elevation_ft)
MLB_PARKS: dict[str, tuple[str, int, str, int]] = {
    "AZ":  ("Chase Field",                 15, "retractable", 1082),
    "ATL": ("Truist Park",                4705, "open",         996),
    "BAL": ("Oriole Park at Camden Yards",   2, "open",          30),
    "BOS": ("Fenway Park",                   3, "open",          20),
    "CHC": ("Wrigley Field",                17, "open",         600),
    "CWS": ("Guaranteed Rate Field",         4, "open",         595),
    "CIN": ("Great American Ball Park",   2602, "open",         490),
    "CLE": ("Progressive Field",             5, "open",         660),
    "COL": ("Coors Field",                  19, "open",        5280),
    "DET": ("Comerica Park",              2394, "open",         600),
    "HOU": ("Minute Maid Park",           2392, "retractable",   42),
    "KC":  ("Kauffman Stadium",              7, "open",         750),
    "LAA": ("Angel Stadium",                 1, "open",         160),
    "LAD": ("Dodger Stadium",               22, "open",         515),
    "MIA": ("LoanDepot Park",             4169, "retractable",    7),
    "MIL": ("American Family Field",        32, "retractable",  640),
    "MIN": ("Target Field",              3312, "open",         830),
    "NYM": ("Citi Field",                 3289, "open",          12),
    "NYY": ("Yankee Stadium",             3313, "open",          50),
    "OAK": ("Oakland Coliseum",             10, "open",           5),
    "PHI": ("Citizens Bank Park",         2681, "open",          20),
    "PIT": ("PNC Park",                     31, "open",         730),
    "SD":  ("Petco Park",                 2680, "open",          13),
    "SF":  ("Oracle Park",                2395, "open",           0),
    "SEA": ("T-Mobile Park",               680, "retractable",   20),
    "STL": ("Busch Stadium",              2889, "open",         455),
    "TB":  ("Tropicana Field",              12, "dome",          44),
    "TEX": ("Globe Life Field",           5325, "retractable",  545),
    "TOR": ("Rogers Centre",                14, "retractable",  270),
    "WSH": ("Nationals Park",             3309, "open",          25),
}

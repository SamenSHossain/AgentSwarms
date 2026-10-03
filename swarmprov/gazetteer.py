"""Built-in item vocabularies: US states and countries.

Task-specific items (fields of study, occupations, institutions) are learned
from the transcript itself by ``rules.learn_items`` or supplied by the adapter.
"""

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico",
}

COUNTRIES = [
    "Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia", "Costa Rica",
    "Czech Republic", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Japan", "Korea", "Latvia", "Lithuania",
    "Luxembourg", "Mexico", "Netherlands", "New Zealand", "Norway", "Poland", "Portugal",
    "Slovak Republic", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey",
    "Türkiye", "Turkiye", "United Kingdom", "United States", "Brazil", "China", "India",
    "Indonesia", "South Africa", "Argentina", "Russia", "Bulgaria", "Croatia", "Romania",
    "Cyprus", "Malta", "Mozambique", "Bosnia and Herzegovina", "Bosnia", "Kenya", "Nigeria",
    "Egypt", "Peru", "Ukraine", "Vietnam", "Thailand", "Philippines", "Pakistan", "Bangladesh",
    "Czech", "Slovak", "South Korea", "Kazakhstan", "Turkmenistan", "Albania", "Bahrain", "Algeria",
]

# Canonical names for aliases.
ALIASES = {"Czechia": "Czech Republic", "Czech": "Czech Republic", "Slovak": "Slovak Republic",
           "South Korea": "Korea", "Slovakia": "Slovak Republic", "Türkiye": "Turkey",
           "Turkiye": "Turkey", "Bosnia": "Bosnia and Herzegovina"}
for code, name in US_STATES.items():
    ALIASES[code] = name


_KNOWN = {n.lower(): n for n in list(US_STATES.values()) + COUNTRIES}


def canonical_item(s: str) -> str:
    s = " ".join(s.strip().split())
    if len(s) == 2 and s.upper() in US_STATES:
        return US_STATES[s.upper()]
    low = s.lower()
    for k, v in ALIASES.items():
        if len(k) > 2 and low == k.lower():
            return v
    if low in _KNOWN:
        return _KNOWN[low]
    return " ".join(w if (w.isupper() and len(w) <= 3) else w[:1].upper() + w[1:].lower() if w.isupper() else w
                    for w in s.split())

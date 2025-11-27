from typing import Dict, Optional, Tuple
import difflib

LatLon = Tuple[float, float]

# Approximate coordinates for U-M North Campus landmarks (may be refined)
_PLACES: Dict[str, LatLon] = {
    "Pierpont Commons": (42.29138557282375, -83.71760893100681),
    "Duderstadt Center": (42.29155630955083, -83.71624040501234),
    "Chrysler Center": (42.29061714651343, -83.71701566707992),
    "EECS Building": (42.29250304063893, -83.71489417758502),
    "Bob and Betty Beyster Building": (42.29266332592735, -83.7163086070343),
    "BBB Building": (42.29266332592735, -83.7163086070343),
    "BBB": (42.29266332592735, -83.7163086070343),
    "G.G. Brown": (42.293340825166176, -83.71393591317657),
    "GGBL": (42.293340825166176, -83.71393591317657),
    "Lurie Engineering Center": (42.29146534903596, -83.71356411979681),
    "Walgreen Drama Center": (42.291642899493375, -83.71794718244585),
    "Lurie Biomedical Engineering": (42.28878050512783, -83.71347496001384),
    "Ann and Lurie Tower": (42.29200310138956, -83.7162230723024),
    "DOW Building": (42.292683208088945, -83.71571707634271),
    "Leinweber Center": (42.29265122749789, -83.71696910481661),
    "Cooley Laboratory": (42.290443206109266, -83.71397287466915),
    "Nuclear Engineering Laboratory": (42.290832423785325, -83.71476000962568),
    "Bursley Hall": (42.29349691731599, -83.72095419419996),
    "North Campus Recreation Building": (42.29532439705124, -83.71999853189281),
    "NCRB": (42.29532439705124, -83.71999853189281),
    "Ford Motor Company Robotics Building": (42.294150486426844, -83.7099275620452),
    "FMCRB": (42.294150486426844, -83.7099275620452),
}

_PLACES_CI: Dict[str, LatLon] = {name.casefold(): coord for name, coord in _PLACES.items()}

def safe_destination(name: str, threshold: float = 0.5) -> Optional[LatLon]:
    """Case-insensitive lookup of a place name."""
    if not name:
        return None
    query = name.strip().casefold()

    # Exact match
    if query in _PLACES_CI:
        return query
    
    # Fuzzy match
    best_key: Optional[str] = None
    best_score = 0.0

    for place_name in _PLACES_CI.keys():
        score = difflib.SequenceMatcher(None, query, place_name).ratio()
        if score > best_score:
            best_score = score
            best_key = place_name
    
    if best_key is not None and best_score >= threshold:
        return best_key
    
    return name

# We parse free-text house descriptions into the structured features our model expects
import re
from typing import Any

WORD_NUMBERS: dict[str, int] = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'a': 1, 'an': 1,
}

NEIGHBORHOOD_KEYWORDS: dict[str, list[str]] = {
    'NoRidge':  ['north ridge', 'northridge'],
    'NridgHt':  ['northridge heights', 'nridge'],
    'StoneBr':  ['stone brook', 'stonebrook'],
    'Timber':   ['timber', 'timberland'],
    'Veenker':  ['veenker'],
    'Somerst':  ['somerset', 'somerst'],
    'ClearCr':  ['clear creek', 'clearcreek'],
    'Crawfor':  ['crawford'],
    'CollgCr':  ['college creek', 'collegecreek'],
    'Blmngtn':  ['bloomington'],
    'Gilbert':  ['gilbert'],
    'NWAmes':   ['northwest ames', 'northwest'],
    'SawyerW':  ['sawyer west'],
    'Sawyer':   ['sawyer'],
    'NAmes':    ['north ames', 'north side'],
    'Mitchel':  ['mitchell', 'mitchel'],
    'Edwards':  ['edwards'],
    'OldTown':  ['old town', 'oldtown', 'historic'],
    'BrkSide':  ['brookside'],
    'IDOTRR':   ['iowa dot', 'idotrr', 'railroad'],
    'MeadowV':  ['meadow', 'meadow view'],
    'BrDale':   ['briardale'],
    'SWISU':    ['sw isu', 'iowa state', 'campus'],
    'Blueste':  ['bluestem'],
    'NPkVill':  ['northpark', 'north park village'],
}

QUALITY_MAP: dict[str, str] = {
    'exceptional': 'Ex', 'excellent': 'Ex', 'outstanding': 'Ex', 'luxury': 'Ex',
    'premium': 'Ex', 'top': 'Ex', 'high-end': 'Ex', 'high end': 'Ex',
    'very good': 'Gd', 'great': 'Gd', 'good': 'Gd', 'nice': 'Gd',
    'average': 'TA', 'typical': 'TA', 'standard': 'TA', 'decent': 'TA',
    'fair': 'Fa', 'below average': 'Fa', 'mediocre': 'Fa',
    'poor': 'Po', 'bad': 'Po', 'run-down': 'Po', 'run down': 'Po',
}

ZONE_MAP: dict[str, str] = {
    'residential': 'RL', 'low density': 'RL',
    'high density': 'RH',
    'medium density': 'RM',
    'commercial': 'C (all)', 'floating village': 'FV',
}

FOUNDATION_MAP: dict[str, str] = {
    'poured concrete': 'PConc', 'concrete': 'PConc',
    'cinder block': 'CBlock', 'cinder': 'CBlock', 'block': 'CBlock',
    'brick': 'BrkTil', 'stone': 'Stone', 'slab': 'Slab', 'wood': 'Wood',
}

GARAGE_TYPE_MAP: dict[str, str] = {
    'attached': 'Attchd', 'detached': 'Detchd', 'built-in': 'BuiltIn',
    'built in': 'BuiltIn', 'basement garage': 'Basement',
    'carport': 'CarPort',
}

_AREA_PAT = r'(\d[\d,]*(?:\.\d+)?)\s*k?\s*(?:sq(?:uare)?\s*f(?:eet|t|\.)|sf\b)'


def _parse_number(text: str):
    m = re.search(r'\b(\d+)\b', text)
    if m:
        return int(m.group(1))
    for word, val in WORD_NUMBERS.items():
        if re.search(rf'\b{word}\b', text, re.I):
            return val
    return None


def _parse_area(text: str):
    m = re.search(_AREA_PAT, text, re.I)
    if m:
        raw = m.group(1).replace(',', '')
        val = float(raw)
        if 'k' in m.group(0).lower().replace(raw, ''):
            val *= 1000
        return val
    return None


def _parse_year(text: str):
    m = re.search(r'\b(1[89]\d{2}|20[012]\d)\b', text)
    return int(m.group(1)) if m else None


class NLPExtractor:
    def extract(self, description: str) -> dict:
        # We lower-case once and reuse throughout
        txt = description.lower()
        f: dict = {}
        self._extract_bedrooms(txt, f)
        self._extract_bathrooms(txt, f)
        self._extract_living_area(txt, f)
        self._extract_year(txt, f)
        self._extract_garage(txt, f)
        self._extract_quality(txt, f)
        self._extract_special_rooms(txt, f)
        self._extract_outdoor(txt, f)
        self._extract_basement(txt, f)
        self._extract_neighborhood(txt, f)
        self._extract_building_type(txt, f)
        self._extract_utilities(txt, f)
        self._extract_sale_info(txt, f)
        self._fill_defaults(f)
        return f

    def _extract_bedrooms(self, txt, f):
        m = re.search(
            r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)'
            r'[\s\-]*(bed(?:room)?s?|br\b)', txt, re.I)
        if m:
            n = _parse_number(m.group(1))
            if n:
                f['BedroomAbvGr'] = n

    def _extract_bathrooms(self, txt, f):
        m = re.search(
            r'(\d+|one|two|three|four|five|a|an)'
            r'[\s\-]*(?:full\s+)?bath(?:room)?s?(?!\s*half)', txt, re.I)
        if m:
            n = _parse_number(m.group(1))
            if n:
                f['FullBath'] = n
        m2 = re.search(r'(\d+|one|a|an)[\s\-]*half[\s\-]*bath', txt, re.I)
        if m2:
            n = _parse_number(m2.group(1))
            if n:
                f['HalfBath'] = n
        if 'FullBath' not in f:
            m3 = re.search(r'(\d+|one|two|three|four|a|an)[\s\-]*baths?\b', txt, re.I)
            if m3:
                n = _parse_number(m3.group(1))
                if n:
                    f['FullBath'] = n

    def _extract_living_area(self, txt, f):
        # We look for a number followed by sq-ft then a living-area label
        pat1 = _AREA_PAT + r'(?:\s+(?:of\s+)?(?:living|floor|above[\s\-]grade))'
        # We also look for the label appearing before the number
        pat2 = r'(?:living|floor)\s+(?:area|space)[^\d]{0,20}' + _AREA_PAT
        area = None
        for pat in (pat1, pat2):
            m = re.search(pat, txt, re.I)
            if m:
                raw = m.group(1).replace(',', '')
                area = float(raw)
                if 'k' in m.group(0).lower().replace(raw, ''):
                    area *= 1000
                break
        if area is None:
            area = _parse_area(txt)
        if area and 200 < area < 15000:
            f['GrLivArea'] = area
            two_story = re.search(r'two[\s\-]stor|2[\s\-]stor', txt, re.I)
            if two_story:
                f['1stFlrSF'] = round(area * 0.55)
                f['2ndFlrSF'] = round(area * 0.45)
            else:
                f['1stFlrSF'] = area

    def _extract_year(self, txt, f):
        m = re.search(r'(?:built|constructed|built\s+in|year\s+built)[^\d]*(\d{4})', txt, re.I)
        if m:
            f['YearBuilt'] = int(m.group(1))
        else:
            yr = _parse_year(txt)
            if yr and 1872 <= yr <= 2025:
                f['YearBuilt'] = yr
        m2 = re.search(r'(?:remodel(?:ed)?|renovated?|updated?)[^\d]*(\d{4})', txt, re.I)
        if m2:
            f['YearRemodAdd'] = int(m2.group(1))

    def _extract_garage(self, txt, f):
        if re.search(r'no\s+garage|without\s+garage', txt, re.I):
            f['GarageCars'] = 0
            f['GarageArea'] = 0
            return
        m = re.search(r'(\d+|one|two|three|four|a|an)[\s\-]*car[\s\-]*garage', txt, re.I)
        if m:
            n = _parse_number(m.group(1))
            if n:
                f['GarageCars'] = n
                f['GarageArea'] = n * 240
        elif re.search(r'\bgarage\b', txt, re.I):
            f['GarageCars'] = 1
            f['GarageArea'] = 240
        for kw, code in GARAGE_TYPE_MAP.items():
            if kw in txt:
                f['GarageType']   = code
                f['GarageFinish'] = 'Fin' if 'finished' in txt else 'Unf'
                break

    def _extract_quality(self, txt, f):
        for phrase, code in QUALITY_MAP.items():
            if phrase in txt:
                f['ExterQual']   = code
                f['KitchenQual'] = code
                f['OverallQual'] = {'Ex': 9, 'Gd': 7, 'TA': 5, 'Fa': 3, 'Po': 2}.get(code, 5)
                break

    def _extract_special_rooms(self, txt, f):
        m = re.search(r'(\d+|one|two|three|a|an)?[\s\-]*fireplace', txt, re.I)
        if m:
            n = _parse_number(m.group(1) or '1') or 1
            f['Fireplaces'] = n
        if re.search(r'\bpool\b|swimming\s+pool', txt, re.I):
            f['PoolArea'] = 500
        m2 = re.search(
            r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
            r'[\s\-]*(?:total\s+)?rooms?(?:\s+above\s+grade)?', txt, re.I)
        if m2:
            n = _parse_number(m2.group(1))
            if n:
                f['TotRmsAbvGrd'] = n

    def _extract_outdoor(self, txt, f):
        if re.search(r'(?:wood\s+)?deck', txt, re.I):
            a = _parse_area(re.sub(r'.*?deck', '', txt, flags=re.I)[:40])
            f['WoodDeckSF'] = int(a) if a else 200
        if re.search(r'open\s+porch|front\s+porch', txt, re.I):
            f['OpenPorchSF'] = 80
        if re.search(r'enclosed\s+porch', txt, re.I):
            f['EnclosedPorch'] = 120
        if re.search(r'screen(?:ed)?\s+porch', txt, re.I):
            f['ScreenPorch'] = 100
        if re.search(r'large\s+(?:yard|lot|garden|land)', txt, re.I):
            f['LotArea'] = 12000
        elif re.search(r'(?:small|tiny)\s+(?:yard|lot|garden)', txt, re.I):
            f['LotArea'] = 5000
        if re.search(r'\bfence\b', txt, re.I):
            f['Fence'] = 'MnPrv'

    def _extract_basement(self, txt, f):
        if re.search(r'no\s+basement|without\s+basement', txt, re.I):
            f['TotalBsmtSF'] = 0
            f['BsmtFinSF1']  = 0
            f['BsmtUnfSF']   = 0
            return
        if re.search(r'(?:full|finished)\s+basement|basement', txt, re.I):
            after = re.sub(r'.*?basement', '', txt, flags=re.I, count=1)[:60]
            a = _parse_area(after)
            ba = int(a) if a else 800
            f['TotalBsmtSF'] = ba
            if re.search(r'finished\s+basement', txt, re.I):
                f['BsmtFinSF1'] = ba
                f['BsmtUnfSF']  = 0
                f['BsmtQual']   = 'Gd'
                f['BsmtCond']   = 'TA'
            else:
                f['BsmtFinSF1'] = 0
                f['BsmtUnfSF']  = ba

    def _extract_neighborhood(self, txt, f):
        for code, kws in NEIGHBORHOOD_KEYWORDS.items():
            if any(kw in txt for kw in kws):
                f['Neighborhood'] = code
                return

    def _extract_building_type(self, txt, f):
        if re.search(r'single\s*[-\s]*family|detached\s+home', txt, re.I):
            f['BldgType'] = '1Fam'
        elif re.search(r'townhouse|townhome|town\s+house', txt, re.I):
            f['BldgType'] = 'Twnhs'
        elif re.search(r'duplex', txt, re.I):
            f['BldgType'] = '2fmCon'
        elif re.search(r'condo|condominium', txt, re.I):
            f['BldgType'] = 'TwnhsE'
        if re.search(r'two[\s\-]stor(?:y|ey)|2[\s\-]stor(?:y|ey)', txt, re.I):
            f['HouseStyle'] = '2Story'
        elif re.search(r'one[\s\-]stor(?:y|ey)|1[\s\-]stor(?:y|ey)|ranch|bungalow', txt, re.I):
            f['HouseStyle'] = '1Story'
        elif re.search(r'split[\s\-]level', txt, re.I):
            f['HouseStyle'] = 'SLvl'
        for kw, code in FOUNDATION_MAP.items():
            if kw in txt:
                f['Foundation'] = code
                break

    def _extract_utilities(self, txt, f):
        if re.search(r'central\s+air|a/?c\b|air\s+conditioning', txt, re.I):
            f['CentralAir'] = 'Y'
        elif re.search(r'no\s+(?:central\s+)?air|no\s+a/?c', txt, re.I):
            f['CentralAir'] = 'N'
        for zk, zc in ZONE_MAP.items():
            if zk in txt:
                f['MSZoning'] = zc
                break

    def _extract_sale_info(self, txt, f):
        if re.search(r'new\s+construction|brand\s+new|newly\s+built', txt, re.I):
            f['SaleType']      = 'New'
            f['SaleCondition'] = 'Partial'
        elif re.search(r'foreclosure|bank[\s\-]owned', txt, re.I):
            f['SaleCondition'] = 'Abnorml'
        else:
            f['SaleType']      = 'WD'
            f['SaleCondition'] = 'Normal'

    def _fill_defaults(self, f):
        # We supply dataset medians/modes for every feature not yet extracted
        defaults = {
            'MSSubClass': 20, 'MSZoning': 'RL', 'LotFrontage': 69.0,
            'LotArea': 9600, 'Street': 'Pave', 'LotShape': 'Reg',
            'LandContour': 'Lvl', 'Utilities': 'AllPub', 'LotConfig': 'Inside',
            'LandSlope': 'Gtl', 'Neighborhood': 'NAmes', 'Condition1': 'Norm',
            'Condition2': 'Norm', 'BldgType': '1Fam', 'HouseStyle': '1Story',
            'OverallQual': 6, 'OverallCond': 5, 'YearBuilt': 1973,
            'YearRemodAdd': 1994, 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg',
            'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd',
            'MasVnrType': 'None', 'MasVnrArea': 0.0,
            'ExterQual': 'TA', 'ExterCond': 'TA', 'Foundation': 'PConc',
            'BsmtQual': 'TA', 'BsmtCond': 'TA', 'BsmtExposure': 'No',
            'BsmtFinType1': 'Unf', 'BsmtFinSF1': 0.0,
            'BsmtFinType2': 'Unf', 'BsmtFinSF2': 0.0,
            'BsmtUnfSF': 477.5, 'TotalBsmtSF': 991.5,
            'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y',
            'Electrical': 'SBrkr', '1stFlrSF': 1084.0, '2ndFlrSF': 0.0,
            'LowQualFinSF': 0.0, 'GrLivArea': 1464.0,
            'BsmtFullBath': 0.0, 'BsmtHalfBath': 0.0,
            'FullBath': 2, 'HalfBath': 0,
            'BedroomAbvGr': 3, 'KitchenAbvGr': 1,
            'KitchenQual': 'TA', 'TotRmsAbvGrd': 6, 'Functional': 'Typ',
            'Fireplaces': 0, 'FireplaceQu': 'Missing',
            'GarageType': 'Attchd', 'GarageYrBlt': 1980.0,
            'GarageFinish': 'Unf', 'GarageCars': 2.0, 'GarageArea': 480.0,
            'GarageQual': 'TA', 'GarageCond': 'TA', 'PavedDrive': 'Y',
            'WoodDeckSF': 0.0, 'OpenPorchSF': 25.0, 'EnclosedPorch': 0.0,
            '3SsnPorch': 0.0, 'ScreenPorch': 0.0, 'PoolArea': 0.0,
            'PoolQC': 'Missing', 'Fence': 'Missing', 'MiscFeature': 'Missing',
            'MiscVal': 0.0, 'MoSold': 6, 'YrSold': 2008,
            'SaleType': 'WD', 'SaleCondition': 'Normal',
        }
        for k, v in defaults.items():
            if k not in f:
                f[k] = v

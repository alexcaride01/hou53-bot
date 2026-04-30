# We define all Pydantic models used by our FastAPI endpoints
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────
# We expose all 79 Ames features as optional fields with sensible defaults
# so callers do not have to supply every column to get a valid prediction
# ─────────────────────────────────────────────────────────────────

class HouseFeatures(BaseModel):
    # We keep numeric fields separate from categorical ones for readability
    MSSubClass:    Optional[float] = Field(default=20,     description="Building class")
    LotFrontage:   Optional[float] = Field(default=69.0,   description="Linear feet of street connected to property")
    LotArea:       Optional[float] = Field(default=9600.0, description="Lot size in square feet")
    OverallQual:   Optional[float] = Field(default=6,      ge=1, le=10, description="Overall material and finish quality (1–10)")
    OverallCond:   Optional[float] = Field(default=5,      ge=1, le=10, description="Overall condition rating (1–10)")
    YearBuilt:     Optional[float] = Field(default=1973,   description="Original construction date")
    YearRemodAdd:  Optional[float] = Field(default=1994,   description="Remodel date (same as construction if no remodel)")
    MasVnrArea:    Optional[float] = Field(default=0.0,    description="Masonry veneer area in sq ft")
    BsmtFinSF1:    Optional[float] = Field(default=0.0,    description="Type-1 finished basement sq ft")
    BsmtFinSF2:    Optional[float] = Field(default=0.0,    description="Type-2 finished basement sq ft")
    BsmtUnfSF:     Optional[float] = Field(default=477.5,  description="Unfinished basement sq ft")
    TotalBsmtSF:   Optional[float] = Field(default=991.5,  description="Total basement area in sq ft")
    FirstFlrSF:    Optional[float] = Field(default=1084.0, alias="1stFlrSF",  description="First floor sq ft")
    SecondFlrSF:   Optional[float] = Field(default=0.0,    alias="2ndFlrSF",  description="Second floor sq ft")
    LowQualFinSF:  Optional[float] = Field(default=0.0,    description="Low-quality finished sq ft")
    GrLivArea:     Optional[float] = Field(default=1464.0, description="Above-grade living area in sq ft")
    BsmtFullBath:  Optional[float] = Field(default=0.0,    description="Basement full bathrooms")
    BsmtHalfBath:  Optional[float] = Field(default=0.0,    description="Basement half bathrooms")
    FullBath:      Optional[float] = Field(default=2.0,    description="Full bathrooms above grade")
    HalfBath:      Optional[float] = Field(default=0.0,    description="Half bathrooms above grade")
    BedroomAbvGr:  Optional[float] = Field(default=3.0,    description="Bedrooms above basement level")
    KitchenAbvGr:  Optional[float] = Field(default=1.0,    description="Kitchens above grade")
    TotRmsAbvGrd:  Optional[float] = Field(default=6.0,    description="Total rooms above grade (excl. bathrooms)")
    Fireplaces:    Optional[float] = Field(default=0.0,    description="Number of fireplaces")
    GarageYrBlt:   Optional[float] = Field(default=1980.0, description="Year garage was built")
    GarageCars:    Optional[float] = Field(default=2.0,    description="Garage size in car capacity")
    GarageArea:    Optional[float] = Field(default=480.0,  description="Garage area in sq ft")
    WoodDeckSF:    Optional[float] = Field(default=0.0,    description="Wood deck area in sq ft")
    OpenPorchSF:   Optional[float] = Field(default=25.0,   description="Open porch area in sq ft")
    EnclosedPorch: Optional[float] = Field(default=0.0,    description="Enclosed porch area in sq ft")
    ThreeSsnPorch: Optional[float] = Field(default=0.0,    alias="3SsnPorch",  description="Three-season porch area in sq ft")
    ScreenPorch:   Optional[float] = Field(default=0.0,    description="Screen porch area in sq ft")
    PoolArea:      Optional[float] = Field(default=0.0,    description="Pool area in sq ft")
    MiscVal:       Optional[float] = Field(default=0.0,    description="Value of miscellaneous feature ($)")
    MoSold:        Optional[float] = Field(default=6.0,    description="Month sold (1–12)")
    YrSold:        Optional[float] = Field(default=2008.0, description="Year sold")

    # We list categorical features with common valid values in the description
    MSZoning:     Optional[str] = Field(default='RL',      description="Zoning: RL, RM, RH, FV, C (all)")
    Street:       Optional[str] = Field(default='Pave',    description="Road access: Grvl, Pave")
    Alley:        Optional[str] = Field(default=None,      description="Alley access type: Grvl, Pave, or missing")
    LotShape:     Optional[str] = Field(default='Reg',     description="Lot shape: Reg, IR1, IR2, IR3")
    LandContour:  Optional[str] = Field(default='Lvl',     description="Land flatness: Lvl, Bnk, HLS, Low")
    Utilities:    Optional[str] = Field(default='AllPub',  description="Utilities available")
    LotConfig:    Optional[str] = Field(default='Inside',  description="Lot configuration")
    LandSlope:    Optional[str] = Field(default='Gtl',     description="Slope: Gtl, Mod, Sev")
    Neighborhood: Optional[str] = Field(default='NAmes',   description="Neighbourhood code (e.g. NAmes, OldTown, StoneBr)")
    Condition1:   Optional[str] = Field(default='Norm',    description="Proximity to road/railroad")
    Condition2:   Optional[str] = Field(default='Norm',    description="Second proximity condition")
    BldgType:     Optional[str] = Field(default='1Fam',    description="Dwelling type: 1Fam, 2fmCon, Duplex, TwnhsE, Twnhs")
    HouseStyle:   Optional[str] = Field(default='1Story',  description="Style: 1Story, 2Story, 1.5Fin, SLvl, …")
    RoofStyle:    Optional[str] = Field(default='Gable',   description="Roof style: Flat, Gable, Hip, …")
    RoofMatl:     Optional[str] = Field(default='CompShg', description="Roof material")
    Exterior1st:  Optional[str] = Field(default='VinylSd', description="Exterior covering")
    Exterior2nd:  Optional[str] = Field(default='VinylSd', description="Second exterior covering")
    MasVnrType:   Optional[str] = Field(default='None',    description="Masonry veneer type: BrkFace, Stone, None")
    ExterQual:    Optional[str] = Field(default='TA',      description="Exterior quality: Ex, Gd, TA, Fa, Po")
    ExterCond:    Optional[str] = Field(default='TA',      description="Exterior condition: Ex, Gd, TA, Fa, Po")
    Foundation:   Optional[str] = Field(default='PConc',   description="Foundation type: PConc, CBlock, BrkTil, Slab, Stone, Wood")
    BsmtQual:     Optional[str] = Field(default='TA',      description="Basement height quality: Ex, Gd, TA, Fa, Po")
    BsmtCond:     Optional[str] = Field(default='TA',      description="Basement condition")
    BsmtExposure: Optional[str] = Field(default='No',      description="Basement walkout: Gd, Av, Mn, No")
    BsmtFinType1: Optional[str] = Field(default='Unf',     description="Basement finish type 1")
    BsmtFinType2: Optional[str] = Field(default='Unf',     description="Basement finish type 2")
    Heating:      Optional[str] = Field(default='GasA',    description="Heating type")
    HeatingQC:    Optional[str] = Field(default='Ex',      description="Heating quality: Ex, Gd, TA, Fa, Po")
    CentralAir:   Optional[str] = Field(default='Y',       description="Central air conditioning: Y, N")
    Electrical:   Optional[str] = Field(default='SBrkr',   description="Electrical system type")
    KitchenQual:  Optional[str] = Field(default='TA',      description="Kitchen quality: Ex, Gd, TA, Fa, Po")
    Functional:   Optional[str] = Field(default='Typ',     description="Home functionality rating")
    FireplaceQu:  Optional[str] = Field(default=None,      description="Fireplace quality (missing = no fireplace)")
    GarageType:   Optional[str] = Field(default='Attchd',  description="Garage type: Attchd, Detchd, BuiltIn, …")
    GarageFinish: Optional[str] = Field(default='Unf',     description="Garage interior finish: Fin, RFn, Unf")
    GarageQual:   Optional[str] = Field(default='TA',      description="Garage quality")
    GarageCond:   Optional[str] = Field(default='TA',      description="Garage condition")
    PavedDrive:   Optional[str] = Field(default='Y',       description="Paved driveway: Y, P, N")
    PoolQC:       Optional[str] = Field(default=None,      description="Pool quality (missing = no pool)")
    Fence:        Optional[str] = Field(default=None,      description="Fence quality")
    MiscFeature:  Optional[str] = Field(default=None,      description="Miscellaneous feature")
    SaleType:     Optional[str] = Field(default='WD',      description="Type of sale: WD, New, COD, …")
    SaleCondition:Optional[str] = Field(default='Normal',  description="Sale condition: Normal, Abnorml, Partial, …")

    model_config = {"populate_by_name": True}

    def to_feature_dict(self) -> dict[str, Any]:
        # We remap our Python-friendly aliases back to the original column names
        raw = self.model_dump(by_alias=False)
        mapping = {
            'FirstFlrSF':    '1stFlrSF',
            'SecondFlrSF':   '2ndFlrSF',
            'ThreeSsnPorch': '3SsnPorch',
        }
        result: dict[str, Any] = {}
        for k, v in raw.items():
            result[mapping.get(k, k)] = v
        return result


# ─────────────────────────────────────────────────────────────────
# We define request / response envelopes for our two prediction routes
# ─────────────────────────────────────────────────────────────────

class TextPredictionRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=10,
        description="Free-text description of the house (e.g. 'A 3-bedroom house built in 1995 with a 2-car garage…')",
        examples=["A 3-bedroom, 2-bathroom single-family home built in 1995 with central air, "
                  "a 2-car attached garage, finished basement, and a large backyard in a quiet neighbourhood."]
    )


class FeatureImportanceItem(BaseModel):
    feature:     str
    value:       Any
    display:     str
    importance:  float
    description: str


class PriceRange(BaseModel):
    low:  float
    high: float


class PredictionResponse(BaseModel):
    predicted_price: float  = Field(description="Point-estimate of the house sale price in USD")
    price_range:     PriceRange = Field(description="Approximate ±12% confidence interval")
    explanation:     list[FeatureImportanceItem] = Field(description="Top features driving this prediction")
    model_metrics:   dict[str, float] = Field(default_factory=dict, description="Held-out test-set metrics")
    extracted_features: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured features parsed from a text description (only present for /predict/text)"
    )


class HealthResponse(BaseModel):
    status:  str
    metrics: dict[str, float]

"""Cleaning v0: raw Inside Airbnb listings -> DataFrame that follows docs/data_contract.md.

This is a first, simple version so the pipeline work can start in parallel.
The column names and types it returns are the contract; the rules inside can be refined.

Important: nothing here is "learned" from the data (no medians, no top-N computed
on the data). Missing values stay null and are imputed later by an Imputer inside the
Pipeline, fit on the training set only. This avoids data leakage.

Usage from a notebook in notebooks/:
    import sys; sys.path.append("..")
    from src.cleaning import clean
    df_clean = clean(raw)
"""
from pyspark.sql import DataFrame, functions as F

# Provisional price limits (euros per night). To confirm with the EDA histogram.
PRICE_MIN = 15.0
PRICE_MAX = 1000.0

# minimum_nights above this value is capped (about 10% of listings are above 30).
MIN_NIGHTS_CAP = 30

# Fixed list: the 10 most frequent property types in the Sept 2026 snapshot (~98% of rows).
# Hard-coded on purpose (not computed from the data), so cleaning learns nothing.
TOP_PROPERTY_TYPES = [
    "Entire rental unit",
    "Private room in rental unit",
    "Entire condo",
    "Room in hotel",
    "Entire loft",
    "Private room in bed and breakfast",
    "Room in boutique hotel",
    "Entire home",
    "Private room in condo",
    "Entire serviced apartment",
]

# Center of Paris (Notre-Dame area), used for the distance feature.
PARIS_CENTER_LAT = 48.8566
PARIS_CENTER_LON = 2.3522

# Final columns, in the order of the data contract.
CONTRACT_COLUMNS = [
    "id", "price", "label",
    "accommodates", "bedrooms", "beds", "bathrooms",
    "room_type", "property_type", "neighbourhood_cleansed",
    "latitude", "longitude", "dist_center_km",
    "review_scores_rating", "number_of_reviews", "host_is_superhost",
    "minimum_nights", "availability_365", "n_amenities",
]


def _parse_price(col):
    """'$1,386.50' -> 1386.5"""
    return F.regexp_replace(col, r"[$,€\s]", "").cast("double")


def _parse_bathrooms(col):
    """'1.5 shared baths' -> 1.5, 'Half-bath' -> 0.5, 'No bathroom' -> 0.0"""
    number = F.regexp_extract(col, r"([0-9]+(\.[0-9]+)?)", 1)
    return (F.when(F.lower(col).contains("half-bath"), 0.5)
             .when(F.lower(col).contains("no bathroom"), 0.0)
             .when(number != "", number.cast("double"))
             .otherwise(None))


def _t_f_to_int(col):
    """'t' -> 1, 'f' -> 0, anything else -> null"""
    return F.when(col == "t", 1).when(col == "f", 0).otherwise(None)


def _haversine_km(lat, lon):
    """Great-circle distance (km) from (lat, lon) to the center of Paris."""
    lat1, lon1 = F.radians(lat), F.radians(lon)
    lat2, lon2 = F.radians(F.lit(PARIS_CENTER_LAT)), F.radians(F.lit(PARIS_CENTER_LON))
    a = (F.sin((lat2 - lat1) / 2) ** 2
         + F.cos(lat1) * F.cos(lat2) * F.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * F.asin(F.sqrt(a))


def _count_amenities(col):
    """'["Wifi", "Kitchen"]' -> 2. Null or unparsable -> null."""
    items = F.from_json(col, "array<string>")
    return F.when(items.isNull(), None).otherwise(F.size(items))


def clean(raw: DataFrame) -> DataFrame:
    """Apply the data contract to a raw DataFrame read as strings."""
    df = (raw
          .withColumn("id", F.col("id").cast("long"))
          .withColumn("price", _parse_price(F.col("price")))
          # The target is required: rows without a price cannot be used for training.
          .filter(F.col("price").isNotNull())
          .filter(F.col("price").between(PRICE_MIN, PRICE_MAX))
          .withColumn("label", F.log(F.col("price")))

          # Size and capacity
          .withColumn("accommodates", F.col("accommodates").cast("int"))
          .withColumn("bedrooms", F.col("bedrooms").cast("double"))
          .withColumn("beds", F.col("beds").cast("double"))
          .withColumn("bathrooms", _parse_bathrooms(F.col("bathrooms_text")))

          # Categories
          .withColumn("property_type",
                      F.when(F.col("property_type").isin(TOP_PROPERTY_TYPES), F.col("property_type"))
                       .otherwise("Other"))

          # Location
          .withColumn("latitude", F.col("latitude").cast("double"))
          .withColumn("longitude", F.col("longitude").cast("double"))
          .withColumn("dist_center_km", _haversine_km(F.col("latitude"), F.col("longitude")))

          # Reputation
          .withColumn("review_scores_rating", F.col("review_scores_rating").cast("double"))
          .withColumn("number_of_reviews", F.col("number_of_reviews").cast("int"))
          .withColumn("host_is_superhost", _t_f_to_int(F.col("host_is_superhost")))

          # Rules and availability
          .withColumn("minimum_nights", F.col("minimum_nights").cast("int"))
          # Cap long stays. (F.least would turn null into the cap, so use when instead.)
          .withColumn("minimum_nights",
                      F.when(F.col("minimum_nights") > MIN_NIGHTS_CAP, MIN_NIGHTS_CAP)
                       .otherwise(F.col("minimum_nights")))
          .withColumn("availability_365", F.col("availability_365").cast("int"))

          # Amenities
          .withColumn("n_amenities", _count_amenities(F.col("amenities")))
          )

    df = df.select(CONTRACT_COLUMNS)

    missing = [c for c in CONTRACT_COLUMNS if c not in df.columns]
    assert not missing, f"Missing contract columns: {missing}"
    return df

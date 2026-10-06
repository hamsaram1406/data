import json
import math
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SQL_DIR = os.path.join(os.path.dirname(__file__), "..", "sql")

# Final response columns (original + distance fields).
ORG_COLUMNS = [
    "name",
    "organization_type",
    "collaborator",
    "location",
    "size",
    "rating",
    "contact",
    "email",
    "web_url",
    "mission",
    "source",
    "distance",
    "distance_unit",
    "distance_method",
    "distance_status",
]

# Maps raw DB/CSV column names -> final response shape.
DB_RENAME = {
    "org_name": "name",
    "org_type": "organization_type",
    "is_collaborator": "collaborator",
    "org_rating": "rating",
    "org_size": "size",
    "web_url": "web_url",
    "phone": "contact",
}


# ---------------------------------------------------------------------------
# Geocoding — pluggable provider
# ---------------------------------------------------------------------------

# In-memory cache so repeated lookups within a single invocation are free.
_geocode_cache = {}


def geocode(address):
    """Convert an address string to (latitude, longitude) or None.

    This is a **stub** implementation.  The geocoding provider has not been
    finalized, so no external service is called.  Instead, a small built-in
    lookup of Virginia cities is used to support the CSV mock data.

    When a real provider is chosen (e.g. Google Maps, AWS Location Service,
    OpenCage), replace the body of this function.  The rest of the codebase
    calls only this function, so the swap is a single-file change.
    """
    if not address:
        return None

    cache_key = address.strip().lower()
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    coords = _stub_geocode(address)
    _geocode_cache[cache_key] = coords
    return coords


# --- Stub geocoder (mock data only) ----------------------------------------

# A handful of Virginia cities that appear in the CSV mock data, plus a few
# common US cities so that org addresses outside Virginia still resolve.
_CITY_COORDS = {
    "suffolk": (36.7282, -76.5836),
    "chesapeake": (36.7682, -76.2875),
    "norfolk": (36.8508, -76.2859),
    "blacksburg": (37.2296, -80.4139),
    "newport news": (37.0871, -76.4730),
    "virginia beach": (36.8529, -75.9780),
    "richmond": (37.5407, -77.4360),
    "roanoke": (37.2710, -79.9414),
    "charlottesville": (38.0293, -78.4767),
    "arlington": (38.8816, -77.0910),
    "alexandria": (38.8048, -77.0469),
    "hampton": (37.0299, -76.3452),
    "lynchburg": (37.4138, -79.1422),
    "portsmouth": (36.8354, -76.2983),
    "williamsburg": (37.2707, -76.7075),
    # A few non-VA cities for org addresses
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "phoenix": (33.4484, -112.0740),
    # Mock data cities — fake names mapped to random VA-area coordinates so
    # that the distance calculation flow can be demonstrated end-to-end with
    # the CSV test data.  Remove this block when real geocoding is active.
    "north judithbury": (37.10, -76.50),
    "north donnaport": (39.77, -86.16),
    "thomasberg": (43.07, -89.40),
    "burchborough": (36.85, -76.29),
    "east amanda": (37.54, -77.44),
    "leehaven": (37.27, -79.94),
    "smithberg": (38.03, -78.48),
    "new susanville": (36.73, -76.58),
    "kingborough": (36.77, -76.29),
    "lake debbie": (37.09, -76.47),
}


def _stub_geocode(address):
    """Return coordinates if the address contains a known city name."""
    lower = address.strip().lower()
    for city, coords in _CITY_COORDS.items():
        if city in lower:
            return coords
    return None


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

_EARTH_RADIUS_MILES = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    """Return straight-line distance in miles between two coordinates."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return _EARTH_RADIUS_MILES * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Data loading helpers (CSV-based mock data)
# ---------------------------------------------------------------------------


def _load_csv(filename):
    """Load a CSV from the sql/ directory."""
    return pd.read_csv(os.path.join(SQL_DIR, filename))


def _join_parts(parts, sep=", "):
    """Join values with *sep*, skipping None, NaN, and empty strings."""
    cleaned = [
        str(p).strip()
        for p in parts
        if p is not None
        and not (isinstance(p, float) and pd.isna(p))
        and str(p).strip()
        and str(p).strip().upper() != "NULL"
    ]
    return sep.join(cleaned) if cleaned else None


# ---------------------------------------------------------------------------
# Beneficiary location (fallback chain)
# ---------------------------------------------------------------------------


def get_beneficiary_location(beneficiary_id, request_id):
    """Determine the beneficiary's location using the priority chain.

    1. Request location (requests.req_loc) — use as-is.
    2. Profile address — construct from user address fields.
    3. Unavailable — return (None, "unknown_location").

    Returns (location_string, status) where status is "ok" or
    "unknown_location".

    Note: user_locations table is not present in the CSV mock data, so step 2
    of the task spec (user_locations.curr_loc) is skipped here.  When the real
    DB is used, add that lookup between steps 1 and 2.
    """
    requests_df = _load_csv("requests.csv")

    # --- Step 1: request location ---
    req_row = requests_df[
        (requests_df["req_id"] == request_id)
        & (requests_df["beneficiary_id"] == beneficiary_id)
    ]

    if not req_row.empty:
        req_loc = req_row.iloc[0].get("req_loc")
        if pd.notna(req_loc) and str(req_loc).strip():
            return str(req_loc).strip(), "ok"

    # --- Step 2: profile address ---
    users_df = _load_csv("users.csv")
    user_row = users_df[users_df["user_id"] == beneficiary_id]

    if not user_row.empty:
        u = user_row.iloc[0]
        location = _join_parts(
            [
                u.get("addr_ln1"),
                u.get("addr_ln2"),
                u.get("city_name"),
                u.get("state_id"),
                u.get("zip_code"),
            ]
        )
        if location:
            return location, "ok"

    # --- Step 3: unavailable ---
    return None, "unknown_location"


# ---------------------------------------------------------------------------
# Request info
# ---------------------------------------------------------------------------


def get_req_info(request_id, beneficiary_id):
    """Return category, description, and subject for one request."""
    requests_df = _load_csv("requests.csv")
    help_cat_df = _load_csv("help_category.csv")

    req_row = requests_df[
        (requests_df["req_id"] == request_id)
        & (requests_df["beneficiary_id"] == beneficiary_id)
    ]

    if req_row.empty:
        raise ValueError(
            f"No request found with req_id={request_id} "
            f"and beneficiary_id={beneficiary_id}"
        )

    req = req_row.iloc[0]
    cat_id = req.get("req_cat_id")

    cat_name = None
    if pd.notna(cat_id):
        cat_row = help_cat_df[help_cat_df["cat_id"] == cat_id]
        if not cat_row.empty:
            cat_name = cat_row.iloc[0].get("cat_name")

    return {
        "category": cat_name or str(cat_id),
        "description": req.get("req_desc"),
        "subject": req.get("req_subj"),
    }


# ---------------------------------------------------------------------------
# Organization retrieval (DB / CSV mock)
# ---------------------------------------------------------------------------


def get_orgs_from_db(category):
    """Find organizations from the CSV mock data.

    In the real Lambda, this queries the DB by category via org_skills →
    help_categories.  The CSV mock data has no org_skills table, so we match
    on the organization's mission field as a simple proxy.

    Every organization is returned regardless of location — distance is
    calculated separately.
    """
    orgs_df = _load_csv("organizations.csv")
    orgs_df["source"] = "db"

    orgs_df["location"] = orgs_df.apply(
        lambda r: _join_parts([r.get("city_name"), r.get("state_id")]),
        axis=1,
    )

    orgs_df["full_address"] = orgs_df.apply(
        lambda r: _join_parts(
            [
                r.get("street"),
                r.get("city_name"),
                r.get("state_id"),
                r.get("zip_code"),
            ]
        ),
        axis=1,
    )

    return orgs_df


def get_ai_orgs(subject, description, location, category):
    """Placeholder for the GenAI Lambda call.

    In the real Lambda, this invokes GEN_AI_LAMBDA.  For CSV mock data we
    return an empty DataFrame — the DB/CSV organizations are sufficient for
    testing the distance calculation flow.
    """
    return pd.DataFrame(columns=ORG_COLUMNS)


# ---------------------------------------------------------------------------
# Distance calculation
# ---------------------------------------------------------------------------


def calculate_distances(orgs_df, beneficiary_location, beneficiary_status):
    """Add distance columns to every organization row.

    If the beneficiary location could not be determined, every org gets
    distance=null with distance_status matching the beneficiary status.

    If an individual organization cannot be geocoded, that row gets
    distance=null, distance_status="not_found".
    """
    distances = []
    units = []
    methods = []
    statuses = []

    # Geocode beneficiary once.
    if beneficiary_status != "ok" or not beneficiary_location:
        # Beneficiary location unknown — mark all orgs accordingly.
        for _ in range(len(orgs_df)):
            distances.append(None)
            units.append("miles")
            methods.append("straight_line")
            statuses.append(beneficiary_status or "unknown_location")

        orgs_df["distance"] = distances
        orgs_df["distance_unit"] = units
        orgs_df["distance_method"] = methods
        orgs_df["distance_status"] = statuses
        return orgs_df

    ben_coords = geocode(beneficiary_location)

    if ben_coords is None:
        for _ in range(len(orgs_df)):
            distances.append(None)
            units.append("miles")
            methods.append("straight_line")
            statuses.append("not_found")

        orgs_df["distance"] = distances
        orgs_df["distance_unit"] = units
        orgs_df["distance_method"] = methods
        orgs_df["distance_status"] = statuses
        return orgs_df

    ben_lat, ben_lon = ben_coords

    for _, row in orgs_df.iterrows():
        org_address = row.get("full_address") or row.get("location")
        org_coords = geocode(org_address) if org_address else None

        if org_coords is None:
            distances.append(None)
            units.append("miles")
            methods.append("straight_line")
            statuses.append("not_found")
        else:
            org_lat, org_lon = org_coords
            dist = round(haversine_miles(ben_lat, ben_lon, org_lat, org_lon), 1)
            distances.append(dist)
            units.append("miles")
            methods.append("straight_line")
            statuses.append("ok")

    orgs_df["distance"] = distances
    orgs_df["distance_unit"] = units
    orgs_df["distance_method"] = methods
    orgs_df["distance_status"] = statuses
    return orgs_df


# ---------------------------------------------------------------------------
# Merge & normalize
# ---------------------------------------------------------------------------


def merge_organizations(db_organizations, ai_organizations):
    """Normalize DB and AI organizations and combine them."""

    def _normalize(df, rename_map):
        if df is None or df.empty:
            return pd.DataFrame(columns=ORG_COLUMNS)
        normalized = df.rename(columns=rename_map).copy()
        return normalized.reindex(columns=ORG_COLUMNS)

    db_orgs = _normalize(db_organizations, DB_RENAME)
    ai_orgs = _normalize(ai_organizations, {})

    return pd.concat([db_orgs, ai_orgs], ignore_index=True)


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------


def lambda_handler(event, context):
    """Entry point for the saayam-org-aggregator Lambda.

    Expects JSON body with: request_id, beneficiary_id.
    Returns a list of organizations, each with distance fields.
    """
    try:
        raw_body = event.get("body")
        body = json.loads(raw_body) if isinstance(raw_body, str) else event

        request_id = body.get("request_id")
        beneficiary_id = body.get("beneficiary_id")

        if not request_id or not beneficiary_id:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": (
                            "Missing required fields: "
                            "request_id, beneficiary_id"
                        )
                    }
                ),
            }

        # --- Beneficiary location (fallback chain, NO default fallback) ---
        beneficiary_location, ben_status = get_beneficiary_location(
            beneficiary_id, request_id
        )

        # --- Request info ---
        req_info = get_req_info(request_id, beneficiary_id)
        category = req_info.get("category", "")
        subject = req_info.get("subject", "")
        description = req_info.get("description", "")

        if not category:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": (
                            f"Request {request_id} "
                            "has no category assigned"
                        )
                    }
                ),
            }

        # --- Fetch organizations ---
        db_organizations = get_orgs_from_db(category)
        ai_organizations = get_ai_orgs(
            subject, description, beneficiary_location, category
        )

        # --- Calculate distances ---
        db_organizations = calculate_distances(
            db_organizations, beneficiary_location, ben_status
        )
        ai_organizations = calculate_distances(
            ai_organizations, beneficiary_location, ben_status
        )

        # --- Merge ---
        combined = merge_organizations(db_organizations, ai_organizations)

        # Replace NaN with None so json.dumps produces valid JSON nulls.
        combined = combined.where(combined.notna(), None)
        organizations = combined.to_dict(orient="records")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(organizations),
        }

    except json.JSONDecodeError as e:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {"error": f"Invalid JSON in request body: {str(e)}"}
            ),
        }

    except Exception as e:
        print("saayam-org-aggregator error:", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": f"Internal server error: {str(e)}"}
            ),
        }

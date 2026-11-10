from pathlib import Path
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Dataset paths
CPCB_FILE = (
    BASE_DIR
    / "outputs"
    / "person1_handoff"
    / "historical_2021_2024"
    / "cpcb"
    / "cpcb_regional_daily_measurements_2021_2024.csv"
)

NO2_FILE = (
    BASE_DIR
    / "outputs"
    / "sentinel5p"
    / "STRATA_NO2_2024_VERIFIED_ANOMALIES_V2.csv"
)

HCHO_FILE = (
    BASE_DIR
    / "outputs"
    / "sentinel5p"
    / "STRATA_HCHO_2024_VERIFIED_ANOMALIES_V2.csv"
)

VIIRS_FILE = (
    BASE_DIR
    / "outputs"
    / "person1_handoff"
    / "historical_2021_2024"
    / "viirs"
    / "viirs_historical_detections_2021_2024.csv"
)


# Create FastAPI application
app = FastAPI(
    title="STRATA Environmental Intelligence API",
    description="API for CPCB, Sentinel-5P and VIIRS environmental datasets",
    version="1.0.0"
)


# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "STRATA Environmental Intelligence",
        "status": "Backend running",
        "message": "Environmental datasets API is ready"
    }


# --------------------------------------------------
# CPCB DATA ENDPOINT
# --------------------------------------------------

@app.get("/api/cpcb")
def get_cpcb(
    date: str = "2024-11-17",
    pollutant: str = "PM2.5"
):
    df = pd.read_csv(CPCB_FILE)

    df["date"] = pd.to_datetime(
        df["period_start"]
    ).dt.strftime("%Y-%m-%d")

    filtered = df[
        (df["date"] == date) &
        (df["parameter_name"] == pollutant)
    ].copy()

    filtered = filtered.dropna(
        subset=["latitude", "longitude", "mean"]
    )

    columns = [
        "station_id",
        "station_name",
        "state_name",
        "city_name",
        "latitude",
        "longitude",
        "parameter_name",
        "mean",
        "unit",
        "n",
        "coverage_flag"
    ]

    return {
        "date": date,
        "pollutant": pollutant,
        "count": len(filtered),
        "stations": filtered[columns].to_dict(orient="records")
    }


# --------------------------------------------------
# SENTINEL-5P SATELLITE DATA ENDPOINT
# --------------------------------------------------

@app.get("/api/satellite")
def get_satellite(
    pollutant: str = "NO2",
    window_id: str = "11-12"
):
    pollutant = pollutant.upper()

    if pollutant == "NO2":
        file_path = NO2_FILE
        value_column = "tropospheric_NO2_column_number_density"

    elif pollutant == "HCHO":
        file_path = HCHO_FILE
        value_column = "tropospheric_HCHO_column_number_density"

    else:
        return {
            "error": "Invalid pollutant. Use NO2 or HCHO."
        }

    df = pd.read_csv(file_path)

    filtered = df[
        df["window_id"] == window_id
    ].copy()

    columns = [
        "point_id",
        "grid_lat",
        "grid_lon",
        "pollutant",
        value_column,
        "unit",
        "window_start",
        "window_end_exclusive",
        "window_id",
        "baseline_n",
        "baseline_mean",
        "anomaly_difference",
        "anomaly_z_score",
        "anomaly_direction",
        "baseline_status",
        "negative_retrieval",
        "score_available"
    ]

    filtered = filtered[columns]

    filtered = filtered.astype(object).where(
        pd.notna(filtered), None
    )

    return {
        "pollutant": pollutant,
        "window_id": window_id,
        "count": len(filtered),
        "points": filtered.to_dict(orient="records")
    }


# --------------------------------------------------
# VIIRS DATA LOADER
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_viirs_data():

    columns = [
        "latitude",
        "longitude",
        "bright_ti4",
        "acq_date",
        "acq_time",
        "satellite",
        "confidence",
        "frp",
        "daynight",
        "type",
        "timestamp"
    ]

    df = pd.read_csv(
        VIIRS_FILE,
        usecols=columns
    )

    df["date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    return df


# --------------------------------------------------
# VIIRS FIRE DETECTION ENDPOINT
# --------------------------------------------------

@app.get("/api/fires")
def get_fires(
    date: str = "2024-11-17",
    min_lat: float = 27.0,
    max_lat: float = 32.0,
    min_lon: float = 73.0,
    max_lon: float = 78.0,
    limit: int = 5000
):
    df = load_viirs_data()

    filtered = df[
        (df["date"] == date) &
        (df["latitude"] >= min_lat) &
        (df["latitude"] <= max_lat) &
        (df["longitude"] >= min_lon) &
        (df["longitude"] <= max_lon)
    ].copy()

    total_count = len(filtered)

    filtered = filtered.head(
        max(1, min(limit, 10000))
    )

    columns = [
        "latitude",
        "longitude",
        "bright_ti4",
        "date",
        "acq_time",
        "satellite",
        "confidence",
        "frp",
        "daynight",
        "type",
        "timestamp"
    ]

    filtered = filtered[columns]

    filtered = filtered.astype(object).where(
        pd.notna(filtered), None
    )

    return {
        "date": date,
        "total_count": total_count,
        "returned_count": len(filtered),
        "detections": filtered.to_dict(orient="records")
    }
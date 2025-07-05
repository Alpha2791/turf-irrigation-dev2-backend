from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import requests
import pandas as pd
import os
from models import MoistureLog, IrrigationLog, WeatherHistory, PredictionMeta
from database import Base, engine, SessionLocal
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://turf-tracker-dev2.netlify.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LATITUDE = 50.415642
LONGITUDE = -5.092041
ELEVATION = 39  # metres
VC_API_KEY = "2ELL5E9A47JT5XB74WGXS7PFV"

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

def get_last_weather_timestamp(db):
    entry = db.query(PredictionMeta).filter_by(key="last_weather_timestamp").first()
    return datetime.fromisoformat(entry.value) if entry else None

def set_last_weather_timestamp(db, timestamp):
    entry = db.query(PredictionMeta).filter_by(key="last_weather_timestamp").first()
    if entry:
        entry.value = timestamp.isoformat()
    else:
        entry = PredictionMeta(key="last_weather_timestamp", value=timestamp.isoformat())
        db.add(entry)
    db.commit()

@app.get("/moisture-log")
def get_moisture_log():
    db = SessionLocal()
    entries = db.query(MoistureLog).order_by(MoistureLog.timestamp.desc()).all()
    db.close()
    return [{"timestamp": e.timestamp.isoformat(), "moisture_mm": e.moisture_mm} for e in entries]

@app.get("/irrigation-log")
def get_irrigation_log():
    db = SessionLocal()
    entries = db.query(IrrigationLog).order_by(IrrigationLog.timestamp.desc()).all()
    db.close()
    return [{"timestamp": e.timestamp.isoformat(), "irrigation_mm": e.irrigation_mm} for e in entries]

@app.post("/log-moisture")
def log_moisture(request: Request, timestamp: str = Body(...), moisture_mm: float = Body(...)):
    db = SessionLocal()
    dt = datetime.fromisoformat(timestamp)
    entry = MoistureLog(timestamp=dt, moisture_mm=moisture_mm)
    db.merge(entry)
    db.commit()
    db.close()
    return {"status": "moisture logged"}

@app.post("/log-irrigation")
def log_irrigation(request: Request, timestamp: str = Body(...), irrigation_mm: float = Body(...)):
    db = SessionLocal()
    dt = datetime.fromisoformat(timestamp)
    entry = IrrigationLog(timestamp=dt, irrigation_mm=irrigation_mm)
    db.merge(entry)
    db.commit()
    db.close()
    return {"status": "irrigation logged"}

def calculate_et_fao56(temp, humidity, windspeed, solar_radiation, pressure=101.3, elevation=ELEVATION):
    # Constants
    Gsc = 0.0820  # MJ m^-2 min^-1
    gamma = 0.665e-3 * pressure  # kPa/°C
    lambda_ = 2.45  # latent heat of vaporization [MJ/kg]
    
    es = 0.6108 * math.exp((17.27 * temp) / (temp + 237.3))  # saturation vapour pressure
    ea = es * (humidity / 100)  # actual vapour pressure
    delta = (4098 * es) / ((temp + 237.3) ** 2)  # slope of saturation vapour pressure curve
    
    Rn = solar_radiation * 0.0036  # convert W/m² to MJ/m²/hr
    u2 = windspeed  # assumed at 2m

    et0 = (0.408 * delta * Rn + gamma * (900 / (temp + 273)) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
    return round(et0, 3)

@app.get("/predicted-moisture")
def predicted_moisture():
    return get_predicted_moisture()

def get_predicted_moisture():
    print("[INFO] Running /predicted-moisture")
    try:
        db = SessionLocal()
        now = datetime.utcnow()

        moist_entries = db.query(MoistureLog).order_by(MoistureLog.timestamp.asc()).all()
        irrig_entries = db.query(IrrigationLog).all()

        df_moist = pd.DataFrame([
            {"timestamp": e.timestamp, "moisture_mm": e.moisture_mm} for e in moist_entries
        ]).set_index("timestamp") if moist_entries else pd.DataFrame(columns=["moisture_mm"])

        df_irrig = pd.DataFrame([
            {"timestamp": e.timestamp, "irrigation_mm": e.irrigation_mm} for e in irrig_entries
        ]).set_index("timestamp") if irrig_entries else pd.DataFrame(columns=["irrigation_mm"])

        latest_log_ts = df_moist.index[-1] if not df_moist.empty else now
        start_date = latest_log_ts.strftime("%Y-%m-%d")
        end_date = (now + timedelta(days=5)).strftime("%Y-%m-%d")

        url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
            f"{LATITUDE},{LONGITUDE}/{start_date}/{end_date}"
            f"?unitGroup=metric&key={VC_API_KEY}&include=hours"
            f"&elements=datetime,temp,humidity,windspeed,solarradiation,precip"
        )
        response = requests.get(url)
        data = response.json()

        weather_data = []
        new_ts = None

        for day in data.get("days", []):
            for hour in day.get("hours", []):
                raw_ts = f"{day['datetime']}T{hour['datetime'][:5]}"
                timestamp = datetime.strptime(raw_ts, "%Y-%m-%dT%H:%M")

                temp = hour.get("temp", 0)
                humidity = hour.get("humidity", 0)
                windspeed = hour.get("windspeed", 0)
                solar_radiation = hour.get("solarradiation", 0) or 0
                rainfall = hour.get("precip", 0) or 0

                et = calculate_et_fao56(temp, humidity, windspeed, solar_radiation)

                weather_data.append({
                    "timestamp": raw_ts,
                    "ET_mm_hour": et,
                    "rainfall_mm": rainfall,
                    "irrigation_mm": 0  # to be filled later
                })

                if not db.query(WeatherHistory).filter_by(timestamp=timestamp).first():
                    try:
                        db.add(WeatherHistory(
                            timestamp=timestamp,
                            et_mm_hour=et,
                            rainfall_mm=rainfall,
                            solar_radiation=solar_radiation,
                            temp_c=temp,
                            humidity=humidity,
                            windspeed=windspeed,
                        ))
                        new_ts = timestamp
                    except Exception as e:
                        db.rollback()
                        print(f"[SKIP] Failed to add weather entry for {timestamp}: {e}")

        if new_ts:
            set_last_weather_timestamp(db, new_ts)
        db.close()

        # Convert weather data to DataFrame
        df_weather = pd.DataFrame(weather_data)
        df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])
        df_weather.set_index("timestamp", inplace=True)

        # Drop irrigation_mm if it already exists in weather data
        if "irrigation_mm" in df_weather.columns:
            df_weather.drop(columns=["irrigation_mm"], inplace=True)

        # Merge irrigation data (avoiding column name conflicts)
        df_combined = df_weather.merge(df_irrig, how="left", left_index=True, right_index=True)
        df_combined["irrigation_mm"] = df_combined["irrigation_mm"].fillna(0)
        df_combined = df_combined.sort_index()
        df = df_combined


        print("[INFO] Forecast dataframe shape:", df.shape)

        results = []
        last_pred = df_moist.iloc[-1]["moisture_mm"] if not df_moist.empty else 25.0

        for ts, row in df_combined.iterrows():
            et_mm = row["ET_mm_hour"]
            rainfall_mm = row["rainfall_mm"]
            irrigation_mm = row["irrigation_mm"]

            predicted_moisture = last_pred - et_mm + rainfall_mm + irrigation_mm
            predicted_moisture = max(min(predicted_moisture, 100), 0)
            last_pred = predicted_moisture

            results.append({
                "timestamp": ts.strftime("%Y-%m-%dT%H"),
                "ET_mm_hour": round(et_mm, 3),
                "rainfall_mm": round(rainfall_mm, 2),
                "irrigation_mm": round(irrigation_mm, 2),
                "predicted_moisture_mm": round(predicted_moisture, 1)
            })

        print(f"[INFO] Returning {len(results)} predicted moisture points")
        return results

    except Exception as e:
        print(f"[ERROR] Unexpected error in predicted moisture: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/wilt-forecast")
def get_wilt_forecast(wilt_point: float = 18.0, upper_limit: float = 22.0):
    predictions = get_predicted_moisture()

    for row in predictions:
        moisture = row.get("predicted_moisture_mm")
        if moisture is None:
            continue

        if moisture < wilt_point:
            ts = row.get("timestamp", "unknown time")
            rec_irrig = upper_limit - moisture
            return {
                "wilt_point_hit": ts,
                "recommended_irrigation_mm": round(rec_irrig, 1),
                "upper_limit": upper_limit,
                "message": f"Apply approx {round(rec_irrig, 1)} mm to reach {upper_limit}%"
            }

    return {
        "wilt_point_hit": None,
        "recommended_irrigation_mm": None,
        "upper_limit": upper_limit,
        "message": "No wilt point drop expected in forecast."
    }

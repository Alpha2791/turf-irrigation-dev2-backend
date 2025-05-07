from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import requests
import pandas as pd
import joblib
import xgboost as xgb
import os

from models import MoistureLog, IrrigationLog
from database import Base, engine, SessionLocal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://turf-tracker-dev2.netlify.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "moisture_model.pkl"
LATITUDE = 51.678929
LONGITUDE = -1.362154
ELEVATION = 69
VC_API_KEY = "2ELL5E9A47JT5XB74WGXS7PFV"

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

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
def log_moisture(timestamp: str = Body(...), moisture_mm: float = Body(...)):
    db = SessionLocal()
    dt = datetime.fromisoformat(timestamp)
    entry = MoistureLog(timestamp=dt, moisture_mm=moisture_mm)
    db.merge(entry)
    db.commit()
    db.close()
    return {"status": "moisture logged"}

@app.post("/log-irrigation")
def log_irrigation(timestamp: str = Body(...), irrigation_mm: float = Body(...)):
    db = SessionLocal()
    dt = datetime.fromisoformat(timestamp)
    entry = IrrigationLog(timestamp=dt, irrigation_mm=irrigation_mm)
    db.merge(entry)
    db.commit()
    db.close()
    return {"status": "irrigation logged"}

@app.get("/predicted-moisture")
def get_predicted_moisture():
    print("Running /predicted-moisture endpoint")

    if not os.path.exists(MODEL_FILE):
        raise HTTPException(status_code=500, detail="Model file not found")

    try:
        model = joblib.load(MODEL_FILE)
        print("Model loaded successfully")

        now = datetime.utcnow()
        start_date = (now - timedelta(days=3)).strftime("%Y-%m-%d")
        end_date = (now + timedelta(days=5)).strftime("%Y-%m-%d")

        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LATITUDE},{LONGITUDE}/{start_date}/{end_date}?unitGroup=metric&key={VC_API_KEY}&include=hours&elements=datetime,temp,humidity,windspeed,solarradiation,precip"
        print(f"Weather API URL: {url}")

        response = requests.get(url)
        print(f"Weather API status code: {response.status_code}")

        if response.status_code != 200:
            print(f"Weather API response error: {response.text}")
            raise HTTPException(status_code=500, detail=f"Weather API error: {response.status_code} - {response.text}")

        data = response.json()

        weather_data = []
        for day in data.get("days", []):
            for hour in day.get("hours", []):
                raw_ts = f"{day['datetime']}T{hour['datetime'][:5]}"
                solar_radiation = hour.get("solarradiation", 0) or 0
                et = round(0.408 * solar_radiation / 1000, 3)
                weather_data.append({
                    "timestamp": raw_ts,
                    "ET_mm_hour": et,
                    "rainfall_mm": hour.get("precip", 0) or 0
                })

        df_weather = pd.DataFrame(weather_data)
        df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"], format="%Y-%m-%dT%H:%M", errors="coerce")
        df_weather.dropna(subset=["timestamp"], inplace=True)
        df_weather.set_index("timestamp", inplace=True)

        db = SessionLocal()
        moisture_entries = db.query(MoistureLog).order_by(MoistureLog.timestamp).all()
        irrigation_entries = db.query(IrrigationLog).order_by(IrrigationLog.timestamp).all()
        db.close()

        df_moist = pd.DataFrame([{"timestamp": e.timestamp, "moisture_mm": e.moisture_mm} for e in moisture_entries])
        df_irrig = pd.DataFrame([{"timestamp": e.timestamp, "irrigation_mm": e.irrigation_mm} for e in irrigation_entries])

        if not df_moist.empty:
            df_moist.set_index("timestamp", inplace=True)
        else:
            df_moist = pd.DataFrame(columns=["moisture_mm"])
            df_moist.index.name = "timestamp"

        if not df_irrig.empty:
            df_irrig.set_index("timestamp", inplace=True)
        else:
            df_irrig = pd.DataFrame(columns=["irrigation_mm"])
            df_irrig.index.name = "timestamp"

        df = df_weather.join(df_irrig, how="left").fillna({"irrigation_mm": 0})
        df = df.sort_index()

        results = []
        last_pred = df_moist.iloc[-1]["moisture_mm"] if not df_moist.empty else 25.0
        sample_count = len(df_moist)

        print(f"Sample count: {sample_count}")

        for ts, row in df.iterrows():
            hour = ts.hour
            dayofyear = ts.dayofyear
            irrigation_mm = row["irrigation_mm"]
            rainfall_mm = row.get("rainfall_mm", 0)
            et_mm = row.get("ET_mm_hour", 0)

            features = pd.DataFrame([{ 
                "prev_moisture": last_pred, 
                "irrigation_mm": irrigation_mm, 
                "hour": hour, 
                "dayofyear": dayofyear
            }])

            model_pred = model.predict(features)[0]
            basic_estimate = last_pred - et_mm + rainfall_mm + irrigation_mm
            alpha = min(sample_count / 50, 1.0)
            predicted_moisture = (alpha * model_pred) + ((1 - alpha) * basic_estimate)
            predicted_moisture = max(min(predicted_moisture, 100), 0)

            results.append({
                "timestamp": ts.strftime("%Y-%m-%dT%H"),
                "ET_mm_hour": et_mm,
                "rainfall_mm": rainfall_mm,
                "irrigation_mm": irrigation_mm,
                "predicted_moisture_mm": round(float(predicted_moisture), 1)
            })

            last_pred = predicted_moisture

        print(f"Returning {len(results)} forecast points.")
        return results

    except Exception as e:
        print(f"Unexpected error in predicted moisture: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/wilt-forecast")
def get_wilt_forecast(wilt_point: float = 18.0, upper_limit: float = 22.0):
    predictions = get_predicted_moisture()

    for row in predictions:
        moisture = row["predicted_moisture_mm"]
        if moisture < wilt_point:
            ts = row["timestamp"]
            rec_irrig = upper_limit - moisture
            return {
                "wilt_point_hit": ts,
                "recommended_irrigation_mm": round(rec_irrig, 1),
                "upper_limit": upper_limit,
                "message": f"Apply approx {round(rec_irrig, 1)} mm to reach {upper_limit}%"
            }

    return {"message": "No wilt point drop expected in forecast."}

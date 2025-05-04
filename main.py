




from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

import openrouteservice
from openrouteservice.exceptions import ApiError

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import logging
from dotenv import load_dotenv
import os




app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    # allow_origins=["https://mahros-alqabasy.github.io"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("urgentroute")




load_dotenv()
API_KEY = os.getenv("ORS_API_KEY")
client = openrouteservice.Client(key=API_KEY)


hospital_data = pd.read_csv("us_hospital_locations.csv")[['NAME', 'LATITUDE', 'LONGITUDE']].dropna()
hospital_data.columns = ['Hospital Name', 'Latitude', 'Longitude']
coords = hospital_data[['Latitude', 'Longitude']].to_numpy()
kdtree = KDTree(np.radians(coords), metric='euclidean')




def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

class UserLocation(BaseModel):
    latitude: float
    longitude: float




@app.exception_handler(ApiError)
async def openrouteservice_api_error_handler(request: Request, exc: ApiError):
    logger.error(f"OpenRouteService API error: {exc}", exc_info=True)
    try:
        detail = exc.args[0]
        if isinstance(detail, dict):
            code = detail.get("error", {}).get("code")
            if code == 2004:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "The route is too long. Please choose a closer location."}
                )
            return JSONResponse(
                status_code=400,
                content={"detail": detail.get("error", {}).get("message", "Routing error.")}
            )
        return JSONResponse(status_code=400, content={"detail": str(detail)})
    except Exception:
        logger.error("Failed to parse ORS ApiError", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Unexpected routing error. Please try again later."}
        )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input.", "errors": exc.errors()}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )





@app.get("/nearest-hospital")
def get_nearest_hospital(lat: float = Query(...), lon: float = Query(...)):
    user_location = (lat, lon)
    user_rad = np.radians([user_location])
    dist_rad, index = kdtree.query(user_rad, k=1)
    idx = index[0][0]
    hospital = hospital_data.iloc[idx]
    dist_km = dist_rad[0][0] * 6371

    hospital_coords = (hospital['Latitude'], hospital['Longitude'])
    coords = [user_location[::-1], hospital_coords[::-1]]

    route = client.directions(coordinates=coords, profile='driving-car', format='geojson')
    geometry = route['features'][0]['geometry']
    props = route['features'][0]['properties']['segments'][0]
    distance_km = props['distance'] / 1000
    duration_min = props['duration'] / 60

    return {
        "hospital": hospital["Hospital Name"],
        "location": [hospital["Latitude"], hospital["Longitude"]],
        "distance_km": round(distance_km, 2),
        "duration_min": round(duration_min, 1),
        "route_geometry": geometry
    }

@app.post("/route")
def get_route(user: UserLocation, hospital: UserLocation):
    coords = [(user.longitude, user.latitude), (hospital.longitude, hospital.latitude)]
    route = client.directions(coordinates=coords, profile='driving-car', format='geojson')
    segment = route['features'][0]['properties']['segments'][0]
    return {
        "distance_km": round(segment["distance"] / 1000, 2),
        "duration_min": round(segment["duration"] / 60, 1),
        "geometry": route["features"][0]["geometry"]
    }

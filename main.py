# Create by Mahros
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from dotenv import load_dotenv
import os
import openrouteservice

app = FastAPI()




from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load environment and data
load_dotenv()
API_KEY = os.getenv("ORS_API_KEY")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
client = openrouteservice.Client(key=API_KEY)

# Load hospital data and prepare KDTree
hospital_data = pd.read_csv("us_hospital_locations.csv")[['NAME', 'LATITUDE', 'LONGITUDE']].dropna()
hospital_data.columns = ['Hospital Name', 'Latitude', 'Longitude']
coords = hospital_data[['Latitude', 'Longitude']].to_numpy()
kdtree = KDTree(np.radians(coords), metric='euclidean')

# Input model for location
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
    properties = route['features'][0]['properties']['segments'][0]
    distance_km = properties['distance'] / 1000
    duration_min = properties['duration'] / 60

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
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
import folium
import uuid

app = FastAPI()

# Mount static file directory for map HTMLs
os.makedirs("maps", exist_ok=True)
app.mount("/maps", StaticFiles(directory="maps"), name="maps")

# Load environment and data
load_dotenv()
API_KEY = os.getenv("ORS_API_KEY")
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

    # Generate map
    m = folium.Map(location=user_location, zoom_start=13)
    folium.Marker(location=user_location, popup="User Location", icon=folium.Icon(color='blue')).add_to(m)
    folium.Marker(location=hospital_coords, popup=hospital['Hospital Name'], icon=folium.Icon(color='red')).add_to(m)
    folium.GeoJson(geometry, name='route').add_to(m)

    map_id = str(uuid.uuid4())
    map_filename = f"map_{map_id}.html"
    map_path = os.path.join("maps", map_filename)
    m.save(map_path)

    return {
        "hospital": hospital["Hospital Name"],
        "location": [hospital["Latitude"], hospital["Longitude"]],
        "distance_km": round(distance_km, 2),
        "duration_min": round(duration_min, 1),
        "map_url": f"/maps/{map_filename}"
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
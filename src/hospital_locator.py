"""
Real-Time Nearby Hospital Finder
Uses OpenStreetMap Overpass API (free, no API key required).
Optionally supports Google Places API via GOOGLE_MAPS_API_KEY env variable.
"""

import os
import math
import httpx
import asyncio
from typing import Optional
from pydantic import BaseModel, Field


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class HospitalRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="User latitude")
    longitude: float = Field(..., ge=-180, le=180, description="User longitude")
    radius_km: float = Field(default=10.0, ge=0.5, le=50.0, description="Search radius in km")
    max_results: int = Field(default=10, ge=1, le=25)
    provider: str = Field(default="osm", description="'osm' (OpenStreetMap) or 'google'")


class Hospital(BaseModel):
    name: str
    address: str
    phone: Optional[str] = None
    distance_km: float
    latitude: float
    longitude: float
    maps_url: str
    emergency: Optional[bool] = None   # True if explicitly tagged as emergency facility


class HospitalResponse(BaseModel):
    total_found: int
    radius_km: float
    user_location: dict
    hospitals: list[Hospital]
    provider: str
    note: str


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in km between two GPS coordinates."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── OpenStreetMap (Overpass API) provider ─────────────────────────────────────

OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

async def _fetch_osm_hospitals(
    lat: float, lon: float, radius_m: int
) -> list[dict]:
    """Query Overpass API for hospitals/clinics within radius."""
    # Query nodes, ways, and relations tagged as hospitals or clinics
    query = f"""
[out:json][timeout:25];
(
  node["amenity"~"hospital|clinic|doctors"](around:{radius_m},{lat},{lon});
  way["amenity"~"hospital|clinic|doctors"](around:{radius_m},{lat},{lon});
  relation["amenity"~"hospital|clinic|doctors"](around:{radius_m},{lat},{lon});
  node["healthcare"~"hospital|clinic|centre"](around:{radius_m},{lat},{lon});
  way["healthcare"~"hospital|clinic|centre"](around:{radius_m},{lat},{lon});
);
out center tags;
"""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(OSM_OVERPASS_URL, data={"data": query})
        resp.raise_for_status()
        return resp.json().get("elements", [])


def _parse_osm_element(el: dict, user_lat: float, user_lon: float) -> Optional[Hospital]:
    tags = el.get("tags", {})
    name = tags.get("name") or tags.get("name:en")
    if not name:
        return None

    # Get coordinates — nodes have lat/lon directly; ways/relations have 'center'
    if el["type"] == "node":
        e_lat, e_lon = el.get("lat"), el.get("lon")
    else:
        center = el.get("center", {})
        e_lat, e_lon = center.get("lat"), center.get("lon")

    if e_lat is None or e_lon is None:
        return None

    # Build address from available tags
    addr_parts = filter(None, [
        tags.get("addr:housenumber"),
        tags.get("addr:street"),
        tags.get("addr:suburb") or tags.get("addr:village") or tags.get("addr:city"),
        tags.get("addr:state"),
    ])
    address = ", ".join(addr_parts) or tags.get("description") or "Address not available"

    phone = tags.get("phone") or tags.get("contact:phone") or tags.get("contact:mobile")

    emergency = tags.get("emergency") == "yes" or tags.get("amenity") == "hospital"

    maps_url = f"https://www.openstreetmap.org/?mlat={e_lat}&mlon={e_lon}#map=17/{e_lat}/{e_lon}"

    dist = haversine_km(user_lat, user_lon, e_lat, e_lon)

    return Hospital(
        name=name,
        address=address,
        phone=phone,
        distance_km=round(dist, 2),
        latitude=e_lat,
        longitude=e_lon,
        maps_url=maps_url,
        emergency=emergency,
    )


async def search_hospitals_osm(req: HospitalRequest) -> HospitalResponse:
    radius_m = int(req.radius_km * 1000)
    elements = await _fetch_osm_hospitals(req.latitude, req.longitude, radius_m)

    hospitals: list[Hospital] = []
    seen_names: set[str] = set()

    for el in elements:
        h = _parse_osm_element(el, req.latitude, req.longitude)
        if h and h.name not in seen_names:
            hospitals.append(h)
            seen_names.add(h.name)

    # Sort by distance; put hospitals with emergency=True first when tied
    hospitals.sort(key=lambda h: (h.distance_km, not (h.emergency or False)))
    hospitals = hospitals[: req.max_results]

    return HospitalResponse(
        total_found=len(hospitals),
        radius_km=req.radius_km,
        user_location={"latitude": req.latitude, "longitude": req.longitude},
        hospitals=hospitals,
        provider="OpenStreetMap (Overpass API)",
        note=(
            "Results sourced from OpenStreetMap community data. "
            "Contact details may be incomplete. In an emergency, call 112 immediately."
        ),
    )


# ── Google Places API provider ────────────────────────────────────────────────

GOOGLE_PLACES_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GOOGLE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


async def _get_google_phone(place_id: str, api_key: str) -> Optional[str]:
    """Fetch phone number via Place Details API."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            GOOGLE_DETAILS_URL,
            params={
                "place_id": place_id,
                "fields": "formatted_phone_number",
                "key": api_key,
            },
        )
        data = resp.json()
        return data.get("result", {}).get("formatted_phone_number")


async def search_hospitals_google(req: HospitalRequest) -> HospitalResponse:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_MAPS_API_KEY environment variable is not set. "
            "Set it or use provider='osm' for the free OpenStreetMap option."
        )

    radius_m = int(req.radius_km * 1000)
    hospitals: list[Hospital] = []

    async with httpx.AsyncClient(timeout=20) as client:
        params = {
            "location": f"{req.latitude},{req.longitude}",
            "radius": radius_m,
            "type": "hospital",
            "key": api_key,
        }
        resp = await client.get(GOOGLE_PLACES_URL, params=params)
        resp.raise_for_status()
        results = resp.json().get("results", [])

    # Fetch phone numbers concurrently (up to max_results)
    results = results[: req.max_results]
    phone_tasks = [_get_google_phone(r["place_id"], api_key) for r in results]
    phones = await asyncio.gather(*phone_tasks, return_exceptions=True)

    for place, phone in zip(results, phones):
        geo = place["geometry"]["location"]
        e_lat, e_lon = geo["lat"], geo["lng"]

        address = place.get("vicinity") or "Address not available"
        dist = haversine_km(req.latitude, req.longitude, e_lat, e_lon)
        maps_url = (
            f"https://www.google.com/maps/search/?api=1"
            f"&query={e_lat},{e_lon}&query_place_id={place['place_id']}"
        )

        hospitals.append(
            Hospital(
                name=place.get("name", "Unknown"),
                address=address,
                phone=phone if isinstance(phone, str) else None,
                distance_km=round(dist, 2),
                latitude=e_lat,
                longitude=e_lon,
                maps_url=maps_url,
                emergency=None,
            )
        )

    hospitals.sort(key=lambda h: h.distance_km)

    return HospitalResponse(
        total_found=len(hospitals),
        radius_km=req.radius_km,
        user_location={"latitude": req.latitude, "longitude": req.longitude},
        hospitals=hospitals,
        provider="Google Places API",
        note="In a snakebite emergency, call 112 / local emergency number immediately.",
    )


# ── Public interface ──────────────────────────────────────────────────────────

async def find_nearby_hospitals(req: HospitalRequest) -> HospitalResponse:
    """Dispatch to the correct provider."""
    if req.provider.lower() == "google":
        return await search_hospitals_google(req)
    else:
        return await search_hospitals_osm(req)

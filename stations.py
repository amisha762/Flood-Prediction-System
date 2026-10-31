"""
Station Metadata and Geographical Profiles for Punjab, Haryana, Madhya Pradesh, and Bihar.
Contains real Central Water Commission (CWC) monitoring stations, baseline danger levels,
warning levels, catchment elevations, and hydrological baseline parameters.
"""

STATIONS_DATA = {
    # ==================== PUNJAB ====================
    "Punjab": {
        "Jalandhar": {
            "Phillaur": {
                "river": "Sutlej",
                "basin": "Indus Basin",
                "warning_level_m": 234.50,
                "danger_level_m": 235.60,
                "high_flood_level_m": 236.80,
                "safe_discharge_cumec": 4500.0,
                "catchment_elevation_m": 240.0,
                "catchment_slope_deg": 1.2,
                "distance_to_river_km": 0.5,
                "latitude": 31.0210,
                "longitude": 75.7860,
                "upstream_station": "Ropar Headworks",
                "upstream_distance_km": 85.0,
                "dam_reservoir": "Bhakra Nangal",
                "historical_flood_count_5yr": 3,
                "description": "Critical monitoring station on Sutlej downstream of Ropar, vulnerable during high Bhakra discharge."
            }
        },
        "Tarn Taran": {
            "Harike Barrage": {
                "river": "Sutlej-Beas Confluence",
                "basin": "Indus Basin",
                "warning_level_m": 209.00,
                "danger_level_m": 210.00,
                "high_flood_level_m": 211.20,
                "safe_discharge_cumec": 7000.0,
                "catchment_elevation_m": 215.0,
                "catchment_slope_deg": 0.8,
                "distance_to_river_km": 0.3,
                "latitude": 31.1500,
                "longitude": 74.9600,
                "upstream_station": "Phillaur",
                "upstream_distance_km": 90.0,
                "dam_reservoir": "Harike Barrage",
                "historical_flood_count_5yr": 4,
                "description": "Major confluence of Sutlej and Beas. Controls water flow to Rajasthan Feeder canal."
            }
        },
        "Hoshiarpur": {
            "Pong Downstream": {
                "river": "Beas",
                "basin": "Indus Basin",
                "warning_level_m": 330.00,
                "danger_level_m": 332.50,
                "high_flood_level_m": 334.10,
                "safe_discharge_cumec": 5500.0,
                "catchment_elevation_m": 335.0,
                "catchment_slope_deg": 2.5,
                "distance_to_river_km": 0.6,
                "latitude": 31.9600,
                "longitude": 75.9500,
                "upstream_station": "Pong Dam",
                "upstream_distance_km": 15.0,
                "dam_reservoir": "Pong Dam (Maharana Pratap Sagar)",
                "historical_flood_count_5yr": 3,
                "description": "Receives spillway discharges from Pong Dam during heavy Himachal catchment precipitation."
            }
        },
        "Rupnagar": {
            "Ropar Headworks": {
                "river": "Sutlej",
                "basin": "Indus Basin",
                "warning_level_m": 260.00,
                "danger_level_m": 261.20,
                "high_flood_level_m": 262.50,
                "safe_discharge_cumec": 5000.0,
                "catchment_elevation_m": 265.0,
                "catchment_slope_deg": 1.8,
                "distance_to_river_km": 0.4,
                "latitude": 30.9700,
                "longitude": 76.5300,
                "upstream_station": "Bhakra Tailrace",
                "upstream_distance_km": 45.0,
                "dam_reservoir": "Bhakra Dam",
                "historical_flood_count_5yr": 2,
                "description": "Directly downstream of Shivalik foothills, sensitive to rapid cloudburst runoff."
            }
        }
    },

    # ==================== HARYANA ====================
    "Haryana": {
        "Yamunanagar": {
            "Hathnikund Barrage": {
                "river": "Yamuna",
                "basin": "Ganga Basin",
                "warning_level_m": 360.50,
                "danger_level_m": 362.00,
                "high_flood_level_m": 363.70,
                "safe_discharge_cumec": 7079.0,  # ~2.5 lakh cusecs
                "catchment_elevation_m": 365.0,
                "catchment_slope_deg": 3.0,
                "distance_to_river_km": 0.2,
                "latitude": 30.3160,
                "longitude": 77.5890,
                "upstream_station": "Dakpathar",
                "upstream_distance_km": 30.0,
                "dam_reservoir": "Hathnikund Barrage",
                "historical_flood_count_5yr": 4,
                "description": "Gateway barrage where Yamuna enters plains. Discharges directly govern Delhi flood risks."
            }
        },
        "Ambala": {
            "Ambala Cantt": {
                "river": "Tangri",
                "basin": "Ghaggar Basin",
                "warning_level_m": 271.00,
                "danger_level_m": 272.50,
                "high_flood_level_m": 273.80,
                "safe_discharge_cumec": 1200.0,
                "catchment_elevation_m": 275.0,
                "catchment_slope_deg": 1.5,
                "distance_to_river_km": 0.3,
                "latitude": 30.3780,
                "longitude": 76.7760,
                "upstream_station": "Pinjore Hills",
                "upstream_distance_km": 40.0,
                "dam_reservoir": "None (Ephemeral Seasonal River)",
                "historical_flood_count_5yr": 3,
                "description": "Flash-flood prone ephemeral river draining Shivalik foothills into Ambala city."
            }
        },
        "Sirsa": {
            "Ottu Weir": {
                "river": "Ghaggar",
                "basin": "Inland Ghaggar Basin",
                "warning_level_m": 191.00,
                "danger_level_m": 192.50,
                "high_flood_level_m": 193.80,
                "safe_discharge_cumec": 2500.0,
                "catchment_elevation_m": 195.0,
                "catchment_slope_deg": 0.5,
                "distance_to_river_km": 0.4,
                "latitude": 29.4900,
                "longitude": 74.9000,
                "upstream_station": "Moonak",
                "upstream_distance_km": 110.0,
                "dam_reservoir": "Ottu Reservoir",
                "historical_flood_count_5yr": 3,
                "description": "Water terminal weir for Ghaggar river near Rajasthan border."
            }
        },
        "Panipat": {
            "Panipat Downstream": {
                "river": "Yamuna",
                "basin": "Ganga Basin",
                "warning_level_m": 230.50,
                "danger_level_m": 231.80,
                "high_flood_level_m": 233.00,
                "safe_discharge_cumec": 6000.0,
                "catchment_elevation_m": 233.0,
                "catchment_slope_deg": 0.9,
                "distance_to_river_km": 0.8,
                "latitude": 29.3900,
                "longitude": 77.0800,
                "upstream_station": "Hathnikund Barrage",
                "upstream_distance_km": 120.0,
                "dam_reservoir": "None",
                "historical_flood_count_5yr": 2,
                "description": "Downstream Yamuna plain station reflecting Hathnikund release travel times."
            }
        }
    },

    # ==================== MADHYA PRADESH ====================
    "Madhya Pradesh": {
        "Narmadapuram": {
            "Hoshangabad Ghat": {
                "river": "Narmada",
                "basin": "Narmada Basin",
                "warning_level_m": 292.00,
                "danger_level_m": 293.83,
                "high_flood_level_m": 296.50,
                "safe_discharge_cumec": 18000.0,
                "catchment_elevation_m": 298.0,
                "catchment_slope_deg": 1.4,
                "distance_to_river_km": 0.2,
                "latitude": 22.7500,
                "longitude": 77.7200,
                "upstream_station": "Mandla / Bargi Dam",
                "upstream_distance_km": 280.0,
                "dam_reservoir": "Tawa Dam & Bargi Dam",
                "historical_flood_count_5yr": 3,
                "description": "Major CWC hydrological observatory; vulnerable to simultaneous Bargi and Tawa dam releases."
            }
        },
        "Mandla": {
            "Mandla Bridge": {
                "river": "Narmada",
                "basin": "Narmada Basin",
                "warning_level_m": 442.00,
                "danger_level_m": 443.50,
                "high_flood_level_m": 445.00,
                "safe_discharge_cumec": 12000.0,
                "catchment_elevation_m": 447.0,
                "catchment_slope_deg": 2.2,
                "distance_to_river_km": 0.3,
                "latitude": 22.6000,
                "longitude": 80.3700,
                "upstream_station": "Amarkantak Source",
                "upstream_distance_km": 150.0,
                "dam_reservoir": "Bargi Dam upstream impact",
                "historical_flood_count_5yr": 2,
                "description": "Upper Narmada river valley station surrounded by dense forest catchment."
            }
        },
        "Sheopur": {
            "Sheopur Chambal": {
                "river": "Chambal",
                "basin": "Ganga Basin (Yamuna sub-basin)",
                "warning_level_m": 196.00,
                "danger_level_m": 198.00,
                "high_flood_level_m": 201.20,
                "safe_discharge_cumec": 15000.0,
                "catchment_elevation_m": 200.0,
                "catchment_slope_deg": 1.6,
                "distance_to_river_km": 0.7,
                "latitude": 25.6600,
                "longitude": 76.7000,
                "upstream_station": "Kota Barrage",
                "upstream_distance_km": 95.0,
                "dam_reservoir": "Gandhi Sagar & Rana Pratap Sagar",
                "historical_flood_count_5yr": 3,
                "description": "Lower Chambal valley vulnerable to ravines overflow and Kota Barrage releases."
            }
        },
        "Raisen": {
            "Ganjbasoda": {
                "river": "Betwa",
                "basin": "Ganga Basin (Yamuna sub-basin)",
                "warning_level_m": 410.00,
                "danger_level_m": 412.00,
                "high_flood_level_m": 414.50,
                "safe_discharge_cumec": 8000.0,
                "catchment_elevation_m": 415.0,
                "catchment_slope_deg": 1.3,
                "distance_to_river_km": 0.5,
                "latitude": 23.8500,
                "longitude": 77.9300,
                "upstream_station": "Bhopal Halali Dam",
                "upstream_distance_km": 70.0,
                "dam_reservoir": "Rajghat & Matatila Dam Downstream",
                "historical_flood_count_5yr": 2,
                "description": "Central MP plateau river system draining Vidisha and Raisen agrarian catchments."
            }
        }
    },

    # ==================== BIHAR ====================
    "Bihar": {
        "Patna": {
            "Digha Ghat": {
                "river": "Ganga",
                "basin": "Ganga Basin",
                "warning_level_m": 49.50,
                "danger_level_m": 50.45,
                "high_flood_level_m": 52.52,
                "safe_discharge_cumec": 65000.0,
                "catchment_elevation_m": 53.0,
                "catchment_slope_deg": 0.3,
                "distance_to_river_km": 0.2,
                "latitude": 25.6500,
                "longitude": 85.1000,
                "upstream_station": "Buxar",
                "upstream_distance_km": 115.0,
                "dam_reservoir": "Tehri / Haridwar regulation",
                "historical_flood_count_5yr": 4,
                "description": "Crucial CWC station in Patna capital city; monitors Ganga combined with Gandak and Sone inflows."
            }
        },
        "Supaul": {
            "Birpur Barrage": {
                "river": "Kosi",
                "basin": "Ganga Basin (Kosi sub-basin)",
                "warning_level_m": 74.00,
                "danger_level_m": 75.20,
                "high_flood_level_m": 76.80,
                "safe_discharge_cumec": 18000.0,
                "catchment_elevation_m": 77.0,
                "catchment_slope_deg": 1.1,
                "distance_to_river_km": 0.3,
                "latitude": 26.5200,
                "longitude": 87.0100,
                "upstream_station": "Barahkshetra (Nepal)",
                "upstream_distance_km": 45.0,
                "dam_reservoir": "Kosi Barrage",
                "historical_flood_count_5yr": 5,
                "description": "The 'Sorrow of Bihar' entry point from Nepal Himalayas, carrying immense silt loads."
            }
        },
        "Gopalganj": {
            "Dumriaghat": {
                "river": "Gandak",
                "basin": "Ganga Basin",
                "warning_level_m": 61.20,
                "danger_level_m": 62.24,
                "high_flood_level_m": 64.00,
                "safe_discharge_cumec": 14000.0,
                "catchment_elevation_m": 64.0,
                "catchment_slope_deg": 0.6,
                "distance_to_river_km": 0.4,
                "latitude": 26.3300,
                "longitude": 84.8500,
                "upstream_station": "Valmikinagar Barrage",
                "upstream_distance_km": 125.0,
                "dam_reservoir": "Valmikinagar Barrage",
                "historical_flood_count_5yr": 4,
                "description": "Major Gandak river corridor feeding north Bihar agricultural floodplains."
            }
        },
        "Muzaffarpur": {
            "Benibad": {
                "river": "Bagmati",
                "basin": "Ganga Basin (Bagmati sub-basin)",
                "warning_level_m": 47.68,
                "danger_level_m": 48.68,
                "high_flood_level_m": 50.10,
                "safe_discharge_cumec": 4500.0,
                "catchment_elevation_m": 50.0,
                "catchment_slope_deg": 0.4,
                "distance_to_river_km": 0.3,
                "latitude": 26.1200,
                "longitude": 85.5800,
                "upstream_station": "Dheng Bridge (Sitamarhi)",
                "upstream_distance_km": 65.0,
                "dam_reservoir": "None",
                "historical_flood_count_5yr": 5,
                "description": "Perennially flood-prone tributary in north Bihar with fast water surges from Nepal Terai."
            }
        },
        "Khagaria": {
            "Baltara": {
                "river": "Kosi",
                "basin": "Ganga Basin",
                "warning_level_m": 33.85,
                "danger_level_m": 34.85,
                "high_flood_level_m": 36.40,
                "safe_discharge_cumec": 20000.0,
                "catchment_elevation_m": 36.0,
                "catchment_slope_deg": 0.2,
                "distance_to_river_km": 0.2,
                "latitude": 25.5500,
                "longitude": 86.6700,
                "upstream_station": "Birpur Barrage",
                "upstream_distance_km": 140.0,
                "dam_reservoir": "None",
                "historical_flood_count_5yr": 5,
                "description": "Downstream confluence area where Kosi meets Ganga, creating backwater inundations."
            }
        }
    }
}


def get_all_states():
    """Return list of supported states."""
    return list(STATIONS_DATA.keys())


def get_districts(state):
    """Return list of districts for a given state."""
    return list(STATIONS_DATA.get(state, {}).keys())


def get_stations(state, district):
    """Return list of stations for a given state and district."""
    return list(STATIONS_DATA.get(state, {}).get(district, {}).keys())


def get_station_profile(state, district, station):
    """Return the full metadata profile dictionary for a station."""
    profile = STATIONS_DATA.get(state, {}).get(district, {}).get(station, None)
    if profile is None:
        return None
    # Add hierarchy tags
    p = dict(profile)
    p["state"] = state
    p["district"] = district
    p["station"] = station
    return p


def get_all_stations_flat():
    """Return a flat list of all station profiles with state and district tags."""
    flat_list = []
    for state, districts in STATIONS_DATA.items():
        for district, stations in districts.items():
            for station, profile in stations.items():
                p = dict(profile)
                p["state"] = state
                p["district"] = district
                p["station"] = station
                flat_list.append(p)
    return flat_list

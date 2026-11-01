"""
Standard Operational Procedures (SOP) and Actionable Flood Early-Warning Advisories
Aligned with NDMA (National Disaster Management Authority) and State SDMA guidelines.
"""

RISK_LEVEL_CONFIG = {
    "Low": {
        "min_prob": 0.0,
        "max_prob": 25.0,
        "color": "#28a745",
        "badge": "LOW RISK",
        "severity": "NORMAL",
        "advisory": (
            "Hydrological and meteorological conditions are within safe buffer limits. "
            "Maintain standard baseflow monitoring, gauge inspections, and automated telemetry logging."
        ),
        "actions": [
            "Maintain routine 24-hour river stage observation log.",
            "Verify real-time telemetry battery and solar panel operational status.",
            "Ensure riverbank drainage sluice gates are clear of vegetative debris.",
            "Normal agricultural and river navigation activities may proceed."
        ]
    },
    "Moderate": {
        "min_prob": 25.0,
        "max_prob": 55.0,
        "color": "#ffc107",
        "badge": "MODERATE RISK",
        "severity": "ALERT",
        "advisory": (
            "River stage is rising or upstream catchments have recorded substantial precipitation. "
            "Local flood management authorities and block-level response teams should be placed on standby."
        ),
        "actions": [
            "Increase gauge monitoring frequency to 6-hour intervals.",
            "Alert Block Development Officers (BDO) and local Revenue Officers.",
            "Inspect vulnerable embankment reaches, revetments, and dykes for piping or seepage.",
            "Issue advisory to riverine farmers (Diara / Chhar tracts) to secure livestock and equipment.",
            "Coordinate reservoir release schedules with upstream barrage engineers."
        ]
    },
    "High": {
        "min_prob": 55.0,
        "max_prob": 80.0,
        "color": "#fd7e14",
        "badge": "HIGH RISK",
        "severity": "WARNING",
        "advisory": (
            "Substantial probability of river stage breaching the Danger Level within 24 hours. "
            "Potential inundation of low-lying settlements and agricultural floodplains. Emergency mobilization recommended."
        ),
        "actions": [
            "Activate District Emergency Operations Centre (DEOC) on 24x7 rotation.",
            "Issue public warning bulletins across local village panchayats and FM radio.",
            "Mobilize State Disaster Response Force (SDRF) boat units and quick-response teams.",
            "Stage sandbags, geo-textile bags, and earthmovers at vulnerable embankment spurs.",
            "Identify designated high-ground relief shelters and ready drinking water provisions.",
            "Enact speed restrictions on submersible bridges and causeways."
        ]
    },
    "Severe": {
        "min_prob": 80.0,
        "max_prob": 100.0,
        "color": "#dc3545",
        "badge": "SEVERE RISK",
        "severity": "EMERGENCY",
        "advisory": (
            "CRITICAL FLOOD IMMINENT: River stage forecast to exceed High Flood Level (HFL) or cause major embankment breaches. "
            "Immediate life-safety evacuation protocols must be triggered under the Disaster Management Act."
        ),
        "actions": [
            "Execute mandatory evacuation of floodway and floodplain settlements within 500m of channel.",
            "Sound community flood sirens and send bulk emergency cell-broadcast alerts.",
            "Deploy National Disaster Response Force (NDRF) battalions for water rescue operations.",
            "Suspend traffic on vulnerable bridges and inundated national/state highways.",
            "Cut power feeder lines in submerged habitations to prevent electrocution hazards.",
            "Open relief camps and deploy medical epidemic prevention teams."
        ]
    }
}


def get_risk_assessment(probability_pct):
    """
    Map a calibrated probability (0-100%) to risk classification and SOP advisory.
    """
    prob = max(0.0, min(100.0, float(probability_pct)))
    
    if prob < 25.0:
        cat = "Low"
    elif prob < 55.0:
        cat = "Moderate"
    elif prob < 80.0:
        cat = "High"
    else:
        cat = "Severe"
        
    info = RISK_LEVEL_CONFIG[cat]
    return {
        "probability_pct": round(prob, 1),
        "risk_category": cat,
        "badge": info["badge"],
        "color": info["color"],
        "severity": info["severity"],
        "advisory": info["advisory"],
        "actions": info["actions"]
    }


OFFICIAL_PORTALS = [
    {
        "agency": "Central Water Commission (CWC)",
        "name": "CWC Flood Forecast & Advisory Portal",
        "url": "https://ffs.india-water.gov.in/",
        "scope": "National real-time hydrological stage, warning & danger levels, 7-day hydrograph forecasts.",
        "type": "Hydrology / River Gauges"
    },
    {
        "agency": "India-WRIS",
        "name": "Water Resources Information System of India",
        "url": "https://indiawris.gov.in/wris/",
        "scope": "Comprehensive historical river discharge, reservoir capacities, rainfall time series.",
        "type": "Historical / Spatial Data"
    },
    {
        "agency": "India Meteorological Department (IMD)",
        "name": "Mausam Flood Meteorology & Rainfall Portal",
        "url": "https://mausam.imd.gov.in/",
        "scope": "District quantitative precipitation forecasts (QPF), radar imagery, monsoon alerts.",
        "type": "Meteorology / Weather Forecast"
    },
    {
        "agency": "National Disaster Management Authority (NDMA)",
        "name": "NDMA Disaster Management Guidelines",
        "url": "https://ndma.gov.in/",
        "scope": "National flood management policy, standard operating protocols, emergency helplines.",
        "type": "Disaster Response / Policy"
    },
    {
        "agency": "Punjab SDMA",
        "name": "Punjab Disaster Management Authority",
        "url": "https://revenue.punjab.gov.in/",
        "scope": "Sutlej, Beas, Ghaggar flood advisories and district emergency relief control rooms.",
        "type": "State Response"
    },
    {
        "agency": "Haryana SDMA",
        "name": "Haryana Revenue & Disaster Management",
        "url": "https://revenueharyana.gov.in/",
        "scope": "Yamuna, Markanda, Tangri flood monitoring and Hathnikund discharge advisories.",
        "type": "State Response"
    },
    {
        "agency": "Madhya Pradesh SDMA",
        "name": "MP State Disaster Management Authority",
        "url": "https://mpsdma.nic.in/",
        "scope": "Narmada, Chambal, Betwa river flood control and reservoir gate regulation.",
        "type": "State Response"
    },
    {
        "agency": "Bihar SDMA / Water Resources Dept (WRD)",
        "name": "Bihar Disaster Management & Flood Management Information System (FMIS)",
        "url": "http://fmis.bih.nic.in/",
        "scope": "Kosi, Gandak, Bagmati, Ganga flood telemetry, embankment vulnerability GIS.",
        "type": "State Response"
    }
]

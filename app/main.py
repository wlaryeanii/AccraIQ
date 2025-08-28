# app/main.py
# AccraIQ: Municipal-Grade Transit Optimization Dashboard - Complete Version

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import folium
from streamlit_folium import st_folium
import zipfile
import os
import warnings
from sklearn.cluster import HDBSCAN
import pyproj
import json
import time
from datetime import datetime
import heapq
import math

warnings.filterwarnings("ignore")

# Core libraries
from dtw import dtw
import pulp as pl

# ---------- Page & Global Styles ----------
st.set_page_config(
    page_title="AccraIQ: Municipal Transit Optimization",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "AccraIQ — DTW + HDBSCAN + Set-Cover to reduce redundant routes while preserving stop coverage."}
)

st.markdown("""
<style>
:root {
  --bg: #ffffff;
  --fg: #121212;
  --muted: #5f6368;
  --primary: #1e3c72;     /* deep blue */
  --accent:  #2a5298;     /* mid blue */
  --ok: #2e7d32;          /* green */
  --warn: #ffb300;        /* amber */
}
[data-theme="dark"] :root {
  --bg: #1f1f1f;
  --fg: #f1f3f4;
  --muted: #a7adb2;
  --primary: #6aa2ff;
  --accent:  #2a5298;
  --ok: #7bdc8a;
  --warn: #ffd65c;
}

/* Header */
.header-style {
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color: #fff;
  padding: 1.1rem 1.25rem;
  border-radius: 14px;
  margin: 0 0 1rem 0;
  box-shadow: 0 6px 18px rgba(0,0,0,.12);
}
.header-style h2 { margin: 0 0 .25rem 0; }
.header-style p  { margin: 0; opacity: .95; }

/* KPI grid */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; }
.kpi {
  background: var(--bg);
  color: var(--fg);
  border-radius: 14px;
  padding: .9rem 1rem;
  border: 1px solid rgba(0,0,0,.06);
  box-shadow: 0 2px 10px rgba(0,0,0,.06);
  text-align: center;
}
.kpi h4 { margin: 0; font-weight: 600; font-size: .95rem; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
.kpi .value { font-size: 1.6rem; line-height: 1.2; margin-top: .25rem; font-weight: 700; }

/* Section title */
.section-title { font-size: 1.1rem; font-weight: 700; margin: .5rem 0 .4rem 0; }

/* Compact sidebar spacing */
section[data-testid="stSidebar"] .stRadio, 
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stToggle {
  margin-bottom: .5rem;
}

.streamlit-expanderHeader { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

def header():
    st.markdown("""
    <div class="header-style">
      <h2>🚌 AccraIQ — Municipal Transit Optimization</h2>
      <p>DTW (shape) + HDBSCAN (families) + Set-Cover (representatives) → fewer routes, same coverage.</p>
    </div>
    """, unsafe_allow_html=True)

# ---- Clear, numeric presets (readable labels) ----
PRESETS = {
    "⚡ Fast demo — 90% coverage, cap 200 routes": {
        "coverage_threshold": 0.90,  # keep at least 90% of stops
        "target_density": 5,         # lighter DTW signature (faster)
        "sample_cap": 200,           # analyze up to 200 routes
        "blurb": "Great for demos. Faster run, still representative."
    },
    "🏛 Municipal default — 92% coverage (all routes)": {
        "coverage_threshold": 0.92,
        "target_density": 8,
        "sample_cap": None,          # all routes
        "blurb": "Balanced. Matches recommended settings for planning."
    },
    "🛡 Coverage-first — 96% coverage (all routes)": {
        "coverage_threshold": 0.96,
        "target_density": 8,
        "sample_cap": None,
        "blurb": "Prioritize coverage. Keeps more routes to protect access."
    },
}

# =============================================================================
# CORE ALGORITHM FUNCTIONS
# =============================================================================

@st.cache_data
def load_real_gtfs_data(uploaded_file=None):
    """Load real Accra GTFS data"""
    
    if uploaded_file is not None:
        gtfs_path = uploaded_file
        print(f"📊 Loading GTFS data from uploaded file...")
    else:
        data_paths = [
            "data/gtfs-accra-ghana-2016.zip",
            "./data/gtfs-accra-ghana-2016.zip",
            "gtfs-accra-ghana-2016.zip",
            "../data/gtfs-accra-ghana-2016.zip",
        ]

        gtfs_path = None
        for path in data_paths:
            if os.path.exists(path):
                gtfs_path = path
                print(f"📊 Loading GTFS data from: {path}")
                break

        if gtfs_path is None:
            st.warning("No GTFS data found. Please ensure gtfs-accra-ghana-2016.zip exists in the data/ directory.")
            return None

    try:
        if uploaded_file is not None:
            zip_file = zipfile.ZipFile(uploaded_file, "r")
        else:
            zip_file = zipfile.ZipFile(gtfs_path, "r")

        with zip_file:
            file_list = zip_file.namelist()
            print(f"📁 Found files: {file_list}")

            data = {}
            gtfs_files = [
                "agency.txt", "calendar.txt", "routes.txt", "stop_times.txt", 
                "stops.txt", "trips.txt", "shapes.txt", "fare_attributes.txt", "fare_rules.txt"
            ]

            for file_name in gtfs_files:
                if file_name in file_list:
                    with zip_file.open(file_name) as file:
                        df = pd.read_csv(file)
                        data[file_name.replace(".txt", "")] = df
                else:
                    data[file_name.replace(".txt", "")] = pd.DataFrame()

            return data

    except Exception as e:
        st.error(f"Error loading GTFS file: {e}")
        return None

def classify_route_type(coords, length_km, straightness_ratio):
    """Classify route type based on geometric characteristics"""
    if length_km < 5:
        return "short_local"
    elif length_km > 25:
        return "long_distance" if straightness_ratio > 0.6 else "meandering_long"
    elif straightness_ratio > 0.8:
        return "direct_connection"
    elif straightness_ratio < 0.3:
        return "circular_complex"
    else:
        return "standard_route"

@st.cache_data
def extract_all_route_geometries(gtfs_data):
    """Extract geometries for ALL routes"""
    
    if gtfs_data["shapes"].empty:
        raise ValueError("No shapes data available")

    route_shape_map = {}
    route_info_map = {}

    if not gtfs_data["trips"].empty and not gtfs_data["routes"].empty:
        for _, route in gtfs_data["routes"].iterrows():
            route_info_map[route["route_id"]] = {
                "route_short_name": route.get("route_short_name", route["route_id"]),
                "route_long_name": route.get("route_long_name", f"Route {route['route_id']}"),
                "route_type": route.get("route_type", 3),
            }

        for _, trip in gtfs_data["trips"].iterrows():
            if "shape_id" in trip and pd.notna(trip["shape_id"]):
                route_shape_map[trip["route_id"]] = trip["shape_id"]

    route_geometries = []
    unique_shapes = gtfs_data["shapes"]["shape_id"].unique()

    processed = 0
    for shape_id in unique_shapes:
        shape_points = gtfs_data["shapes"][gtfs_data["shapes"]["shape_id"] == shape_id].sort_values("shape_pt_sequence")

        if len(shape_points) >= 2:
            coords = [(row["shape_pt_lon"], row["shape_pt_lat"]) for _, row in shape_points.iterrows()]

            try:
                line = LineString(coords)

                route_id = None
                route_info = {}

                for r_id, s_id in route_shape_map.items():
                    if s_id == shape_id:
                        route_id = r_id
                        route_info = route_info_map.get(r_id, {})
                        break

                if route_id is None:
                    route_id = f"route_{shape_id}"
                    route_info = {
                        "route_short_name": route_id,
                        "route_long_name": f"Route {route_id}",
                        "route_type": 3,
                    }

                length_km = line.length * 111
                start_point = Point(coords[0])
                end_point = Point(coords[-1])
                direct_distance = start_point.distance(end_point) * 111
                straightness_ratio = direct_distance / length_km if length_km > 0 else 0
                route_type_geo = classify_route_type(coords, length_km, straightness_ratio)

                route_geometries.append({
                    "route_id": route_id,
                    "shape_id": shape_id,
                    "route_short_name": route_info.get("route_short_name", route_id),
                    "route_long_name": route_info.get("route_long_name", f"Route {route_id}"),
                    "geometry": line,
                    "length_km": length_km,
                    "num_points": len(shape_points),
                    "straightness_ratio": straightness_ratio,
                    "point_density_per_km": len(shape_points) / length_km if length_km > 0 else 0,
                    "start_lat": coords[0][1],
                    "start_lon": coords[0][0],
                    "end_lat": coords[-1][1],
                    "end_lon": coords[-1][0],
                    "centroid_lat": line.centroid.y,
                    "centroid_lon": line.centroid.x,
                    "route_type_geo": route_type_geo,
                })

                processed += 1

            except Exception as e:
                continue

    return gpd.GeoDataFrame(route_geometries, crs="EPSG:4326")

class CoordinateNormalizer:
    """Normalize coordinates for consistent distance calculations"""

    def __init__(self, method="utm"):
        self.method = method
        self.transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)

    def normalize(self, coords):
        """Convert coordinates to normalized form"""
        if self.method == "utm":
            normalized = []
            for lon, lat in coords:
                x, y = self.transformer.transform(lon, lat)
                normalized.append([x, y])
            return np.array(normalized)
        else:
            return np.array(coords)

class ImprovedDTWAnalyzer:
    """Improved DTW analysis"""

    def __init__(self, route_gdf, normalizer=None):
        self.route_gdf = route_gdf
        self.normalizer = normalizer or CoordinateNormalizer("utm")
        self.route_signatures = {}
        self.valid_route_indices = []

    def extract_route_signature(self, route_geometry, target_points=30):
        """Extract comprehensive route signature"""

        if route_geometry.geom_type != "LineString":
            return None

        coords = list(route_geometry.coords)
        if len(coords) < 2:
            return None

        try:
            normalized_coords = self.normalizer.normalize(coords)
            line = LineString(normalized_coords)

            distances = np.linspace(0, line.length, target_points)
            interpolated_points = []

            for dist in distances:
                point = line.interpolate(dist)
                interpolated_points.append([point.x, point.y])

            interpolated_points = np.array(interpolated_points)

            bearings = []
            for i in range(len(interpolated_points) - 1):
                dx = interpolated_points[i + 1, 0] - interpolated_points[i, 0]
                dy = interpolated_points[i + 1, 1] - interpolated_points[i, 1]
                bearing = np.arctan2(dy, dx)
                bearings.append(bearing)

            bearings = np.array(bearings)

            bearing_hist, _ = np.histogram(bearings, bins=12, range=(-np.pi, np.pi))
            bearing_hist = bearing_hist / np.sum(bearing_hist) if np.sum(bearing_hist) > 0 else bearing_hist

            route_signature = {
                "coordinates": interpolated_points,
                "bearing_histogram": bearing_hist,
                "length_m": line.length,
                "start_bearing": bearings[0] if len(bearings) > 0 else 0,
                "end_bearing": bearings[-1] if len(bearings) > 0 else 0,
                "bearing_variance": np.var(bearings) if len(bearings) > 0 else 0,
                "start_point": interpolated_points[0],
                "end_point": interpolated_points[-1],
                "centroid": interpolated_points.mean(axis=0),
            }

            return route_signature

        except Exception as e:
            return None

    def calculate_improved_distance(self, sig1, sig2, weights=(0.5, 0.3, 0.2)):
        """Calculate improved composite distance"""

        w_dtw, w_bearing, w_spatial = weights

        try:
            alignment = dtw(sig1["coordinates"], sig2["coordinates"], distance_only=True)
            scale_factor = np.sqrt(max(sig1["length_m"], sig2["length_m"]) / 1000)
            dtw_dist = alignment.distance / (len(sig1["coordinates"]) * scale_factor)
        except:
            dtw_dist = np.linalg.norm(sig1["centroid"] - sig2["centroid"]) / 1000

        hist1, hist2 = sig1["bearing_histogram"], sig2["bearing_histogram"]
        dot_product = np.dot(hist1, hist2)
        norm1, norm2 = np.linalg.norm(hist1), np.linalg.norm(hist2)

        if norm1 > 0 and norm2 > 0:
            bearing_similarity = dot_product / (norm1 * norm2)
            bearing_dist = 1 - bearing_similarity
        else:
            bearing_dist = 1.0

        length_ratio = min(sig1["length_m"], sig2["length_m"]) / max(sig1["length_m"], sig2["length_m"])
        length_dist = 1 - length_ratio

        start_dist = np.linalg.norm(sig1["start_point"] - sig2["start_point"]) / 1000
        end_dist = np.linalg.norm(sig1["end_point"] - sig2["end_point"]) / 1000
        endpoint_dist = (start_dist + end_dist) / 2
        endpoint_dist = min(endpoint_dist, 10)

        spatial_dist = 0.5 * length_dist + 0.5 * (endpoint_dist / 10)

        composite_dist = w_dtw * dtw_dist + w_bearing * bearing_dist + w_spatial * spatial_dist

        return composite_dist

    def compute_improved_similarity_matrix(self, target_density=8, sample_size=None):
        """Compute improved similarity matrix with rescaling"""

        n_routes = len(self.route_gdf) if sample_size is None else min(sample_size, len(self.route_gdf))

        signatures = {}
        valid_routes = []

        route_sample = self.route_gdf.head(n_routes) if sample_size else self.route_gdf

        for idx, (row_idx, route) in enumerate(route_sample.iterrows()):
            sig = self.extract_route_signature(route["geometry"], target_points=target_density * 4)
            if sig is not None:
                signatures[row_idx] = sig
                valid_routes.append(row_idx)

        n_valid = len(valid_routes)
        distance_matrix = np.zeros((n_valid, n_valid))

        for i in range(n_valid):
            for j in range(i, n_valid):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    idx1, idx2 = valid_routes[i], valid_routes[j]
                    dist = self.calculate_improved_distance(signatures[idx1], signatures[idx2])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist

        distance_values_for_scaling = distance_matrix[distance_matrix > 0]

        if len(distance_values_for_scaling) > 0:
            scale_factor = np.median(distance_values_for_scaling)
            distance_matrix_scaled = distance_matrix / scale_factor
        else:
            distance_matrix_scaled = distance_matrix

        self.route_signatures = signatures
        self.valid_route_indices = valid_routes

        return distance_matrix_scaled

def density_based_clustering_leaf_method(distance_matrix, route_indices, min_cluster_size=5, min_samples=5):
    """Fixed clustering with HDBSCAN leaf method"""

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        cluster_selection_method="leaf",
        allow_single_cluster=False,
    )

    labels = clusterer.fit_predict(distance_matrix)

    n_clusters = len(set(labels[labels >= 0]))
    n_noise = np.sum(labels == -1)

    final_labels = labels.copy()
    noise_indices = np.where(labels == -1)[0]

    if len(noise_indices) > 5:
        noise_sub_matrix = distance_matrix[np.ix_(noise_indices, noise_indices)]

        noise_clusterer = HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric="precomputed",
            cluster_selection_method="leaf",
            allow_single_cluster=False,
        )

        try:
            noise_labels = noise_clusterer.fit_predict(noise_sub_matrix)

            n_micro_clusters = len(set(noise_labels[noise_labels >= 0]))
            remaining_noise = np.sum(noise_labels == -1)

            next_label = n_clusters
            for i, noise_idx in enumerate(noise_indices):
                if noise_labels[i] >= 0:
                    final_labels[noise_idx] = next_label + noise_labels[i]
                else:
                    final_labels[noise_idx] = next_label + n_micro_clusters + (remaining_noise - np.sum(noise_labels[i:] == -1))

            total_real_clusters = n_clusters + n_micro_clusters
            total_singletons = remaining_noise

        except Exception as e:
            next_label = n_clusters
            for noise_idx in noise_indices:
                final_labels[noise_idx] = next_label
                next_label += 1
            total_real_clusters = n_clusters
            total_singletons = len(noise_indices)
    else:
        next_label = n_clusters
        for noise_idx in noise_indices:
            final_labels[noise_idx] = next_label
            next_label += 1
        total_real_clusters = n_clusters
        total_singletons = len(noise_indices)

    return final_labels

def pulp_set_cover_optimization(final_labels, route_indices, route_gdf, gtfs_data, coverage_threshold=0.92):
    """PuLP-based set cover optimization"""

    if gtfs_data["stop_times"].empty or gtfs_data["trips"].empty:
        return [], 0.0, 0

    route_to_stops = {}
    trips_df = gtfs_data["trips"]
    stop_times_df = gtfs_data["stop_times"]
    merged = pd.merge(trips_df[["route_id", "trip_id"]], stop_times_df[["trip_id", "stop_id"]], on="trip_id")

    for rid, group in merged.groupby("route_id"):
        route_to_stops[rid] = set(group["stop_id"].unique())

    all_stops = set(stop_times_df["stop_id"].unique())

    clusters = {}
    for route_idx, cluster_label in enumerate(final_labels):
        if route_idx < len(route_indices):
            route_id = route_gdf.iloc[route_indices[route_idx]]["route_id"]
            clusters.setdefault(cluster_label, []).append(route_id)

    all_candidate_routes = []
    routes_covering = {stop: [] for stop in all_stops}

    for cluster_routes in clusters.values():
        all_candidate_routes.extend(cluster_routes)

    for route_id in all_candidate_routes:
        if route_id in route_to_stops:
            for stop in route_to_stops[route_id]:
                routes_covering[stop].append(route_id)

    coverable_stops = [stop for stop in all_stops if len(routes_covering[stop]) > 0]
    target_stops_count = int(len(coverable_stops) * coverage_threshold)

    x = {route: pl.LpVariable(f"x_{route}", 0, 1, pl.LpBinary) for route in all_candidate_routes}

    prob = pl.LpProblem("MinimalSetCover", pl.LpMinimize)
    prob += pl.lpSum(x.values())

    stop_coverage_vars = {}

    for stop in coverable_stops:
        stop_coverage_vars[stop] = pl.LpVariable(f"covered_{stop}", 0, 1, pl.LpBinary)

        if len(routes_covering[stop]) > 0:
            prob += stop_coverage_vars[stop] <= pl.lpSum(x[route] for route in routes_covering[stop])

    prob += pl.lpSum(stop_coverage_vars.values()) >= target_stops_count

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    if prob.status == pl.LpStatusOptimal:
        selected_routes = [route for route, var in x.items() if var.value() == 1]

        covered_stops = set()
        for route in selected_routes:
            covered_stops.update(route_to_stops.get(route, set()))

        coverage_percent = len(covered_stops) / len(all_stops) * 100

        return selected_routes, coverage_percent, len(selected_routes)

    else:
        raise RuntimeError("PuLP optimization failed to find optimal solution")

class MunicipalImpactCalculator:
    """Calculate comprehensive municipal impact"""

    def __init__(self, route_gdf):
        self.route_gdf = route_gdf

        self.fuel_efficiency_l_per_km = 0.12
        self.co2_kg_per_liter = 2.68
        self.diesel_price_ghs_per_liter = 14.5
        self.vehicle_cost_ghc = 150000
        self.daily_trips_per_route = 8
        self.maintenance_cost_per_km_ghc = 0.60
        self.driver_salary_per_route_ghc = 3000

        self.routes_per_vehicle = 3
        self.vehicle_utilization_rate = 0.75

    def calculate_comprehensive_impact(self, final_labels, valid_indices, selected_routes, coverage_pct):
        """Calculate full municipal impact"""

        total_routes = len(valid_indices)
        retained_routes = len(selected_routes)
        eliminated_routes = total_routes - retained_routes

        total_original_length = self.route_gdf.loc[valid_indices]["length_km"].sum()
        selected_route_gdf = self.route_gdf[self.route_gdf["route_id"].isin(selected_routes)]
        retained_length = selected_route_gdf["length_km"].sum() if not selected_route_gdf.empty else 0
        eliminated_length_km = total_original_length - retained_length

        daily_route_km_saved = eliminated_length_km * self.daily_trips_per_route
        daily_fuel_saved_l = daily_route_km_saved * self.fuel_efficiency_l_per_km
        annual_fuel_saved_l = daily_fuel_saved_l * 365
        annual_fuel_cost_saved_ghc = annual_fuel_saved_l * self.diesel_price_ghs_per_liter

        annual_maintenance_saved_ghc = eliminated_length_km * 365 * self.daily_trips_per_route * self.maintenance_cost_per_km_ghc
        annual_driver_savings_ghc = eliminated_routes * self.driver_salary_per_route_ghc
        total_annual_savings_ghc = annual_fuel_cost_saved_ghc + annual_maintenance_saved_ghc + annual_driver_savings_ghc

        original_vehicles_needed = total_routes / self.routes_per_vehicle
        retained_vehicles_needed = retained_routes / self.routes_per_vehicle
        vehicles_saved = max(0, original_vehicles_needed - retained_vehicles_needed)

        vehicle_capital_freed_ghc = vehicles_saved * self.vehicle_cost_ghc * self.vehicle_utilization_rate

        annual_co2_saved_tonnes = (annual_fuel_saved_l * self.co2_kg_per_liter) / 1000
        cars_equivalent_removed = annual_co2_saved_tonnes / 4.6
        trees_equivalent_planted = annual_co2_saved_tonnes * 16

        cluster_analysis = []
        unique_clusters = np.unique(final_labels)
        valid_route_data = self.route_gdf.loc[valid_indices].copy()
        valid_route_data["cluster"] = final_labels

        for cluster_id in unique_clusters:
            cluster_routes = valid_route_data[valid_route_data["cluster"] == cluster_id]
            representatives = [r for r in selected_routes if r in cluster_routes["route_id"].values]

            cluster_analysis.append({
                "cluster_id": cluster_id,
                "n_routes": len(cluster_routes),
                "representatives_selected": len(representatives),
                "cluster_length_km": cluster_routes["length_km"].sum(),
                "avg_route_length": cluster_routes["length_km"].mean(),
                "reduction_in_cluster": 1 - (len(representatives) / len(cluster_routes)) if len(cluster_routes) > 0 else 0,
            })

        return {
            "total_routes": total_routes,
            "retained_routes": retained_routes,
            "eliminated_routes": eliminated_routes,
            "route_reduction_pct": (eliminated_routes / total_routes) * 100,
            "total_length_km": total_original_length,
            "retained_length_km": retained_length,
            "eliminated_length_km": eliminated_length_km,
            "length_reduction_pct": (eliminated_length_km / total_original_length) * 100,
            "annual_fuel_saved_l": annual_fuel_saved_l,
            "annual_co2_saved_tonnes": annual_co2_saved_tonnes,
            "annual_fuel_cost_saved_ghc": annual_fuel_cost_saved_ghc,
            "annual_maintenance_saved_ghc": annual_maintenance_saved_ghc,
            "annual_driver_savings_ghc": annual_driver_savings_ghc,
            "total_annual_savings_ghc": total_annual_savings_ghc,
            "vehicle_capital_freed_ghc": vehicle_capital_freed_ghc,
            "fleet_vehicles_saved": vehicles_saved,
            "fleet_utilization_improvement": (retained_routes / max(1, retained_vehicles_needed)) / self.routes_per_vehicle,
            "coverage_retention_pct": coverage_pct,
            "selected_routes": selected_routes,
            "cars_equivalent_removed": cars_equivalent_removed,
            "trees_equivalent_planted": trees_equivalent_planted,
            "cluster_analysis": cluster_analysis,
            "total_clusters": len(unique_clusters),
        }

# ---------- ROUTE CHECKER HELPERS (no extra deps) ----------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

@st.cache_data
def build_stop_graph(gtfs_data):
    """Build undirected stop graph from GTFS stop_times; edge weight = distance (km).
       Also returns stops dataframe (id, name, lat, lon) and per-edge route_ids."""
    if gtfs_data is None or gtfs_data["stops"].empty or gtfs_data["stop_times"].empty or gtfs_data["trips"].empty:
        return None, None, None

    stops = gtfs_data["stops"][["stop_id","stop_name","stop_lat","stop_lon"]].dropna().copy()
    stop_pos = stops.set_index("stop_id")[["stop_lat","stop_lon"]].to_dict("index")

    # trip_id -> route_id
    trip_to_route = gtfs_data["trips"][["trip_id","route_id"]].set_index("trip_id")["route_id"].to_dict()

    # Build adjacency and edge->routes map
    adj = {}                     # stop_id -> list[(nbr_id, weight_km)]
    edge_routes = {}             # frozenset({u,v}) -> set(route_ids)

    for trip_id, g in gtfs_data["stop_times"].sort_values(["trip_id","stop_sequence"]).groupby("trip_id"):
        rid = trip_to_route.get(trip_id)
        seq = g[["stop_id","stop_sequence"]].values
        for i in range(1, len(seq)):
            u = str(seq[i-1][0]); v = str(seq[i][0])
            if u not in stop_pos or v not in stop_pos:
                continue
            lat1, lon1 = stop_pos[u]["stop_lat"], stop_pos[u]["stop_lon"]
            lat2, lon2 = stop_pos[v]["stop_lat"], stop_pos[v]["stop_lon"]
            w = max(haversine_km(lat1, lon1, lat2, lon2), 1e-4)  # avoid zero-weight

            # undirected insert with minimum weight
            adj.setdefault(u, [])
            adj.setdefault(v, [])
            # store; allow parallel routes via edge_routes
            key = frozenset({u, v})
            edge_routes.setdefault(key, set())
            if rid is not None:
                edge_routes[key].add(str(rid))

            # keep smallest observed weight per neighbor
            def upsert(a, b, w):
                for i,(nb, ww) in enumerate(a):
                    if nb == b:
                        if w < ww: a[i] = (b, w)
                        return
                a.append((b, w))
            upsert(adj[u], v, w)
            upsert(adj[v], u, w)

    return adj, stops, edge_routes

def dijkstra_path(adj, start, goal):
    """Plain Dijkstra on the adjacency from build_stop_graph; returns (path_stop_ids, total_km)."""
    if start not in adj or goal not in adj:
        return None, float("inf")
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen: 
            continue
        seen.add(u)
        if u == goal:
            break
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if goal not in dist:
        return None, float("inf")
    # reconstruct
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist[goal]

def estimate_fare(path, stops_df, edge_routes, gtfs_data, fallback_rate_per_km=0.9):
    """Estimate fare: if GTFS fares exist, count boardings by contiguous route segments; else km * rate."""
    if not path or len(path) < 2:
        return 0.0, 0
    # distance
    pos = stops_df.set_index("stop_id")[["stop_lat","stop_lon"]].to_dict("index")
    total_km = sum(haversine_km(pos[path[i]]["stop_lat"], pos[path[i]]["stop_lon"],
                                pos[path[i+1]]["stop_lat"], pos[path[i+1]]["stop_lon"])
                   for i in range(len(path)-1))

    fares = gtfs_data.get("fare_attributes")
    rules = gtfs_data.get("fare_rules")
    trips = gtfs_data.get("trips")
    route_to_fare = {}
    if fares is not None and not fares.empty and rules is not None and not rules.empty:
        # map route_id -> min fare price found
        route_to_fare = (rules[["route_id","fare_id"]]
            .merge(fares[["fare_id","price"]], on="fare_id", how="left")
            .dropna(subset=["route_id","price"])
            .groupby("route_id")["price"].min().to_dict())

    if route_to_fare:
        # infer contiguous route segments along the path using edge_routes
        def edge_key(a,b): return frozenset({a,b})
        route_seq = []
        for i in range(len(path)-1):
            candidates = list(edge_routes.get(edge_key(path[i], path[i+1]), []))
            route_seq.append(candidates[0] if candidates else None)

        # compress consecutive duplicates, ignore None
        boardings = []
        last = None
        for r in route_seq:
            if r is None: 
                continue
            if r != last:
                boardings.append(r)
                last = r
        fare_est = float(sum(route_to_fare.get(r, 0.0) for r in boardings))
        return fare_est, len(boardings)-1 if boardings else 0
    else:
        # fallback per-km
        return float(total_km * fallback_rate_per_km), 0

# =============================================================================
# STREAMLINED UI FUNCTIONS
# =============================================================================

def create_simple_map(route_geometries, color, label, selected_routes=None):
    """Create a simple map for visualization"""
    bounds = route_geometries.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    if selected_routes:
        selected_gdf = route_geometries[route_geometries["route_id"].isin(selected_routes)]
        routes_to_show = selected_gdf
    else:
        if len(route_geometries) > 100:
            routes_to_show = route_geometries.sample(100, random_state=42)
        else:
            routes_to_show = route_geometries

    for idx, route in routes_to_show.iterrows():
        if route["geometry"].geom_type == "LineString":
            coords = [(lat, lon) for lon, lat in route["geometry"].coords]
            folium.PolyLine(
                coords,
                color=color,
                weight=3,
                opacity=0.7,
                popup=f"{label}: {route['route_short_name']} ({route['length_km']:.1f} km)",
            ).add_to(m)

    return m

def setup_sidebar():
    """Sidebar: data source + explicit preset with numbers and intent."""
    with st.sidebar:
        st.header("⚙️ Dataset")
        data_mode = st.radio("Source", ["Sample (Accra 2016)", "Upload GTFS"], index=0)
        uploaded_file = st.file_uploader("GTFS .zip", type=["zip"]) if data_mode == "Upload GTFS" else None

        st.divider()
        st.header("🎯 Optimization goal")

        # present readable labels directly from PRESETS
        labels = list(PRESETS.keys())
        default_ix = labels.index("🏛 Municipal default — 92% coverage (all routes)") if "🏛 Municipal default — 92% coverage (all routes)" in labels else 0
        preset_label = st.selectbox("Preset", options=labels, index=default_ix)

        # show exact parameters so it's obvious what will run
        cfg = PRESETS[preset_label]
        cap_text = "all routes" if cfg["sample_cap"] in (None, 0) else f"cap {cfg['sample_cap']} routes"
        st.caption(
            f"**Preset details:** "
            f"Coverage ≥ **{int(cfg['coverage_threshold']*100)}%**, "
            f"DTW signature density **{cfg['target_density']}×4 pts**, "
            f"{cap_text}. "
            f"{cfg.get('blurb','')}"
        )

        # optional tiny override for folks who want control—off by default
        with st.expander("Advanced (optional)", expanded=False):
            if st.toggle("Tweak this preset"):
                cov = st.slider("Minimum stop coverage (%)", 85, 98, int(cfg["coverage_threshold"]*100), step=1)
                dens = st.slider("Route signature density (×4 points)", 4, 12, cfg["target_density"])
                cap = st.number_input("Max routes to analyze (0 = all)", min_value=0, value=0 if cfg["sample_cap"] in (None, 0) else int(cfg["sample_cap"]), step=50)
                cfg = {
                    "coverage_threshold": cov/100.0,
                    "target_density": int(dens),
                    "sample_cap": None if cap == 0 else int(cap)
                }
                st.caption("Advanced override active.")

        return uploaded_file, preset_label, cfg

def run_pipeline(cfg, uploaded_file):
    """Streamlined pipeline using explicit cfg dict (from preset or override)."""
    gtfs_data = load_real_gtfs_data(uploaded_file)
    if gtfs_data is None:
        st.error("No GTFS data available. Upload a zip or ensure sample file exists.")
        st.stop()

    route_geometries = extract_all_route_geometries(gtfs_data)
    if route_geometries.empty:
        st.error("No route geometries could be extracted.")
        st.stop()

    analyzer = ImprovedDTWAnalyzer(route_geometries, normalizer=CoordinateNormalizer("utm"))
    dm = analyzer.compute_improved_similarity_matrix(
        target_density=cfg["target_density"],
        sample_size=cfg["sample_cap"]
    )

    # Auto min_cluster_size from neighbor density
    pos = dm[dm > 0]
    thr = np.nanmedian(pos) if pos.size else 0
    neighbor_counts = (dm < thr).sum(axis=1) if thr > 0 else np.full(dm.shape[0], 6)
    min_cluster_size = int(np.clip(np.percentile(neighbor_counts, 30), 5, 12))

    final_labels = density_based_clustering_leaf_method(
        dm, analyzer.valid_route_indices,
        min_cluster_size=min_cluster_size,
        min_samples=min_cluster_size
    )

    selected_routes, coverage_percent, _ = pulp_set_cover_optimization(
        final_labels, analyzer.valid_route_indices, route_geometries, gtfs_data, cfg["coverage_threshold"]
    )

    impact = MunicipalImpactCalculator(route_geometries).calculate_comprehensive_impact(
        final_labels, analyzer.valid_route_indices, selected_routes, coverage_percent
    )

    st.session_state.update({
        "gtfs_data": gtfs_data,
        "route_geometries": route_geometries,
        "distance_matrix": dm,
        "final_labels": final_labels,
        "selected_routes": selected_routes,
        "impact": impact,
        "valid_indices": analyzer.valid_route_indices,
        "cfg": cfg,
    })

def render_route_checker_tab(uploaded_file):
    """Public-facing shortest path + fare estimator using GTFS stops."""
    # Ensure GTFS loaded (works even if user didn't run the optimizer)
    gtfs = st.session_state.get("gtfs_data") or load_real_gtfs_data(uploaded_file)
    if gtfs is None or gtfs["stops"].empty:
        st.info("Load the sample or upload a GTFS zip on the left.")
        return

    adj, stops_df, edge_routes = build_stop_graph(gtfs)
    if adj is None:
        st.info("GTFS missing stop_times/trips; cannot build the route graph.")
        return

    st.markdown('<div class="section-title">Route & Fare Checker</div>', unsafe_allow_html=True)

    # Simple search lists
    stops_df = stops_df.copy()
    stops_df["label"] = stops_df["stop_name"].astype(str) + " • " + stops_df["stop_id"].astype(str)

    c1, c2, c3 = st.columns([1.2, 1.2, 0.6])
    with c1:
        src_label = st.selectbox("From", options=stops_df["label"].sort_values().tolist(), index=0)
    with c2:
        dst_label = st.selectbox("To", options=stops_df["label"].sort_values().tolist(), index=min(1, len(stops_df)-1))
    with c3:
        fallback_rate = st.number_input("Fallback fare (₵/km)", min_value=0.0, value=0.9, step=0.1, help="Used only if GTFS has no fare tables")

    # Extract IDs
    src_id = src_label.split(" • ")[-1]
    dst_id = dst_label.split(" • ")[-1]

    go = st.button("🧭 Find route", use_container_width=True)
    if go:
        st.session_state["route_query"] = {
            "src_id": src_id,
            "dst_id": dst_id,
            "fallback_rate": float(fallback_rate),
        }

    # If no prior click, wait for user action
    if "route_query" not in st.session_state:
        return

    # Use last confirmed inputs so the results persist across reruns
    src_id = st.session_state["route_query"]["src_id"]
    dst_id = st.session_state["route_query"]["dst_id"]
    fallback_rate = st.session_state["route_query"]["fallback_rate"]

    path, km = dijkstra_path(adj, src_id, dst_id)
    if not path:
        st.warning("No path found between those stops.")
        return

    fare_est, transfers = estimate_fare(path, stops_df, edge_routes, gtfs, fallback_rate_per_km=fallback_rate)

    # KPIs
    c = st.columns(4)
    with c[0]: st.metric("Shortest distance", f"{km:.1f} km")
    with c[1]: st.metric("Transfers", f"{max(transfers,0)}")
    with c[2]: st.metric("Estimated fare", f"₵{fare_est:.2f}")
    with c[3]: st.metric("Stops on path", f"{len(path)}")

    # Map
    pos = stops_df.set_index("stop_id")[["stop_lat","stop_lon"]].to_dict("index")
    m = folium.Map(location=[pos[path[0]]["stop_lat"], pos[path[0]]["stop_lon"]], zoom_start=12)
    coords = [(pos[s]["stop_lat"], pos[s]["stop_lon"]) for s in path]
    folium.PolyLine(coords, color="#d32f2f", weight=5, opacity=0.85).add_to(m)
    folium.CircleMarker(location=coords[0], radius=5, color="#1a73e8", fill=True, popup="Start").add_to(m)
    folium.CircleMarker(location=coords[-1], radius=5, color="#2e7d32", fill=True, popup="End").add_to(m)
    st_folium(m, height=460, use_container_width=True)

    # Steps list
    names = stops_df.set_index("stop_id")["stop_name"].to_dict()
    st.write("**Stops along the way**")
    st.write(" → ".join([names.get(s, s) for s in path]))

def kpi_card(title, value):
    """Render a KPI card"""
    st.markdown(f"""
    <div class="kpi">
      <h4>{title}</h4>
      <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def render_results_tabs():
    """Render results in streamlined tabs"""
    impact           = st.session_state["impact"]
    route_geometries = st.session_state["route_geometries"]
    selected_routes  = st.session_state["selected_routes"]
    final_labels     = st.session_state["final_labels"]
    valid_indices    = st.session_state["valid_indices"]

    # KPIs
    st.markdown('<div class="section-title">Key outcomes</div>', unsafe_allow_html=True)
    c = st.columns(4)
    with c[0]: kpi_card("Route reduction", f"{impact['route_reduction_pct']:.1f}%")
    with c[1]: kpi_card("Coverage kept", f"{impact['coverage_retention_pct']:.1f}%")
    with c[2]: kpi_card("Annual savings", f"₵{impact['total_annual_savings_ghc']:,.0f}")
    with c[3]: kpi_card("CO₂ saved", f"{impact['annual_co2_saved_tonnes']:.0f} t/yr")

    # Only three tabs
    t1, t2, t3 = st.tabs(["Map", "Clusters", "Exports"])

    # Before/After map
    with t1:
        show_after = st.toggle("Show optimized network", value=True)
        m = create_simple_map(
            route_geometries,
            "red" if show_after else "blue",
            "Selected" if show_after else "Original",
            selected_routes if show_after else None
        )
        st_folium(m, height=480, use_container_width=True)

    # Clusters summary
    with t2:
        st.markdown("**Cluster summary** (representatives = routes we kept)")
        labels = pd.Series(final_labels, name="cluster")
        rows = pd.DataFrame({"row_idx": valid_indices}).reset_index(drop=True)
        tmp = pd.concat([rows, labels], axis=1)
        tmp["route_id"] = tmp["row_idx"].apply(lambda i: route_geometries.iloc[i]["route_id"])
        tmp["is_representative"] = tmp["route_id"].isin(selected_routes)
        agg = tmp.groupby("cluster").agg(
            routes_in_cluster=("route_id","count"),
            representatives=("is_representative","sum")
        ).reset_index().sort_values("routes_in_cluster", ascending=False)
        st.dataframe(agg, use_container_width=True, height=360)

    # Exports (CSV + JSON only)
    with t3:
        col1, col2 = st.columns(2)

        with col1:
            selected_df = route_geometries[route_geometries["route_id"].isin(selected_routes)]
            st.download_button(
                "📄 Selected routes (CSV)",
                data=selected_df.drop("geometry", axis=1).to_csv(index=False),
                file_name="accra_optimized_routes.csv",
                mime="text/csv"
            )

        with col2:
            summary = {
                "route_reduction_pct": impact["route_reduction_pct"],
                "coverage_retention_pct": impact["coverage_retention_pct"],
                "annual_savings_ghc": impact["total_annual_savings_ghc"],
                "annual_co2_saved_tonnes": impact["annual_co2_saved_tonnes"],
                "selected_routes": selected_routes
            }
            st.download_button(
                "📊 Summary (JSON)",
                data=json.dumps(summary, indent=2),
                file_name="accra_optimization_summary.json",
                mime="application/json"
            )

def main():
    """Main application"""
    header()
    uploaded_file, preset_label, cfg = setup_sidebar()

    st.markdown(
        f"<div class='section-title'>Selected preset</div>"
        f"<div style='font-size:.95rem;color:var(--muted);'>{preset_label}</div>",
        unsafe_allow_html=True
    )

    t_admin, t_public = st.tabs(["🏛️ Optimization (Admin)", "🧭 Route & Fare Checker"])

    with t_admin:
        run = st.button("🚀 Optimize network", type="primary", use_container_width=True)
        if run:
            t0 = time.time()
            with st.spinner("Optimizing..."):
                run_pipeline(cfg, uploaded_file)
            st.success(f"Done in {(time.time()-t0):.1f}s")
        if "impact" in st.session_state:
            render_results_tabs()
        else:
            st.info("Load the sample or upload GTFS, pick a preset, then **Optimize network**.")

    with t_public:
        render_route_checker_tab(uploaded_file)

if __name__ == "__main__":
    main()

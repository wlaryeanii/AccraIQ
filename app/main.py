# app/main.py
# AccraIQ: Municipal-Grade Transit Optimization Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import folium
from streamlit_folium import st_folium
import plotly.express as px
import folium.plugins
import zipfile
import os
import warnings
from sklearn.cluster import HDBSCAN
import pyproj
import json
import time
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from datetime import datetime
from io import BytesIO
import base64
import tempfile

# PDF Generation imports
try:
    import weasyprint
    from jinja2 import Template

    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    st.warning("WeasyPrint not available. Install with: pip install weasyprint jinja2")

# Map screenshot imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    st.warning("Selenium not available. Install with: pip install selenium")

warnings.filterwarnings("ignore")

# Core libraries - matching notebook
from dtw import dtw
import pulp as pl

# Configure Streamlit
st.set_page_config(
    page_title="AccraIQ: Municipal Transit Optimization",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS - Simplified
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-card.success {
        border-left: 5px solid #28a745;
    }
    
    .metric-card.warning {
        border-left: 5px solid #ffc107;
    }
    
    .metric-card.info {
        border-left: 5px solid #17a2b8;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .section-header {
        color: #1e3c72;
        border-bottom: 2px solid #2a5298;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    .run-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        border: none;
    }
    
    .results-summary {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        border-left: 5px solid #2a5298;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# EXACT NOTEBOOK IMPLEMENTATIONS
# =============================================================================


@st.cache_data
def load_real_gtfs_data(uploaded_file=None):
    """Load real Accra GTFS data - exact notebook implementation"""

    if uploaded_file is not None:
        # Handle uploaded file
        gtfs_path = uploaded_file
        print(f"📊 Loading GTFS data from uploaded file...")
    else:
        # Always try data directory first, then fallback paths
        data_paths = [
            "data/gtfs-accra-ghana-2016.zip",  # Primary data directory
            "./data/gtfs-accra-ghana-2016.zip",  # Current directory data
            "gtfs-accra-ghana-2016.zip",  # Root directory
            "../data/gtfs-accra-ghana-2016.zip",  # Parent data directory
        ]

        gtfs_path = None
        for path in data_paths:
            if os.path.exists(path):
                gtfs_path = path
                print(f"📊 Loading GTFS data from: {path}")
                break

        if gtfs_path is None:
            st.warning(
                "No GTFS data found. Please ensure gtfs-accra-ghana-2016.zip exists in the data/ directory."
            )
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
                "agency.txt",
                "calendar.txt",
                "routes.txt",
                "stop_times.txt",
                "stops.txt",
                "trips.txt",
                "shapes.txt",
                "fare_attributes.txt",
                "fare_rules.txt",
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
    """Classify route type based on geometric characteristics - exact notebook implementation"""

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
    """Extract geometries for ALL routes - exact notebook implementation"""

    if gtfs_data["shapes"].empty:
        raise ValueError("No shapes data available")

    # Get route-to-shape mapping
    route_shape_map = {}
    route_info_map = {}

    if not gtfs_data["trips"].empty and not gtfs_data["routes"].empty:
        for _, route in gtfs_data["routes"].iterrows():
            route_info_map[route["route_id"]] = {
                "route_short_name": route.get("route_short_name", route["route_id"]),
                "route_long_name": route.get(
                    "route_long_name", f"Route {route['route_id']}"
                ),
                "route_type": route.get("route_type", 3),
            }

        for _, trip in gtfs_data["trips"].iterrows():
            if "shape_id" in trip and pd.notna(trip["shape_id"]):
                route_shape_map[trip["route_id"]] = trip["shape_id"]

    route_geometries = []
    unique_shapes = gtfs_data["shapes"]["shape_id"].unique()

    processed = 0
    for shape_id in unique_shapes:
        shape_points = gtfs_data["shapes"][
            gtfs_data["shapes"]["shape_id"] == shape_id
        ].sort_values("shape_pt_sequence")

        if len(shape_points) >= 2:
            coords = [
                (row["shape_pt_lon"], row["shape_pt_lat"])
                for _, row in shape_points.iterrows()
            ]

            try:
                line = LineString(coords)

                # Find corresponding route
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

                # Calculate metrics
                length_km = line.length * 111
                start_point = Point(coords[0])
                end_point = Point(coords[-1])
                direct_distance = start_point.distance(end_point) * 111
                straightness_ratio = direct_distance / length_km if length_km > 0 else 0
                route_type_geo = classify_route_type(
                    coords, length_km, straightness_ratio
                )

                route_geometries.append(
                    {
                        "route_id": route_id,
                        "shape_id": shape_id,
                        "route_short_name": route_info.get(
                            "route_short_name", route_id
                        ),
                        "route_long_name": route_info.get(
                            "route_long_name", f"Route {route_id}"
                        ),
                        "geometry": line,
                        "length_km": length_km,
                        "num_points": len(shape_points),
                        "straightness_ratio": straightness_ratio,
                        "point_density_per_km": len(shape_points) / length_km
                        if length_km > 0
                        else 0,
                        "start_lat": coords[0][1],
                        "start_lon": coords[0][0],
                        "end_lat": coords[-1][1],
                        "end_lon": coords[-1][0],
                        "centroid_lat": line.centroid.y,
                        "centroid_lon": line.centroid.x,
                        "route_type_geo": route_type_geo,
                    }
                )

                processed += 1

            except Exception as e:
                continue

    return gpd.GeoDataFrame(route_geometries, crs="EPSG:4326")


class CoordinateNormalizer:
    """Normalize coordinates for consistent distance calculations - exact notebook implementation"""

    def __init__(self, method="utm"):
        self.method = method
        self.transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32630",
            always_xy=True,  # UTM Zone 30N for Accra
        )

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
    """Improved DTW analysis - exact notebook implementation"""

    def __init__(self, route_gdf, normalizer=None):
        self.route_gdf = route_gdf
        self.normalizer = normalizer or CoordinateNormalizer("utm")
        self.route_signatures = {}
        self.valid_route_indices = []

    def extract_route_signature(self, route_geometry, target_points=30):
        """Extract comprehensive route signature - exact notebook implementation"""

        if route_geometry.geom_type != "LineString":
            return None

        coords = list(route_geometry.coords)
        if len(coords) < 2:
            return None

        try:
            normalized_coords = self.normalizer.normalize(coords)
            line = LineString(normalized_coords)

            # Interpolate to fixed number of points
            distances = np.linspace(0, line.length, target_points)
            interpolated_points = []

            for dist in distances:
                point = line.interpolate(dist)
                interpolated_points.append([point.x, point.y])

            interpolated_points = np.array(interpolated_points)

            # Calculate bearing sequence
            bearings = []
            for i in range(len(interpolated_points) - 1):
                dx = interpolated_points[i + 1, 0] - interpolated_points[i, 0]
                dy = interpolated_points[i + 1, 1] - interpolated_points[i, 1]
                bearing = np.arctan2(dy, dx)
                bearings.append(bearing)

            bearings = np.array(bearings)

            # Create bearing histogram
            bearing_hist, _ = np.histogram(bearings, bins=12, range=(-np.pi, np.pi))
            bearing_hist = (
                bearing_hist / np.sum(bearing_hist)
                if np.sum(bearing_hist) > 0
                else bearing_hist
            )

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
        """Calculate improved composite distance - exact notebook implementation"""

        w_dtw, w_bearing, w_spatial = weights

        # Enhanced DTW distance
        try:
            alignment = dtw(
                sig1["coordinates"], sig2["coordinates"], distance_only=True
            )
            scale_factor = np.sqrt(max(sig1["length_m"], sig2["length_m"]) / 1000)
            dtw_dist = alignment.distance / (len(sig1["coordinates"]) * scale_factor)
        except:
            dtw_dist = np.linalg.norm(sig1["centroid"] - sig2["centroid"]) / 1000

        # Bearing similarity
        hist1, hist2 = sig1["bearing_histogram"], sig2["bearing_histogram"]
        dot_product = np.dot(hist1, hist2)
        norm1, norm2 = np.linalg.norm(hist1), np.linalg.norm(hist2)

        if norm1 > 0 and norm2 > 0:
            bearing_similarity = dot_product / (norm1 * norm2)
            bearing_dist = 1 - bearing_similarity
        else:
            bearing_dist = 1.0

        # Spatial features
        length_ratio = min(sig1["length_m"], sig2["length_m"]) / max(
            sig1["length_m"], sig2["length_m"]
        )
        length_dist = 1 - length_ratio

        start_dist = np.linalg.norm(sig1["start_point"] - sig2["start_point"]) / 1000
        end_dist = np.linalg.norm(sig1["end_point"] - sig2["end_point"]) / 1000
        endpoint_dist = (start_dist + end_dist) / 2
        endpoint_dist = min(endpoint_dist, 10)

        spatial_dist = 0.5 * length_dist + 0.5 * (endpoint_dist / 10)

        # Combine distances
        composite_dist = (
            w_dtw * dtw_dist + w_bearing * bearing_dist + w_spatial * spatial_dist
        )

        return composite_dist

    def compute_improved_similarity_matrix(self, target_density=8, sample_size=None):
        """Compute improved similarity matrix with rescaling - exact notebook implementation"""

        n_routes = (
            len(self.route_gdf)
            if sample_size is None
            else min(sample_size, len(self.route_gdf))
        )

        # Extract route signatures
        signatures = {}
        valid_routes = []

        route_sample = self.route_gdf.head(n_routes) if sample_size else self.route_gdf

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (row_idx, route) in enumerate(route_sample.iterrows()):
            status_text.text(f"Extracting route signatures: {idx + 1}/{n_routes}")
            progress_bar.progress((idx + 1) / n_routes)

            sig = self.extract_route_signature(
                route["geometry"], target_points=target_density * 4
            )
            if sig is not None:
                signatures[row_idx] = sig
                valid_routes.append(row_idx)

        # Compute distance matrix
        n_valid = len(valid_routes)
        distance_matrix = np.zeros((n_valid, n_valid))

        status_text.text("Computing distance matrix...")
        progress_bar.progress(0)

        for i in range(n_valid):
            progress_bar.progress(i / n_valid)
            for j in range(i, n_valid):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    idx1, idx2 = valid_routes[i], valid_routes[j]
                    dist = self.calculate_improved_distance(
                        signatures[idx1], signatures[idx2]
                    )
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist

        # Rescale by median
        distance_values_for_scaling = distance_matrix[distance_matrix > 0]

        if len(distance_values_for_scaling) > 0:
            scale_factor = np.median(distance_values_for_scaling)
            distance_matrix_scaled = distance_matrix / scale_factor
        else:
            distance_matrix_scaled = distance_matrix

        self.route_signatures = signatures
        self.valid_route_indices = valid_routes

        progress_bar.empty()
        status_text.empty()

        return distance_matrix_scaled


def density_based_clustering_leaf_method(
    distance_matrix, route_indices, min_cluster_size=5, min_samples=5
):
    """Fixed clustering with HDBSCAN leaf method - exact notebook implementation"""

    # HDBSCAN with leaf method
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

    # Two-pass clustering for noise refinement
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
                    final_labels[noise_idx] = (
                        next_label
                        + n_micro_clusters
                        + (remaining_noise - np.sum(noise_labels[i:] == -1))
                    )

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


def pulp_set_cover_optimization(
    final_labels, route_indices, route_gdf, gtfs_data, coverage_threshold=0.92
):
    """PuLP-based set cover optimization - exact notebook implementation"""

    # Build route-to-stops mapping
    if gtfs_data["stop_times"].empty or gtfs_data["trips"].empty:
        return [], 0.0, 0

    route_to_stops = {}
    trips_df = gtfs_data["trips"]
    stop_times_df = gtfs_data["stop_times"]
    merged = pd.merge(
        trips_df[["route_id", "trip_id"]],
        stop_times_df[["trip_id", "stop_id"]],
        on="trip_id",
    )

    for rid, group in merged.groupby("route_id"):
        route_to_stops[rid] = set(group["stop_id"].unique())

    all_stops = set(stop_times_df["stop_id"].unique())

    # Build cluster mapping
    clusters = {}
    for route_idx, cluster_label in enumerate(final_labels):
        if route_idx < len(route_indices):
            route_id = route_gdf.iloc[route_indices[route_idx]]["route_id"]
            clusters.setdefault(cluster_label, []).append(route_id)

    # Get candidate routes
    all_candidate_routes = []
    routes_covering = {stop: [] for stop in all_stops}

    for cluster_routes in clusters.values():
        all_candidate_routes.extend(cluster_routes)

    # Build coverage mapping
    for route_id in all_candidate_routes:
        if route_id in route_to_stops:
            for stop in route_to_stops[route_id]:
                routes_covering[stop].append(route_id)

    # Filter coverable stops
    coverable_stops = [stop for stop in all_stops if len(routes_covering[stop]) > 0]
    target_stops_count = int(len(coverable_stops) * coverage_threshold)

    # Set up PuLP problem
    x = {
        route: pl.LpVariable(f"x_{route}", 0, 1, pl.LpBinary)
        for route in all_candidate_routes
    }

    prob = pl.LpProblem("MinimalSetCover", pl.LpMinimize)
    prob += pl.lpSum(x.values())

    stop_coverage_vars = {}

    for stop in coverable_stops:
        stop_coverage_vars[stop] = pl.LpVariable(f"covered_{stop}", 0, 1, pl.LpBinary)

        if len(routes_covering[stop]) > 0:
            prob += stop_coverage_vars[stop] <= pl.lpSum(
                x[route] for route in routes_covering[stop]
            )

    prob += pl.lpSum(stop_coverage_vars.values()) >= target_stops_count

    # Solve
    prob.solve(pl.PULP_CBC_CMD(msg=False))

    if prob.status == pl.LpStatusOptimal:
        selected_routes = [route for route, var in x.items() if var.value() == 1]

        # Calculate coverage
        covered_stops = set()
        for route in selected_routes:
            covered_stops.update(route_to_stops.get(route, set()))

        coverage_percent = len(covered_stops) / len(all_stops) * 100

        return selected_routes, coverage_percent, len(selected_routes)

    else:
        raise RuntimeError("PuLP optimization failed to find optimal solution")


class MunicipalImpactCalculator:
    """Calculate comprehensive municipal impact - exact notebook implementation"""

    def __init__(self, route_gdf):
        self.route_gdf = route_gdf

        # Realistic parameters for Ghana
        self.fuel_efficiency_l_per_km = 0.12  # 12 L/100 km
        self.co2_kg_per_liter = 2.68
        self.diesel_price_ghs_per_liter = 14.5  # Current average
        self.vehicle_cost_ghc = 150000  # Realistic used trotro cost
        self.daily_trips_per_route = 8
        self.maintenance_cost_per_km_ghc = 0.60  # Adjusted for local costs
        self.driver_salary_per_route_ghc = 3000  # GHS 250/month

        # Fleet optimization parameters
        self.routes_per_vehicle = 3  # One vehicle can serve ~3 routes efficiently
        self.vehicle_utilization_rate = 0.75  # 75% utilization (realistic for trotros)

    def calculate_comprehensive_impact(
        self, final_labels, valid_indices, selected_routes, coverage_pct
    ):
        """Calculate full municipal impact - exact notebook implementation"""

        total_routes = len(valid_indices)
        retained_routes = len(selected_routes)
        eliminated_routes = total_routes - retained_routes

        # Length analysis
        total_original_length = self.route_gdf.loc[valid_indices]["length_km"].sum()
        selected_route_gdf = self.route_gdf[
            self.route_gdf["route_id"].isin(selected_routes)
        ]
        retained_length = (
            selected_route_gdf["length_km"].sum() if not selected_route_gdf.empty else 0
        )
        eliminated_length_km = total_original_length - retained_length

        # Economic calculations
        daily_route_km_saved = eliminated_length_km * self.daily_trips_per_route
        daily_fuel_saved_l = daily_route_km_saved * self.fuel_efficiency_l_per_km
        annual_fuel_saved_l = daily_fuel_saved_l * 365
        annual_fuel_cost_saved_ghc = (
            annual_fuel_saved_l * self.diesel_price_ghs_per_liter
        )

        annual_maintenance_saved_ghc = (
            eliminated_length_km
            * 365
            * self.daily_trips_per_route
            * self.maintenance_cost_per_km_ghc
        )
        annual_driver_savings_ghc = eliminated_routes * self.driver_salary_per_route_ghc
        total_annual_savings_ghc = (
            annual_fuel_cost_saved_ghc
            + annual_maintenance_saved_ghc
            + annual_driver_savings_ghc
        )

        # Fleet optimization - realistic vehicle capital savings
        # Calculate how many fewer vehicles are needed
        original_vehicles_needed = total_routes / self.routes_per_vehicle
        retained_vehicles_needed = retained_routes / self.routes_per_vehicle
        vehicles_saved = max(0, original_vehicles_needed - retained_vehicles_needed)

        # Apply utilization rate to get actual capital freed
        vehicle_capital_freed_ghc = (
            vehicles_saved * self.vehicle_cost_ghc * self.vehicle_utilization_rate
        )

        # Environmental impact
        annual_co2_saved_tonnes = (annual_fuel_saved_l * self.co2_kg_per_liter) / 1000
        cars_equivalent_removed = annual_co2_saved_tonnes / 4.6
        trees_equivalent_planted = annual_co2_saved_tonnes * 16

        # Cluster analysis
        cluster_analysis = []
        unique_clusters = np.unique(final_labels)
        valid_route_data = self.route_gdf.loc[valid_indices].copy()
        valid_route_data["cluster"] = final_labels

        for cluster_id in unique_clusters:
            cluster_routes = valid_route_data[valid_route_data["cluster"] == cluster_id]
            representatives = [
                r for r in selected_routes if r in cluster_routes["route_id"].values
            ]

            cluster_analysis.append(
                {
                    "cluster_id": cluster_id,
                    "n_routes": len(cluster_routes),
                    "representatives_selected": len(representatives),
                    "cluster_length_km": cluster_routes["length_km"].sum(),
                    "avg_route_length": cluster_routes["length_km"].mean(),
                    "reduction_in_cluster": 1
                    - (len(representatives) / len(cluster_routes))
                    if len(cluster_routes) > 0
                    else 0,
                }
            )

        return {
            "total_routes": total_routes,
            "retained_routes": retained_routes,
            "eliminated_routes": eliminated_routes,
            "route_reduction_pct": (eliminated_routes / total_routes) * 100,
            "total_length_km": total_original_length,
            "retained_length_km": retained_length,
            "eliminated_length_km": eliminated_length_km,
            "length_reduction_pct": (eliminated_length_km / total_original_length)
            * 100,
            "annual_fuel_saved_l": annual_fuel_saved_l,
            "annual_co2_saved_tonnes": annual_co2_saved_tonnes,
            "annual_fuel_cost_saved_ghc": annual_fuel_cost_saved_ghc,
            "annual_maintenance_saved_ghc": annual_maintenance_saved_ghc,
            "annual_driver_savings_ghc": annual_driver_savings_ghc,
            "total_annual_savings_ghc": total_annual_savings_ghc,
            "vehicle_capital_freed_ghc": vehicle_capital_freed_ghc,
            "fleet_vehicles_saved": vehicles_saved,
            "fleet_utilization_improvement": (
                retained_routes / max(1, retained_vehicles_needed)
            )
            / self.routes_per_vehicle,
            "coverage_retention_pct": coverage_pct,
            "selected_routes": selected_routes,
            "cars_equivalent_removed": cars_equivalent_removed,
            "trees_equivalent_planted": trees_equivalent_planted,
            "cluster_analysis": cluster_analysis,
            "total_clusters": len(unique_clusters),
        }


def validate_municipal_viability(
    impact_summary,
    target_reduction_range=(25, 40),
    min_coverage=90,
    max_cluster_share=40,
):
    """Validate municipal viability - exact notebook implementation"""

    reduction_pct = impact_summary["route_reduction_pct"]
    coverage_pct = impact_summary["coverage_retention_pct"]

    # Calculate cluster statistics
    cluster_analysis = impact_summary["cluster_analysis"]
    total_routes = impact_summary["total_routes"]

    largest_cluster_size = max([c["n_routes"] for c in cluster_analysis])
    largest_cluster_share = (largest_cluster_size / total_routes) * 100

    # Check criteria
    reduction_ok = (
        target_reduction_range[0] <= reduction_pct <= target_reduction_range[1]
    )
    coverage_ok = coverage_pct >= (min_coverage - 1e-9)
    cluster_size_ok = largest_cluster_share <= max_cluster_share

    all_criteria_met = reduction_ok and coverage_ok and cluster_size_ok

    return {
        "viable": all_criteria_met,
        "reduction_ok": reduction_ok,
        "coverage_ok": coverage_ok,
        "cluster_size_ok": cluster_size_ok,
        "reduction_pct": reduction_pct,
        "coverage_pct": coverage_pct,
        "largest_cluster_share": largest_cluster_share,
    }


# =============================================================================
# STOP DENSITY OVERLAY FUNCTIONS
# =============================================================================


@st.cache_data
def extract_stop_data(gtfs_data):
    """Extract and process stop data from GTFS for density analysis"""

    if gtfs_data["stops"].empty:
        return None

    stops_df = gtfs_data["stops"].copy()

    # Ensure we have required columns
    required_cols = ["stop_id", "stop_lat", "stop_lon"]
    if not all(col in stops_df.columns for col in required_cols):
        return None

    # Clean and filter stops
    stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])
    stops_df = stops_df[
        (stops_df["stop_lat"] != 0)
        & (stops_df["stop_lon"] != 0)
        & (stops_df["stop_lat"].between(-90, 90))
        & (stops_df["stop_lon"].between(-180, 180))
    ]

    if len(stops_df) == 0:
        return None

    # Create geometry
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
        crs="EPSG:4326",
    )

    return stops_gdf


def calculate_stop_density(stops_gdf, method="kde", grid_size=0.01, bandwidth=0.005):
    """Calculate stop density using KDE or hexbin method"""

    if stops_gdf is None or len(stops_gdf) == 0:
        return None

    if method == "kde":
        # Kernel Density Estimation
        coords = np.column_stack([stops_gdf.geometry.x, stops_gdf.geometry.y])

        # Create regular grid for density estimation
        x_min, y_min, x_max, y_max = stops_gdf.total_bounds
        x_range = np.arange(x_min, x_max, grid_size)
        y_range = np.arange(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

        # Calculate KDE
        kde = gaussian_kde(coords.T, bw_method=bandwidth)
        density = kde(grid_coords.T).reshape(xx.shape)

        # Create density GeoDataFrame
        density_data = []
        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                if density[i, j] > 0:
                    density_data.append(
                        {
                            "geometry": Point(x, y),
                            "density": density[i, j],
                            "x": x,
                            "y": y,
                        }
                    )

        density_gdf = gpd.GeoDataFrame(density_data, crs="EPSG:4326")

    elif method == "hexbin":
        # Hexagonal binning (requires h3 library)
        try:
            from shapely.geometry import Polygon
            import h3

            # Convert to H3 hexagons
            hex_data = []
            for _, stop in stops_gdf.iterrows():
                h3_index = h3.latlng_to_h3(stop.geometry.y, stop.geometry.x, 8)
                hex_data.append(
                    {
                        "h3_index": h3_index,
                        "geometry": Point(stop.geometry.x, stop.geometry.y),
                    }
                )

            # Count stops per hexagon
            hex_counts = (
                pd.DataFrame(hex_data)
                .groupby("h3_index")
                .size()
                .reset_index(name="count")
            )

            # Create hexagon geometries
            density_data = []
            for _, row in hex_counts.iterrows():
                hex_boundary = h3.h3_to_geo_boundary(row["h3_index"])
                hex_poly = Polygon(hex_boundary)
                density_data.append(
                    {
                        "geometry": hex_poly,
                        "density": row["count"],
                        "h3_index": row["h3_index"],
                    }
                )

            density_gdf = gpd.GeoDataFrame(density_data, crs="EPSG:4326")
        except ImportError:
            st.warning("H3 library not available, falling back to grid method")
            return calculate_stop_density(stops_gdf, method="grid", grid_size=grid_size)

    else:  # method == 'grid'
        # Simple grid-based counting
        x_min, y_min, x_max, y_max = stops_gdf.total_bounds
        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        # Count stops in each grid cell
        density_data = []
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                x_min_cell, x_max_cell = x_bins[i], x_bins[i + 1]
                y_min_cell, y_max_cell = y_bins[j], y_bins[j + 1]

                # Count stops in this cell
                mask = (
                    (stops_gdf.geometry.x >= x_min_cell)
                    & (stops_gdf.geometry.x < x_max_cell)
                    & (stops_gdf.geometry.y >= y_min_cell)
                    & (stops_gdf.geometry.y < y_max_cell)
                )
                count = mask.sum()

                if count > 0:
                    cell_center = Point(
                        (x_min_cell + x_max_cell) / 2, (y_min_cell + y_max_cell) / 2
                    )
                    density_data.append(
                        {
                            "geometry": cell_center,
                            "density": count,
                            "x": (x_min_cell + x_max_cell) / 2,
                            "y": (y_min_cell + y_max_cell) / 2,
                        }
                    )

        density_gdf = gpd.GeoDataFrame(density_data, crs="EPSG:4326")

    return density_gdf


def add_stop_density_to_map(m, density_gdf, opacity=0.6, max_density=None):
    """Add stop density heatmap layer to folium map"""

    if density_gdf is None or len(density_gdf) == 0:
        return m

    # Normalize density values
    if max_density is None:
        max_density = density_gdf["density"].max()

    if max_density == 0:
        return m

    # Create heatmap data
    heatmap_data = []
    for _, row in density_gdf.iterrows():
        if "x" in row and "y" in row:
            lat, lon = row["y"], row["x"]
        else:
            lat, lon = row.geometry.y, row.geometry.x

        # Normalize density to 0-1 range
        normalized_density = min(row["density"] / max_density, 1.0)

        heatmap_data.append([lat, lon, normalized_density])

    if heatmap_data:
        # Add heatmap layer
        folium.plugins.HeatMap(
            heatmap_data,
            radius=15,
            blur=10,
            max_zoom=13,
            opacity=opacity,
            gradient={0.0: "blue", 0.3: "cyan", 0.6: "yellow", 1.0: "red"},
        ).add_to(m)

    return m


def create_stop_density_analysis(gtfs_data):
    """Create comprehensive stop density analysis"""

    # Extract stop data
    stops_gdf = extract_stop_data(gtfs_data)

    if stops_gdf is None:
        return None, None, None

    # Calculate density using different methods
    kde_density = calculate_stop_density(stops_gdf, method="kde")
    grid_density = calculate_stop_density(stops_gdf, method="grid")

    # Basic statistics
    total_stops = len(stops_gdf)
    bounds = stops_gdf.total_bounds
    area_km2 = (
        (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 111 * 111
    )  # Rough conversion
    density_per_km2 = total_stops / area_km2 if area_km2 > 0 else 0

    # Find high-density areas
    if kde_density is not None and len(kde_density) > 0:
        high_density_threshold = kde_density["density"].quantile(0.9)
        high_density_areas = kde_density[
            kde_density["density"] >= high_density_threshold
        ]
    else:
        high_density_areas = None

    return {
        "stops_gdf": stops_gdf,
        "kde_density": kde_density,
        "grid_density": grid_density,
        "total_stops": total_stops,
        "area_km2": area_km2,
        "density_per_km2": density_per_km2,
        "high_density_areas": high_density_areas,
        "bounds": bounds,
    }


# =============================================================================
# PDF REPORT GENERATION
# =============================================================================


def export_plotly_chart_as_base64(fig):
    """Export Plotly chart as base64 encoded image"""
    try:
        import plotly.io as pio

        # Update the figure with proper styling for PDF
        fig.update_layout(
            font=dict(color="black"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                gridcolor="lightgray", linecolor="black", tickfont=dict(color="black")
            ),
            yaxis=dict(
                gridcolor="lightgray", linecolor="black", tickfont=dict(color="black")
            ),
        )

        img_bytes = pio.to_image(fig, format="png", width=800, height=500)
        return base64.b64encode(img_bytes).decode()
    except Exception as e:
        st.warning(f"Chart export failed: {e}")
        return None


def export_folium_map_as_base64(folium_map, width=800, height=600):
    """Export Folium map as base64 encoded image using Selenium"""
    if not SELENIUM_AVAILABLE:
        return None

    try:
        # Save map as HTML
        map_html = folium_map._repr_html_()

        # Use selenium to capture screenshot
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--disable-images")
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument(f"--window-size={width},{height}")

        # Check for Docker environment variables
        chrome_bin = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
        chromedriver_path = os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

        if os.path.exists(chrome_bin):
            options.binary_location = chrome_bin

        # Try different driver initialization methods
        driver = None
        try:
            if os.path.exists(chromedriver_path):
                from selenium.webdriver.chrome.service import Service

                service = Service(executable_path=chromedriver_path)
                driver = webdriver.Chrome(service=service, options=options)
            else:
                # Fallback to default Chrome driver
                driver = webdriver.Chrome(options=options)
        except Exception as e:
            st.warning(f"Chrome driver initialization failed: {e}")
            return None

        if driver is None:
            return None

        try:
            driver.set_window_size(width, height)

            # Create temporary HTML file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False
            ) as f:
                f.write(f"""
                <!DOCTYPE html>
                <html><head><meta charset="utf-8"></head>
                <body style="margin:0;padding:0;">{map_html}</body></html>
                """)
                temp_path = f.name

            driver.get(f"file://{temp_path}")
            time.sleep(5)  # Wait longer for map to load in Docker

            screenshot = driver.get_screenshot_as_png()

            # Clean up temp file
            os.unlink(temp_path)

            return base64.b64encode(screenshot).decode()

        finally:
            driver.quit()

    except Exception as e:
        st.warning(f"Map export failed: {e}")
        return None


def generate_professional_pdf_report(
    impact,
    route_geometries,
    selected_routes,
    gtfs_data,
    viability=None,  # Now optional/unused
    stop_density_analysis=None,
):
    """Generate comprehensive PDF report with maps and charts (no viability logic)"""

    if not WEASYPRINT_AVAILABLE:
        st.error(
            "WeasyPrint not available. Please install: pip install weasyprint jinja2"
        )
        return None

    # Generate charts and maps
    charts_data = {}

    # Economic breakdown chart
    economic_data = pd.DataFrame(
        {
            "Category": [
                "Fuel Savings",
                "Maintenance Savings",
                "Driver Savings",
                "Vehicle Capital",
            ],
            "Value": [
                impact["annual_fuel_cost_saved_ghc"],
                impact["annual_maintenance_saved_ghc"],
                impact["annual_driver_savings_ghc"],
                impact["vehicle_capital_freed_ghc"],
            ],
        }
    )

    economic_fig = px.bar(
        economic_data,
        x="Category",
        y="Value",
        title="Economic Benefits by Category",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    )
    charts_data["economic_chart"] = export_plotly_chart_as_base64(economic_fig)

    # Route type distribution
    type_counts = route_geometries["route_type_geo"].value_counts()
    route_fig = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Route Type Distribution",
        color_discrete_sequence=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
        ],
    )
    charts_data["route_distribution"] = export_plotly_chart_as_base64(route_fig)

    # Create maps
    def create_before_map_for_pdf(_route_geometries):
        bounds = _route_geometries.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        for idx, route in _route_geometries.iterrows():
            if route["geometry"].geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in route["geometry"].coords]
                folium.PolyLine(
                    coords,
                    color="blue",
                    weight=2,
                    opacity=0.7,
                    popup=f"ORIGINAL: {route['route_short_name']} ({route['length_km']:.1f} km)",
                ).add_to(m)
        return m

    def create_after_map_for_pdf(_route_geometries, selected_routes):
        bounds = _route_geometries.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        selected_gdf = _route_geometries[
            _route_geometries["route_id"].isin(selected_routes)
        ]
        for idx, route in selected_gdf.iterrows():
            if route["geometry"].geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in route["geometry"].coords]
                folium.PolyLine(
                    coords,
                    color="red",
                    weight=4,
                    opacity=0.8,
                    popup=f"SELECTED: {route['route_short_name']} ({route['length_km']:.1f} km)",
                ).add_to(m)
        return m

    before_map = create_before_map_for_pdf(route_geometries)
    after_map = create_after_map_for_pdf(route_geometries, selected_routes)

    try:
        charts_data["before_map"] = export_folium_map_as_base64(before_map)
    except Exception as e:
        charts_data["before_map"] = None
    try:
        charts_data["after_map"] = export_folium_map_as_base64(after_map)
    except Exception as e:
        charts_data["after_map"] = None

    # --- Add stop density heatmap ---
    if stop_density_analysis is None:
        stop_density_analysis = create_stop_density_analysis(gtfs_data)
    stop_density_img = None
    if stop_density_analysis and stop_density_analysis.get("kde_density") is not None:
        # Create folium map with heatmap overlay
        kde_density = stop_density_analysis["kde_density"]
        stops_gdf = stop_density_analysis["stops_gdf"]
        bounds = stops_gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        m_density = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        add_stop_density_to_map(m_density, kde_density)
        try:
            stop_density_img = export_folium_map_as_base64(m_density)
        except Exception as e:
            stop_density_img = None
    charts_data["stop_density_map"] = stop_density_img

    # HTML Template for PDF (no viability logic)
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AccraIQ Municipal Transit Optimization Report</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; }
            .header { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 2rem; text-align: center; border-radius: 10px; margin-bottom: 2rem; }
            .section { margin-bottom: 2rem; }
            .metrics-grid { display: flex; gap: 1rem; }
            .metric-card { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; text-align: center; border: 1px solid #e0e0e0; flex: 1; }
            .metric-card h3 { margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
            .metric-card .value { font-size: clamp(1.1rem, 2vw, 1.4rem); font-weight: bold; color: #333; margin: 0; overflow-wrap: break-word; word-break: break-all; }
            .metric-card .subtitle { font-size: 0.8rem; color: #666; margin: 0.25rem 0 0 0; }
            .chart-container { text-align: center; margin: 1rem 0; }
            .chart-container img { max-width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 8px; }
            .map-comparison { display: flex; gap: 1rem; }
            .map-container { text-align: center; flex: 1; }
            .map-container img { width: 100%; max-width: 350px; height: auto; border: 2px solid #e0e0e0; border-radius: 8px; }
            .heatmap-section { margin: 2rem 0; text-align: center; }
            .heatmap-section img { width: 100%; max-width: 500px; height: auto; border: 2px solid #e0e0e0; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚌 AccraIQ Municipal Transit Optimization</h1>
            <p>Comprehensive Analysis Report</p>
            <p>Generated: {{ report_date }}</p>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Route Reduction</h3>
                    <div class="value">{{ safe_format(impact.route_reduction_pct, ".1f") }}%</div>
                    <div class="subtitle">{{ safe_int_format(impact.eliminated_routes) }} of {{ safe_int_format(impact.total_routes) }} routes</div>
                </div>
                <div class="metric-card">
                    <h3>Coverage Retained</h3>
                    <div class="value">{{ safe_format(impact.coverage_retention_pct, ".1f") }}%</div>
                    <div class="subtitle">{{ safe_int_format(impact.retained_length_km) }} km maintained</div>
                </div>
                <div class="metric-card">
                    <h3>Annual CO₂ Saved</h3>
                    <div class="value">{{ safe_int_format(impact.annual_co2_saved_tonnes) }}</div>
                    <div class="subtitle">tonnes per year</div>
                </div>
                <div class="metric-card">
                    <h3>Annual Savings</h3>
                    <div class="value">₵{{ safe_int_format(impact.total_annual_savings_ghc) }}</div>
                    <div class="subtitle">total economic benefit</div>
                </div>
            </div>
        </div>

        <div class="heatmap-section">
            <h2>Transit Stop Density Heatmap</h2>
            {% if charts_data.stop_density_map %}
            <img src="data:image/png;base64,{{ charts_data.stop_density_map }}" alt="Transit Stop Density Heatmap">
            <p><small>High-density areas indicate clusters of transit stops across Accra.</small></p>
            {% else %}
            <p><em>Stop density heatmap unavailable.</em></p>
            {% endif %}
        </div>

        <div class="section">
            <h2>Economic Impact Analysis</h2>
            <ul>
                <li><strong>Fuel Savings:</strong> ₵{{ safe_int_format(impact.annual_fuel_cost_saved_ghc) }}</li>
                <li><strong>Maintenance Savings:</strong> ₵{{ safe_int_format(impact.annual_maintenance_saved_ghc) }}</li>
                <li><strong>Driver Savings:</strong> ₵{{ safe_int_format(impact.annual_driver_savings_ghc) }}</li>
                <li><strong>Total Annual:</strong> ₵{{ safe_int_format(impact.total_annual_savings_ghc) }}</li>
            </ul>
            {% if charts_data.economic_chart %}
            <div class="chart-container">
                <h3>Economic Benefits by Category</h3>
                <img src="data:image/png;base64,{{ charts_data.economic_chart }}" alt="Economic Benefits Chart">
            </div>
            {% endif %}
        </div>

        <div class="section">
            <h2>Network Analysis</h2>
            {% if charts_data.route_distribution %}
            <div class="chart-container">
                <h3>Route Type Distribution</h3>
                <img src="data:image/png;base64,{{ charts_data.route_distribution }}" alt="Route Distribution Chart">
            </div>
            {% endif %}
            <ul>
                <li><strong>Original Network:</strong> {{ impact.total_routes }} routes, {{ safe_format(impact.total_length_km, ".0f") }} km</li>
                <li><strong>Optimized Network:</strong> {{ impact.retained_routes }} routes, {{ safe_format(impact.retained_length_km, ".0f") }} km</li>
                <li><strong>Length Reduction:</strong> {{ safe_format(impact.length_reduction_pct, ".1f") }}%</li>
            </ul>
        </div>

        <div class="section">
            <h2>Network Visualization Comparison</h2>
            <div class="map-comparison">
                <div class="map-container">
                    <h4>📊 Original Network</h4>
                    {% if charts_data.before_map %}
                    <img src="data:image/png;base64,{{ charts_data.before_map }}" alt="Original Network Map">
                    {% endif %}
                    <p><small>All {{ impact.total_routes }} routes ({{ safe_format(impact.total_length_km, ".0f") }} km total)</small></p>
                </div>
                <div class="map-container">
                    <h4>🎯 Optimized Network</h4>
                    {% if charts_data.after_map %}
                    <img src="data:image/png;base64,{{ charts_data.after_map }}" alt="Optimized Network Map">
                    {% endif %}
                    <p><small>{{ impact.retained_routes }} selected routes ({{ safe_format(impact.retained_length_km, ".0f") }} km retained)</small></p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    # Render template with data
    from jinja2 import Template

    template = Template(html_template)

    def safe_format(value, format_str=".1f"):
        try:
            if value is None:
                return "0"
            if format_str.startswith("."):
                return f"{float(value):{format_str}}"
            else:
                return f"{float(value):{format_str}}"
        except (ValueError, TypeError):
            return "0"

    def safe_int_format(value):
        try:
            if value is None:
                return "0"
            return f"{int(float(value)):,}"
        except (ValueError, TypeError):
            return "0"

    template_data = {
        "impact": impact,
        "charts_data": charts_data,
        "report_date": datetime.now().strftime("%B %d, %Y"),
        "safe_format": safe_format,
        "safe_int_format": safe_int_format,
    }

    html_content = template.render(**template_data)

    # Generate PDF
    try:
        pdf_buffer = BytesIO()
        weasyprint.HTML(string=html_content).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")
        return None


# =============================================================================
# STREAMLIT APP INTERFACE - SIMPLIFIED
# =============================================================================


def main():
    # Clean, simple header
    st.markdown(
        """
    <div class="main-header">
        <h1>🚌 AccraIQ: Municipal Transit Optimization</h1>
        <p>Optimize Accra's transit network with advanced algorithms</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Simple sidebar with minimal controls
    with st.sidebar:
        st.header("⚙️ Settings")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload GTFS Data",
            type=["zip"],
            help="Upload gtfs-accra-ghana-2016.zip or compatible GTFS file",
        )

        st.divider()

        # Analysis parameters
        st.subheader("Analysis Parameters")
        min_cluster_size = st.slider(
            "Min Cluster Size", 3, 15, 5, help="Minimum routes per cluster"
        )
        coverage_threshold = st.slider(
            "Coverage Threshold",
            0.85,
            0.98,
            0.92,
            step=0.01,
            help="Minimum stop coverage required",
        )
        # Sample size will be set after loading data
        sample_size = None

    # Load data
    with st.spinner("Loading transit data..."):
        gtfs_data = load_real_gtfs_data(uploaded_file)

    if gtfs_data is None:
        st.error("No GTFS data available. Please upload a GTFS zip file.")
        st.info("Expected file: `data/gtfs-accra-ghana-2016.zip`")
        return

    route_geometries = extract_all_route_geometries(gtfs_data)
    if route_geometries.empty:
        st.error("No route geometries could be extracted.")
        return

    # Set sample size slider now that we know the number of routes
    with st.sidebar:
        max_routes = len(route_geometries)
        default_sample = max_routes
        sample_size = st.slider(
            "Sample Size", 50, max_routes, default_sample, help="Routes to analyze"
        )

    # Simple network overview
    st.markdown(
        '<h2 class="section-header">📊 Network Overview</h2>', unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card info">
                <div class="metric-label">Total Routes</div>
                <div class="metric-value">{len(route_geometries)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card info">
                <div class="metric-label">Network Length</div>
                <div class="metric-value">{route_geometries["length_km"].sum():.0f} km</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card info">
                <div class="metric-label">Avg Route</div>
                <div class="metric-value">{route_geometries["length_km"].mean():.1f} km</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card info">
                <div class="metric-label">Longest Route</div>
                <div class="metric-value">{route_geometries["length_km"].max():.1f} km</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Simple run button
    st.markdown(
        """
        <div class="run-button">
            <h2>🚀 Ready to Optimize?</h2>
            <p>Click below to run the complete municipal optimization analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 RUN OPTIMIZATION", type="primary", use_container_width=True):
        try:
            start_time = time.time()

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: DTW similarity matrix
            status_text.text("Step 1/4: Computing route similarities...")
            progress_bar.progress(25)

            analyzer = ImprovedDTWAnalyzer(
                route_geometries, normalizer=CoordinateNormalizer("utm")
            )
            distance_matrix_scaled = analyzer.compute_improved_similarity_matrix(
                target_density=5, sample_size=sample_size
            )

            # Step 2: HDBSCAN clustering
            status_text.text("Step 2/4: Clustering similar routes...")
            progress_bar.progress(50)

            final_labels = density_based_clustering_leaf_method(
                distance_matrix_scaled,
                analyzer.valid_route_indices,
                min_cluster_size=min_cluster_size,
                min_samples=min_cluster_size,
            )

            # Step 3: PuLP optimization
            status_text.text("Step 3/4: Optimizing route selection...")
            progress_bar.progress(75)

            selected_routes, coverage_percent, final_count = (
                pulp_set_cover_optimization(
                    final_labels,
                    analyzer.valid_route_indices,
                    route_geometries,
                    gtfs_data,
                    coverage_threshold,
                )
            )

            # Step 4: Impact calculation
            status_text.text("Step 4/4: Calculating impact...")
            progress_bar.progress(100)

            impact_calculator = MunicipalImpactCalculator(route_geometries)
            impact = impact_calculator.calculate_comprehensive_impact(
                final_labels,
                analyzer.valid_route_indices,
                selected_routes,
                coverage_percent,
            )

            elapsed_time = time.time() - start_time

            # Store results
            st.session_state.update(
                {
                    "impact": impact,
                    "route_geometries": route_geometries,
                    "selected_routes": selected_routes,
                    "analysis_time": elapsed_time,
                    "gtfs_data": gtfs_data,
                }
            )

            progress_bar.empty()
            status_text.empty()

            st.success(f"✅ Optimization complete in {elapsed_time / 60:.1f} minutes!")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    # Display results if available
    if "impact" in st.session_state:
        display_simplified_results()


def display_simplified_results():
    """Display simplified, focused results"""

    impact = st.session_state.impact
    route_geometries = st.session_state.route_geometries
    selected_routes = st.session_state.selected_routes
    analysis_time = st.session_state.get("analysis_time", 0)
    gtfs_data = st.session_state.get("gtfs_data")

    # Results header
    st.markdown(
        '<h2 class="section-header">📈 Optimization Results</h2>',
        unsafe_allow_html=True,
    )

    # Key metrics in simple cards (white background, dark text)
    card_style = "background:#fff;color:#222;border:1px solid #e0e0e0;border-radius:10px;padding:1.5rem;text-align:center;"
    label_style = (
        "font-size:0.9rem;color:#666;text-transform:uppercase;letter-spacing:1px;"
    )
    value_style = "font-size:2rem;font-weight:bold;color:#222;margin:0.5rem 0;"
    subtitle_style = "font-size:0.9rem;color:#888;"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style='{card_style}'>
                <div style='{label_style}'>Route Reduction</div>
                <div style='{value_style}'>{impact["route_reduction_pct"]:.1f}%</div>
                <div style='{subtitle_style}'>{impact["eliminated_routes"]} routes eliminated</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style='{card_style}'>
                <div style='{label_style}'>Coverage Retained</div>
                <div style='{value_style}'>{impact["coverage_retention_pct"]:.1f}%</div>
                <div style='{subtitle_style}'>of original stops</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style='{card_style}'>
                <div style='{label_style}'>Annual Savings</div>
                <div style='{value_style}'>₵{impact["total_annual_savings_ghc"]:,.0f}</div>
                <div style='{subtitle_style}'>vs original network</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div style='{card_style}'>
                <div style='{label_style}'>CO₂ Saved</div>
                <div style='{value_style}'>{impact["annual_co2_saved_tonnes"]:.0f}</div>
                <div style='{subtitle_style}'>tonnes per year</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Simple network comparison
    st.markdown(
        '<h3 class="section-header">🗺️ Network Comparison</h3>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Original Network**")
        before_map = create_simple_map(route_geometries, "blue", "Original")
        st_folium(before_map, width=350, height=300, key="before_map")
        st.caption(f"All {len(route_geometries)} routes")

    with col2:
        st.markdown("**🎯 Optimized Network**")
        after_map = create_simple_map(
            route_geometries, "red", "Selected", selected_routes
        )
        st_folium(after_map, width=350, height=300, key="after_map")
        st.caption(f"{len(selected_routes)} selected routes")

    # Simple summary (white background, dark text)
    st.markdown(
        f"""
        <div style='background:#fff;color:#222;border:1px solid #e0e0e0;border-radius:10px;padding:2rem;margin:2rem 0;text-align:center;'>
            <h3 style='color:#1e3c72;'>🎯 Optimization Summary</h3>
            <p><strong>Routes:</strong> {impact["total_routes"]} → {impact["retained_routes"]} ({impact["route_reduction_pct"]:.1f}% reduction)</p>
            <p><strong>Coverage:</strong> {impact["coverage_retention_pct"]:.1f}% of stops maintained</p>
            <p><strong>Annual Savings:</strong> ₵{impact["total_annual_savings_ghc"]:,.0f}</p>
            <p><strong>Environmental Impact:</strong> {impact["annual_co2_saved_tonnes"]:.0f} tonnes CO₂ saved annually</p>
            <p><strong>Analysis Time:</strong> {analysis_time / 60:.1f} min</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Simple export options
    st.markdown(
        '<h3 class="section-header">📁 Export Results</h3>', unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        # Selected routes CSV
        selected_df = route_geometries[
            route_geometries["route_id"].isin(selected_routes)
        ]
        selected_csv = selected_df.drop("geometry", axis=1).to_csv(index=False)
        st.download_button(
            label="📄 Download Selected Routes (CSV)",
            data=selected_csv,
            file_name="accra_optimized_routes.csv",
            mime="text/csv",
        )

    with col2:
        # Impact summary JSON
        impact_summary = {
            "route_reduction_pct": impact["route_reduction_pct"],
            "coverage_retention_pct": impact["coverage_retention_pct"],
            "annual_savings_ghc": impact["total_annual_savings_ghc"],
            "annual_co2_saved_tonnes": impact["annual_co2_saved_tonnes"],
        }
        impact_json = json.dumps(impact_summary, indent=2)
        st.download_button(
            label="📊 Download Summary (JSON)",
            data=impact_json,
            file_name="accra_optimization_summary.json",
            mime="application/json",
        )

    with col3:
        # PDF Export (restored)
        if WEASYPRINT_AVAILABLE:
            if st.button("📄 Generate PDF Report", type="primary"):
                with st.spinner("Generating PDF report... This may take a minute."):
                    try:
                        pdf_data = generate_professional_pdf_report(
                            impact,
                            route_geometries,
                            selected_routes,
                            gtfs_data,
                            None,  # No viability
                            None,
                        )
                        if pdf_data:
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_data,
                                file_name=f"AccraIQ_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                            )
                        else:
                            st.warning("PDF generation failed.")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
        else:
            st.info("Install WeasyPrint: pip install weasyprint jinja2")


def create_simple_map(route_geometries, color, label, selected_routes=None):
    """Create a simple map for visualization"""
    bounds = route_geometries.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    if selected_routes:
        # Show only selected routes
        selected_gdf = route_geometries[
            route_geometries["route_id"].isin(selected_routes)
        ]
        routes_to_show = selected_gdf
    else:
        # Show all routes (sampled if too many)
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


if __name__ == "__main__":
    main()

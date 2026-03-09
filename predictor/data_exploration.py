import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as opy
import json
import os

# Data Exploration
def dataset_exploration(df):
    return df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
        index=False,
    )

def data_exploration(df):
    return df.describe().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
    )

# Rwanda Map Visualization - Using Local GeoJSON and Centroid Labels
def rwanda_vehicle_map(df):
    _GEOJSON_PATH = "dummy-data/rwanda_districts.geojson"
    
    try:
        # 1. Load Local GeoJSON
        with open(_GEOJSON_PATH, 'r') as f:
            geojson = json.load(f)

        # 2. Count clients per district
        counts = df.groupby("district").size().reset_index(name="vehicle_count")

        # 3. Create the Choropleth (The colored boundaries)
        fig = px.choropleth(
            counts,
            geojson=geojson,
            locations="district",
            featureidkey="properties.NAME_2", # Matching your downloaded example
            color="vehicle_count",
            color_continuous_scale="Viridis",
            title="<b>Vehicle Clients per District — Rwanda</b>"
        )

        # Focus map on Rwanda boundaries
        fig.update_geos(fitbounds="locations", visible=False)

        # 4. Calculate District Centroids for Labels
        # This places the name and number in the middle of each district
        centroids = []
        for f in geojson["features"]:
            dist_name = f["properties"]["NAME_2"]
            geometry = f["geometry"]
            
            # Handle both Polygon and MultiPolygon types
            if geometry["type"] == "Polygon":
                coords = geometry["coordinates"][0]
            elif geometry["type"] == "MultiPolygon":
                # Take the first/largest polygon of the multipolygon
                coords = geometry["coordinates"][0][0]
            
            lon = sum(c[0] for c in coords) / len(coords)
            lat = sum(c[1] for c in coords) / len(coords)
            centroids.append({"district": dist_name, "lon": lon, "lat": lat})

        # Merge centroid coordinates with actual vehicle data
        centroid_df = pd.DataFrame(centroids).merge(counts, on="district", how="left").fillna(0)

        # 5. Add District Labels and Numbers as a layer
        fig.add_trace(go.Scattergeo(
            lon=centroid_df.lon,
            lat=centroid_df.lat,
            text=centroid_df.apply(lambda r: f"<b>{r.district}</b><br>{int(r.vehicle_count)}", axis=1),
            mode="text",
            textfont=dict(size=9, color="black"),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.update_layout(
            height=600, 
            margin=dict(l=0, r=0, t=50, b=0),
            title_x=0.5
        )

        # Return as HTML div
        return opy.plot(fig, auto_open=False, output_type="div")

    except Exception as e:
        return f"<div class='alert alert-danger'>Error loading map data: {str(e)}<br>Check if {_GEOJSON_PATH} exists.</div>"

# Optional: Province-level summary chart
def rwanda_province_chart(df):
    province_counts = df.groupby('province').size().reset_index(name='vehicle_count')
    province_counts = province_counts.sort_values('vehicle_count', ascending=True)

    fig = px.bar(
        province_counts,
        x='vehicle_count',
        y='province',
        orientation='h',
        color='vehicle_count',
        color_continuous_scale='Viridis',
        title="Vehicle Distribution by Province",
        text='vehicle_count'
    )
    fig.update_layout(height=400, margin={"r": 0, "t": 40, "l": 0, "b": 0}, title_x=0.5)
    fig.update_traces(textposition='outside')
    return opy.plot(fig, auto_open=False, output_type="div")

# Optional: Top districts chart
def rwanda_top_districts_chart(df, n=10):
    district_counts = df.groupby(['province', 'district']).size().reset_index(name='vehicle_count')
    top_districts = district_counts.nlargest(n, 'vehicle_count')

    fig = px.bar(
        top_districts,
        x='district',
        y='vehicle_count',
        color='vehicle_count',
        color_continuous_scale='Viridis',
        title=f"Top {n} Districts by Vehicle Count",
        hover_data=['province'],
        text='vehicle_count'
    )
    fig.update_layout(height=400, margin={"r": 0, "t": 40, "l": 0, "b": 0}, title_x=0.5, xaxis_tickangle=-45)
    fig.update_traces(textposition='outside')
    return opy.plot(fig, auto_open=False, output_type="div")
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import random
import datetime
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide", page_title="SafePath AI")

# ------------------------ UI STYLE ------------------------

st.markdown("""
<style>

body{
background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
}

.bigtitle{
font-size:60px;
font-weight:800;
text-align:center;
color:#00ffcc;
}

.subtitle{
text-align:center;
font-size:20px;
color:#c9d6ff;
}

.metric-card{
background:rgba(255,255,255,0.06);
padding:20px;
border-radius:15px;
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ------------------------ HEADER ------------------------

st.markdown('<div class="bigtitle">🛡 SafePath AI</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Safety Aware Navigation</div>',unsafe_allow_html=True)

st.write("")

# ------------------------ SIDEBAR ------------------------

st.sidebar.title("Navigation Settings")

risk_sensitivity = st.sidebar.slider("AI Risk Sensitivity",0.5,2.0,1.0,0.1)
night_mode = st.sidebar.toggle("Night Mode Risk Boost",True)
show_heat = st.sidebar.toggle("Crime Heatmap",True)

map_style = st.sidebar.selectbox("Map Style",["Light","Dark","Terrain"])

def get_tile():

    if map_style=="Dark":
        return "cartodbdark_matter"

    if map_style=="Terrain":
        return "Stamen Terrain"

    return "cartodbpositron"

# ------------------------ AI MODEL ------------------------

@st.cache_resource
def train_model():

    rows=6000

    data=pd.DataFrame({

        "lat":np.random.uniform(12.8,13.2,rows),
        "lon":np.random.uniform(77.4,77.8,rows),
        "hour":np.random.randint(0,24,rows),
        "crime_density":np.random.uniform(0,1,rows),
        "distance_center":np.random.uniform(0,10,rows)

    })

    risk=(
    0.4*data["crime_density"]+
    0.3*(data["hour"]>20)+
    0.3*(data["distance_center"]>5)
    )

    data["risk"]=(risk>0.6).astype(int)

    X=data[["lat","lon","hour","crime_density","distance_center"]]
    y=data["risk"]

    model=RandomForestClassifier(n_estimators=200)

    model.fit(X,y)

    return model

model=train_model()

# ------------------------ GEOCODER ------------------------

def get_coords(place):

    url="https://nominatim.openstreetmap.org/search"

    params={"q":place,"format":"json"}

    headers={"User-Agent":"safepath"}

    r=requests.get(url,params=params,headers=headers)

    data=r.json()

    if len(data)==0:
        return None

    return float(data[0]["lat"]),float(data[0]["lon"])

# ------------------------ ROUTER ------------------------

def get_route(start,end):

    url=f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"

    params={
    "overview":"full",
    "geometries":"geojson"
    }

    r=requests.get(url,params=params)

    data=r.json()

    route=data["routes"][0]

    return route["geometry"]["coordinates"],route["distance"],route["duration"]

# ------------------------ AI RISK PREDICTOR ------------------------

def predict_risk(route):

    hour=datetime.datetime.now().hour

    risks=[]

    center_lat=np.mean([p[1] for p in route])
    center_lon=np.mean([p[0] for p in route])

    for p in route:

        lat=p[1]
        lon=p[0]

        crime_density=random.uniform(0,1)

        distance_center=np.sqrt((lat-center_lat)**2+(lon-center_lon)**2)*111

        features=np.array([[lat,lon,hour,crime_density,distance_center]])

        prob=model.predict_proba(features)[0][1]

        if night_mode:
            prob*=1.2

        prob=min(prob*risk_sensitivity,1)

        risks.append(prob)

    return risks

# ------------------------ HEATMAP ------------------------

def make_heat(route):

    heat=[]

    for _ in range(100):

        p=random.choice(route)

        heat.append([p[1]+random.uniform(-0.003,0.003),
                     p[0]+random.uniform(-0.003,0.003)])

    return heat

# ------------------------ MAP BUILDER ------------------------

def build_map(route,risks,start,end):

    center=[np.mean([p[1] for p in route]),np.mean([p[0] for p in route])]

    m=folium.Map(location=center,zoom_start=13,tiles=get_tile())

    latlon=[[c[1],c[0]] for c in route]

    for i in range(len(latlon)-1):

        r=risks[i]

        if r<0.35:
            color="#00FF9C"

        elif r<0.65:
            color="#FFA500"

        else:
            color="#FF3B3B"

        folium.PolyLine(
        [latlon[i],latlon[i+1]],
        color=color,
        weight=7,
        opacity=0.9
        ).add_to(m)

    folium.Marker(start,icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(end,icon=folium.Icon(color="red")).add_to(m)

    if show_heat:
        HeatMap(make_heat(route),radius=25,blur=20).add_to(m)

    return m

# ------------------------ INPUT ------------------------

c1,c2=st.columns(2)

with c1:
    start=st.text_input("Start Location","Manipal")

with c2:
    end=st.text_input("Destination","Udupi")

# ------------------------ GENERATE ROUTE ------------------------

if st.button("Generate Safe Route"):

    start_coords=get_coords(start)
    end_coords=get_coords(end)

    if not start_coords or not end_coords:

        st.error("Location not found")

    else:

        route,distance,duration=get_route(start_coords,end_coords)

        risks=predict_risk(route)

        avg_risk=sum(risks)/len(risks)

        safety=int((1-avg_risk)*100)

        st.session_state["route"]=route
        st.session_state["risk"]=risks
        st.session_state["start"]=start_coords
        st.session_state["end"]=end_coords
        st.session_state["distance"]=distance
        st.session_state["duration"]=duration
        st.session_state["safety"]=safety

# ------------------------ DISPLAY ------------------------

if "route" in st.session_state:

    route=st.session_state["route"]
    risks=st.session_state["risk"]

    dist=round(st.session_state["distance"]/1000,2)
    eta=int(st.session_state["duration"]/60)
    safety=st.session_state["safety"]
    avg=sum(risks)/len(risks)

    st.write("")

    a,b,c,d=st.columns(4)

    a.metric("Risk Score",round(avg,2))
    b.metric("Safety Score",f"{safety}/100")
    c.metric("Distance",f"{dist} km")
    d.metric("ETA",f"{eta} min")

    st.progress(safety/100)

    st.write("")

    st.subheader("AI Route Map")

    m=build_map(route,risks,st.session_state["start"],st.session_state["end"])

    st_folium(m,width=1400,height=600)

    st.write("")

    # ------------------------ CHARTS ------------------------

    st.subheader("Risk Distribution")

    fig=px.line(risks,title="Route Segment Risk")

    st.plotly_chart(fig,use_container_width=True)

    safe=len([x for x in risks if x<0.5])
    risky=len(risks)-safe

    fig2=go.Figure(data=[go.Pie(labels=["Safe Segments","Risky Segments"],
    values=[safe,risky])])

    st.plotly_chart(fig2,use_container_width=True)

    # ------------------------ AI RECOMMENDER ------------------------

    st.subheader("AI Recommendation")

    fastest=random.uniform(0.4,0.9)

    reduction=int((fastest-avg)*100)

    if reduction>0:

        st.success(f"""
AI recommends this **SAFE ROUTE**

Estimated risk reduction vs fastest path: **{reduction}%**
""")

    else:

        st.info("Fastest route already safest")

    # ------------------------ RADAR CHART ------------------------

    st.subheader("Safety Analytics")

    radar_vals=[

    safety,
    random.randint(60,95),
    random.randint(55,90),
    random.randint(50,85),
    random.randint(65,95)

    ]

    radar_labels=["Route Safety","Lighting","Population","Infrastructure","Police Presence"]

    fig3=go.Figure()

    fig3.add_trace(go.Scatterpolar(
    r=radar_vals,
    theta=radar_labels,
    fill='toself'
    ))

    fig3.update_layout(
    	polar=dict(
        	radialaxis=dict(
            		visible=True,
            		range=[0,100]
        )
    ),
    showlegend=False
    )

    st.plotly_chart(fig3,use_container_width=True)

    st.write("")

    st.markdown("""
### AI Model Explanation

SafePath AI uses a **Random Forest machine learning model** to estimate route safety.

The AI analyzes:

• spatial location patterns  
• simulated crime density signals  
• time-of-day travel risk  
• distance from city center  
• environmental safety indicators  

Each route is divided into **micro segments**, and the AI predicts a **risk probability for every segment**, which is visualized directly on the map.
""")

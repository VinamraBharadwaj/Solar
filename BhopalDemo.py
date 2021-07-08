#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import ee
import numpy as np
import math
from scipy.optimize import fsolve
import urllib.request as ulr
import warnings
import plotly.express as px
import geopandas as gpd
from PIL import Image
import json
import os


# In[2]:
service_account = 'vinamrabharadwaj@gisproject-319220.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'https://raw.githubusercontent.com/VinamraBharadwaj/SolarWebApp/BhopalDemo/gisproject-319220-3d22c76c6e29.json')
ee.Initialize(credentials)
#ee.Authenticate()
#ee.Initialize()


# In[55]:


#Adding title, cover picture and background color to the interface
st.title("Estimating Rooftop Solar Potential")
st.text("Model Demo Based on Smart City Area Bhopal")


# In[4]:

imageurl = 'https://raw.githubusercontent.com/VinamraBharadwaj/SolarWebApp/main/Slide2.PNG'
img = ulr.urlopen(imageurl)

image = Image.open(img)
st.image(image,use_column_width=True)
st.markdown('<style>body{background-color: black;}</style>',unsafe_allow_html=True)


# In[5]:


#@st.cache()
collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA').filterDate('2020-01-01', '2020-12-31').filterBounds(ee.Geometry.Point(77.4126,23.2599))
count = collection.size()


# In[34]:


n = 12 # count.getInfo()
colList = collection.toList(n)
colpd = pd.DataFrame()
for i in range(n-1):
  img = ee.Image(colList.get(i))
  new = dict(img.getInfo())
  new_2 = dict(new['properties'])
  new_2['id'] = new['id']
  new_2.pop('system:footprint')
  colpd = colpd.append(new_2, ignore_index = True)


# In[35]:


col = colpd[['id','SUN_ELEVATION','EARTH_SUN_DISTANCE','RADIANCE_MULT_BAND_2','RADIANCE_MULT_BAND_3','RADIANCE_MULT_BAND_4','RADIANCE_MULT_BAND_5','RADIANCE_MULT_BAND_6','RADIANCE_MULT_BAND_7','RADIANCE_MULT_BAND_9','RADIANCE_ADD_BAND_2','RADIANCE_ADD_BAND_3','RADIANCE_ADD_BAND_4','RADIANCE_ADD_BAND_5','RADIANCE_ADD_BAND_6','RADIANCE_ADD_BAND_7','RADIANCE_ADD_BAND_9']]


# In[39]:


esun = [2067, 1893, 1603, 972.6, 245, 79.72,399.7]
mult = (list(col))[3:10]
add = (list(col))[10:18]


# In[37]:


dn = 655.35
q = np.mean(90 - col['SUN_ELEVATION'])*(np.pi/180)
d = np.mean(col['EARTH_SUN_DISTANCE'])
p = math.pi

def calc():
    gres = []
    for i in range(7):
        m = np.mean(col[mult[i]])
        a = np.mean(col[add[i]])
        e = esun[i]
        
        def f1(x):
            return (m*dn + a - (0.01*((e*np.cos(q)-3*p*x)*(e*np.cos(q))**np.arctan(q))/(p*(d**2)*(e*np.cos(q)-4*p*x)**np.arctan(q))) - x)
        
        z = fsolve(f1,1)
        
        lp = float(z)
        t = -np.cos(q)*np.log(1-(4*p*lp/(e*(np.cos(q)))))
        td = np.exp(-t/np.cos(q))
        ed = 4*lp
        g = e*np.cos(q)*td+ed
        
        gres.append(g)
    
    return gres
    
ans = calc()


# In[38]:


G = np.mean(ans)


# In[29]:


R = 0.18
P = 0.75
I = 0.4


# In[43]:


#geojson = gpd.read_file("D:/ABD/ABD_BLU.geojson")
response = ulr.urlopen('https://raw.githubusercontent.com/VinamraBharadwaj/Solar/main/ABD_BLU.geojson')
geojson = gpd.read_file(response)


# In[45]:


geojson= geojson.to_crs({'init': 'epsg:32643'})
geojson["area"] = round(geojson['geometry'].area,2)
warnings.filterwarnings("ignore")


# In[46]:


def func_sub(geojson):
    if geojson['Landuse'] == 'Residential':
        sub = 0.6
    else:
        sub = 1
    return sub

def func_tar(geojson):
    if geojson['Landuse'] == 'Residential':
        tar = 5
    elif geojson['Landuse'] == 'PSP':
        tar = 2
    elif geojson['Landuse'] == 'Recreation':
        tar = 3.5
    else:
        tar = 7
    return tar

def func_need(geojson):
    if geojson['Landuse'] == 'Residential':
        need = 60*geojson["area"]
    elif geojson['Landuse'] == 'PSP':
        need = 70*geojson["area"]
    elif geojson['Landuse'] == 'Recreation':
        need = 25*geojson["area"]
    else:
        need = 100*geojson["area"]
    return need

def func_feasibility(geojson):
    if geojson['a_need'] > geojson['a_gen']:
        feasibility = 0
    else:
        feasibility = 1
    return feasibility

def func_totfea(geojson):
    if (geojson['feasibility'] == 1 and geojson['recovery_time'] < 20):
        total_feasibility = 1
    else:
        total_feasibility = 0
    return total_feasibility


# In[47]:


geojson['a_gen'] = I * geojson['area'] * R * G * P

geojson['subisidy'] = geojson.apply(func_sub,axis=1)

geojson['tariff'] = geojson.apply(func_tar,axis=1)

geojson['cost'] = round(geojson['subisidy']*geojson['a_gen']*47000/1500,2)

geojson['life_time_profit'] = round(geojson['a_gen']*geojson['tariff']*25 - geojson['cost'],2)

geojson['a_need'] = round(geojson.apply(func_need,axis=1),2)

geojson['feasibility'] = round(geojson.apply(func_feasibility,axis=1),2)

geojson['recovery_time'] = round(geojson['cost']*25/geojson['life_time_profit'],2)

geojson['final_feasibility'] = round(geojson.apply(func_totfea,axis=1),2)


# In[49]:


response = ulr.urlopen('https://raw.githubusercontent.com/VinamraBharadwaj/SolarWebAPP/main/ABD_BLU.geojson')
ABD_json = json.loads(response.read())


# In[22]:


if not st.checkbox('Hide Animation', False, key = '1'):
    fig1 = px.choropleth_mapbox(geojson, geojson=ABD_json, title = 'Supply v/s Demand Gap', color="feasibility",
                    locations="OBJECTID", featureidkey="properties.OBJECTID",
                    range_color = (0,1),center={"lat": 23.231293602324225, "lon": 77.39709988601008},
                           mapbox_style="open-street-map",hover_name = 'Landuse',
                           zoom=13.5, color_continuous_scale=[(0.00, "purple"),   (0.5, "purple"),
                                                     (0.5, "yellow"),  (1.00, "yellow")])
    fig1.update_layout(coloraxis_colorbar=dict(
        title="Supply v/s Demand",
        tickvals=[0,1],
        ticktext=["Not Satisfied", 'Satisfied'],
        lenmode="pixels", len=100,
    ))
    st.plotly_chart(fig1)


# In[23]:


if not st.checkbox('Hide Animation', False, key = '2'):
    fig2 = px.choropleth_mapbox(geojson, geojson=ABD_json,title = "Profit in Life Time i.e. 25 years",
                    color="life_time_profit",
                    locations="OBJECTID", featureidkey="properties.OBJECTID",
                    range_color = (0,2500000),
                    hover_name = 'Landuse',
                    center={"lat": 23.231293602324225, "lon": 77.39709988601008},
                    mapbox_style="open-street-map",zoom=13.5,
                   )
    st.plotly_chart(fig2)


# In[24]:


if not st.checkbox('Hide Animation', False, key = '3'):
    fig3 = px.choropleth_mapbox(geojson, title = "Feasibility based on Recovery Period",
                           geojson=ABD_json,
                           locations="OBJECTID", featureidkey="properties.OBJECTID",
                           color="final_feasibility",
                           center={"lat": 23.231293602324225, "lon": 77.39709988601008},
                           mapbox_style="open-street-map",hover_name = 'Landuse',
                           zoom=13.5, color_continuous_scale=[(0.00, "purple"),   (0.5, "purple"),
                                                     (0.5, "yellow"),  (1.00, "yellow")])
    
    fig3.update_layout(coloraxis_colorbar=dict(
        title="Feasibility",
        tickvals=[0,1],
        ticktext=["Not Feasible", 'Feasible'],
        lenmode="pixels", len=100))
    
    st.plotly_chart(fig3)


# In[ ]:





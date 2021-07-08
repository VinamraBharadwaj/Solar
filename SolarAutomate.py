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
from geopy.geocoders import Nominatim
import os
import webbrowser
import sys

ee.Initialize()
geolocator = Nominatim(user_agent=os.path.abspath(sys.argv[0]))

st.title("Estimating Rooftop Solar Potential")
st.text("Know your building's solar potential and make the right decision! Protect environment!!")

imageurl = 'https://raw.githubusercontent.com/VinamraBharadwaj/SolarWebApp/main/Slide1.PNG'
img = ulr.urlopen(imageurl)

image = Image.open(img)
st.image(image,use_column_width=True)
st.markdown('<style>body{background-color: black;}</style>',unsafe_allow_html=True)

form = st.form(key='my-form')
place = form.text_input('Enter City Name:')
area = form.text_input('Rooftop Area (m.sq.):')
cover = form.slider("Percentage of Roof for Solar Installation:")
btype = form.selectbox('Select Category of Building:',('Residential', 'Commercial', 'PSP', 'Recreation'))
tariff = form.text_input('Averge Electricity Cost (₹/kWh):')
submit = form.form_submit_button('Submit')

if submit:
    #st.header('ss')
    location = geolocator.geocode(place)
    lat = location.latitude
    lon = location.longitude
    collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA').filterDate('2020-01-01', '2020-12-31').filterBounds(ee.Geometry.Point(lon,lat))
    count = collection.size()

    n = 2#count.getInfo()
    colList = collection.toList(n)
    colpd = pd.DataFrame()
    for i in range(n-1):
        img = ee.Image(colList.get(i))
        new = dict(img.getInfo())
        new_2 = dict(new['properties'])
        new_2['id'] = new['id']
        new_2.pop('system:footprint')
        colpd = colpd.append(new_2, ignore_index = True)
    col = colpd [['id','SUN_ELEVATION','EARTH_SUN_DISTANCE','RADIANCE_MULT_BAND_2','RADIANCE_MULT_BAND_3','RADIANCE_MULT_BAND_4','RADIANCE_MULT_BAND_5','RADIANCE_MULT_BAND_6','RADIANCE_MULT_BAND_7','RADIANCE_MULT_BAND_9','RADIANCE_ADD_BAND_2','RADIANCE_ADD_BAND_3','RADIANCE_ADD_BAND_4','RADIANCE_ADD_BAND_5','RADIANCE_ADD_BAND_6','RADIANCE_ADD_BAND_7','RADIANCE_ADD_BAND_9']]


    esun = [2067, 1893, 1603, 972.6, 245, 79.72,399.7]
    mult = (list(col))[3:10]
    add = (list(col))[10:18]
    #st.write(add)


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

    def func_sub():
        if btype == 'Residential':
            sub = 40
        else:
            sub = 0
        return sub


    def func_need():
        if btype == 'Residential':
            need = 60*A
        elif btype == 'PSP':
            need = 70*A
        elif btype == 'Recreation':
            need = 25*A
        else:
            need = 100*A
        return need

    
    def func_feasibility():
        if need > E:
            feasibility = 'Not Satistied'
        else:
            feasibility = 'Satistied'
        return feasibility

    
    def func_totfea():
        if (feasibility == 'Satistied' and recovery_time < 20):
            total_feasibility = 'Feasibile'
        else:
            total_feasibility = 'Not Feasible'
        return total_feasibility

    
    G = round(np.mean(ans),2)
    
    A = float(area)
    R = 0.18
    P = 0.75
    I = int(cover)
    T = int(tariff)
    
    E = round(G*A*I*R*P,2)
    
    sub = func_sub()
    
    cost = round((100-sub)*E*47000/150000,2)
    
    life_time_profit = round(E*T*25-cost,2)
    
    need = func_need()
    
    feasibility = func_feasibility()
    
    recovery_time = round(cost*25/life_time_profit,2)
    
    final_feasibility = func_totfea()
    
    
    
    ds = {'Solar Irradiance (kWh/m.sq.)':G, 'Annual Energy Generation (kWh)' : E, 'Subsidy (%):': sub, 'Cost of Installation (₹)':cost, 'Profit in Life Time i.e. 25 years (₹)': life_time_profit, 'Approximate Need (kWh/year):' :need, 'Supply vs Demand:' : feasibility, 'Recovery Time (years):':recovery_time, 'Feasibility:': final_feasibility}
    
    df = pd.Series(ds).to_frame('')
    st.table(df)
    
    st.write('Installation of Solar Panel for your building is ' + final_feasibility)
    st.write('')
    st.write('')
    
url = 'https://share.streamlit.io/vinamrabharadwaj/covid19webapp/main/covid_webapp.py'

#submit1 = st.form_submit_button('See Bhopal Demo')
if st.button('See Bhopal Demo'):
    webbrowser.open_new_tab(url)
        
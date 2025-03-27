import os, sys, time, datetime, traceback
import spaceteams as st
import numpy as np
import pandas as pd

moon = st.ProcPlanet.DataStore()



moon.AddGeoBinAltimetryLayer(3.0, "../SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m", st.ProcPlanet.GeoBin_Extra_Args()) # Most important is the (kinda) high-res data over Nobile Rim 1.
moon.AddGeoBinAltimetryLayer(2.0, "../SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPole75_30m", st.ProcPlanet.GeoBin_Extra_Args()) # Second is the kind of okay res data over some parts we might miss with the above data.
moon.AddGeoBinAltimetryLayer(1.0, "../SharedData/PlanetData/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014", st.ProcPlanet.GeoBin_Extra_Args()) # Last is the really crummy global data so worst-case scenario we still have *something*. 

boundSize = 3
latLongStart = (np.deg2rad(-85.292466-boundSize), np.deg2rad(36.920242-boundSize)) # First term is latitude, then longitude. Note that it should be in radians.
latLongEnd = (np.deg2rad(-85.292466+boundSize), np.deg2rad(36.920242+boundSize)) # Same as above
inbetweenLatPoints = 100 # Other way of doing this is range(). We might actually want to use this (since we want square cells for A*). But I think I have a solution...
inbetweenLongPoints = round(abs((latLongEnd[1]-latLongStart[1])/(latLongEnd[0]-latLongStart[0])*inbetweenLatPoints)) # Find step size for Latitude and then divide the Longitude "length" by it to get the points for the longitude? That'll keep it square? I think?

radius = 1737400.0 # Radius of moon in meters

d = 0 #[m]distance from start to sample point (sum of d_steps)
d_max = 200 #[m]maximum distance for an individual step
d_catch = 60000 #[m] "catch-all distance" where we stop iterating
max_horizon_angle = [] #maximum angle from sample point to horizon (epsilon in alex's example)
degrees_between_azimuths = 1 #[degrees] rotation angle between each azimuth
rotations = 360/degrees_between_azimuths #number of steps the azimuth angle rotates through

azimuths = np.linspace(0,2*np.pi,rotations)

lat_steps = 100
lon_steps = 100
latitudes = np.linspace(np.deg2rad(-85.5),np.deg2rad(-84),lat_steps)
longitudes = np.linspace(np.deg2rad(28),np.deg2rad(38),lon_steps)

def d_step(d,d_max):
    step = d_max - np.exp(-0.095*(d/1000))

tan6 = np.tan(np.deg2rad(6))
data = {
    "latitude" : [],
    "longitude" : [],
    "illumination" : []
}

for lat in latitudes:
    for lon in longitudes:
        #loop over each latitude longitude
        illuminated = False
        for az in azimuths:
            #loop over each azimuth (viewing angle)
            d = 0
            loc = st.PlanetUtils.LLA_to_PCPF(st.PlaentUtils.LatLonAlt(lat,lon,2000), radius)
            #get start location
            flu = st.PlanetUtils.ForwardLeftUpFromAzimuth(loc,az,radius)
            #get frame of reference
            f_hat = flu.forward()
            #get forward facing fector
            if illuminated == True:
                break
                #dont need to loop over every angle if the site is illuminated from any single direction
            else:
                while d < d_catch:
                    #step along direction and test for illumination
                    d += d_step(d,d_max)
                    loc_sample = loc + f_hat*d #stepping
                    h = np.linalg.norm(st.ProcPlanet.SampleGround(moon, loc, radius, 0.0, 16) - loc_sample) #height = start location elevation - sample location elevation
                    if h/d > tan6 and az != azimuths[-2]:
                        #if angle gets above 6 degrees (shaded) and the azimuth angle is less than the maximum azimuth angle (2*pi) continue to next azimuth angle
                        illuminated == False
                        break

                    elif h/d > tan6 and az == azimuths[-2]:
                        #if angle gets above 6 degrees (shaded) and the azimuth angle is at its maximum value, consider the site a PSR
                        illuminated == False
                        data["latitude"].append(lat)
                        data["longitude"].append(lon)
                        data["illumination"].append(0)                        
                        break  

                    elif d >= d_catch:
                        #if catch distance is reached without triggering the previous if statement, site is illuminated from current azimuth angle, and considered illuminated
                        illuminated = True
                        data["latitude"].append(lat)
                        data["longitude"].append(lon)
                        data["illumination"].append(1)
                        break

df = pd.DataFrame(data)
df.to_csv("illumination.csv")
                



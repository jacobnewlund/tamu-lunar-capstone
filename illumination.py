import os, sys, time, datetime, traceback
import spaceteams as st
import numpy as np
import pandas as pd

moon = st.ProcPlanet.DataStore()



# moon.AddGeoBinAltimetryLayer(3.0, "../SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m", st.ProcPlanet.GeoBin_Extra_Args()) # Most important is the (kinda) high-res data over Nobile Rim 1.
# moon.AddGeoBinAltimetryLayer(2.0, "../SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPole75_30m", st.ProcPlanet.GeoBin_Extra_Args()) # Second is the kind of okay res data over some parts we might miss with the above data.
# moon.AddGeoBinAltimetryLayer(1.0, "../SharedData/PlanetData/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014", st.ProcPlanet.GeoBin_Extra_Args()) # Last is the really crummy global data so worst-case scenario we still have *something*. 

moon.AddGeoBinAltimetryLayer(3.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m", st.ProcPlanet.GeoBin_Extra_Args())
moon.AddGeoBinAltimetryLayer(2.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPole75_30m", st.ProcPlanet.GeoBin_Extra_Args())
moon.AddGeoBinAltimetryLayer(1.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014", st.ProcPlanet.GeoBin_Extra_Args()) 


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
degrees_between_azimuths = 10 #[degrees] rotation angle between each azimuth
rotations = 360/degrees_between_azimuths #number of steps the azimuth angle rotates through

azimuths = np.linspace(0,2*np.pi,int(rotations))

lat_steps = 455
lon_steps = 277
latitudes = np.linspace(np.deg2rad(-85.5),np.deg2rad(-84),lat_steps)
longitudes = np.linspace(np.deg2rad(28),np.deg2rad(38),lon_steps)

# latitudes = np.linspace(np.deg2rad(-85.15016),np.deg2rad(-84),lat_steps)
# longitudes = np.linspace(np.deg2rad(35.34668),np.deg2rad(38),lon_steps)
def d_step(d,d_max):
    return d_max - np.exp(-0.095*(d/1000 - 39))

tan6 = np.tan(np.deg2rad(6))
data = {
    "latitude" : [],
    "longitude" : [],
    "illumination" : []
}
steps = 0
for lat in latitudes:
    tan = np.tan(np.deg2rad(1.5 + (90 + np.rad2deg(lat))))
    print(round(steps/(lat_steps*lon_steps)*100,2), "%")
    for lon in longitudes:
        # if int(steps/(lat_steps*lon_steps)*100) % 5 == 0:
        #     print(round(steps/(lat_steps*lon_steps)*100,2), "%")
        #loop over each latitude longitude
        illuminated = False
        for az in azimuths:
            #loop over each azimuth (viewing angle)
            d = 0
            loc = st.PlanetUtils.LLA_to_PCPF(st.PlanetUtils.LatLonAlt(lat,lon,2000), radius)
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
                    loc_sample = st.ProcPlanet.SampleGround(moon, loc_sample, radius, 0.0, 16)[0]
                    h = -np.linalg.norm(st.ProcPlanet.SampleGround(moon, loc, radius, 0.0, 16)[0]) + np.linalg.norm(loc_sample) #height = start location elevation - sample location elevation
                    # print("d:",d)
                    # print("h:",h)
                    # print(h/d, tan6)
                    if h/d > tan and az != azimuths[-2]:
                        #if angle gets above 6 degrees (shaded) and the azimuth angle is less than the maximum azimuth angle (2*pi) continue to next azimuth angle
                        illuminated = False
                        break

                    elif h/d > tan and az == azimuths[-2]:
                        #if angle gets above 6 degrees (shaded) and the azimuth angle is at its maximum value, consider the site a PSR
                        illuminated = False
                        data["latitude"].append(np.rad2deg(lat) )
                        data["longitude"].append(np.rad2deg(lon) )
                        data["illumination"].append(0)                        
                        break  

                    elif d >= d_catch:
                        #if catch distance is reached without triggering the previous if statement, site is illuminated from current azimuth angle, and considered illuminated
                        illuminated = True
                        data["latitude"].append(np.rad2deg(lat) )
                        data["longitude"].append(np.rad2deg(lon) )
                        data["illumination"].append(1)
                        break
        steps+=1
print("100%")

df = pd.DataFrame(data)
df.to_csv("illumination.csv")
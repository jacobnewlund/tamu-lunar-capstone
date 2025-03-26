# THIS COMMENT LINE SHOULD BE THE FIRST LINE OF THE FILE!
# DON'T CHANGE ANY OF THE BELOW; NECESSARY FOR JOINING SIMULATION
import os, sys, time, datetime, traceback
import spaceteams as st
# def custom_exception_handler(exctype, value, tb):
#     error_message = "".join(traceback.format_exception(exctype, value, tb))
#     st.logger_fatal(error_message)
#     exit(1)
# sys.excepthook = custom_exception_handler
# st.connect_to_sim(sys.argv)
import numpy as np
# DON'T CHANGE ANY OF THE ABOVE; NECESSARY FOR JOINING SIMULATION
################################################################

# The Actual Data Grabbing Part
#   We want to get the data over some latitude and longitude bounds, crunch it so that we get elevation and grade data (illumination data will come later), and get it A*-able.
#   And then spit it out in a .csv file... maybe. Or we could just put it all in here during the initial phases of development. When this thing gets huge we'll probably not
#   want to recalculate the data every time we run the program.
#
#   The steps to do this is to first define the bounds. My best guess is to assign two points as the corners and then fill in from there in evenly-spaced points. You know, linspace().
#   Due to the fineness of the data we'll probably run into some floating-point shenanigans in the lats and longs. I surely hope Space Teams PRO is robust enough to handle this. 
#   If not, then we're in trouble.
#
#   The basic data pulled from 'ground' can be used to get the elevation and grade. Illumination is a big deal for us, though, so Alex proposes we can get that done by a complicated process.
#   The process is outlined in a paper he co-wrote in the Resources tab in Teams.
#   What about roughness? I have no idea. That's going to be someone else's problem. 
# 
#   But there would be the four values used in the weights for A*. Then we get a simple path... but we aren't just using one transportation system so we can't just do things the simple way.
#   Good news is that for the data-grabbing we don't actually care about multi-modal transportation systems yet. That's for later. 

# First, grab the data from Nobile Rim 1.
moon = st.ProcPlanet.DataStore()
radius = 1737400.0 # Radius of moon in meters (required for absolutely everything DO NOT CHANGE)

moon.AddGeoBinAltimetryLayer(3.0, "../SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m", st.ProcPlanet.GeoBin_Extra_Args()) # Most important is the (kinda) high-res data over Nobile Rim 1.
moon.AddGeoBinAltimetryLayer(2.0, "../SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPole75_30m", st.ProcPlanet.GeoBin_Extra_Args()) # Second is the kind of okay res data over some parts we might miss with the above data.
moon.AddGeoBinAltimetryLayer(1.0, "../SharedData/PlanetData/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014", st.ProcPlanet.GeoBin_Extra_Args()) # Last is the really crummy global data so worst-case scenario we still have *something*. 

# uncomment the below lines if you couldn't stuff the STPRO files into your main drive
# moon.AddGeoBinAltimetryLayer(3.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m", st.ProcPlanet.GeoBin_Extra_Args())
# moon.AddGeoBinAltimetryLayer(2.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPole75_30m", st.ProcPlanet.GeoBin_Extra_Args())
# moon.AddGeoBinAltimetryLayer(1.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014", st.ProcPlanet.GeoBin_Extra_Args()) 

# As an aside: USEFUL LONG/LATITUDE STUFF:
# Center of Nobile Rim 1: Lat -85.292466 Long 36.920242
# dot of desired location -85.292466, 36.920242
# im 2 landing site -84.7906, 29.1957
# ground [0] = position at surface in cartesian planet-centered planet-fixed (PCPF) frame
# ground [1] = normal vector (also in same frame)

# Then, define the bounds. I'm a fan of actually readable variable names because I like typing more and wasting my time (just kidding, but still - it means less comments later)
boundSize = 3
# latLongStart = (np.deg2rad(-85.292466-0.5), np.deg2rad(36.920242-boundSize*3)) # First term is latitude, then longitude. Note that it should be in radians.
# latLongEnd = (np.deg2rad(-85.292466+0.5), np.deg2rad(36.920242+boundSize*3)) # Same as above
# or use the LROC nobile rim 1 bounds
latLongStart = (np.deg2rad(-85.5), np.deg2rad(38)) # First term is latitude, then longitude. Note that it should be in radians.
latLongEnd = (np.deg2rad(-84), np.deg2rad(28)) # Same as above

# -85.35140, 31.64367
# -85.89749, 36.47997
# -85.47479, 43.20664
# -84.97460, 38.05472

# Testing "squareness" of bounds for the fun of it:
latDist = np.linalg.norm(st.PlanetUtils.LLA_to_PCPF(st.PlanetUtils.LatLonAlt(latLongStart[0], latLongEnd[0], 2000.0), radius))
longDist = np.linalg.norm(st.PlanetUtils.LLA_to_PCPF(st.PlanetUtils.LatLonAlt(latLongStart[1], latLongEnd[1], 2000.0), radius))
if latDist/longDist == 1.0:
    print("Latitude/longitude bounds square")
else:
    print("Latitude/longitude bounds not square, {}".format(latDist/longDist))

inbetweenLatPoints = 200 # change these to increase fineness of data
inbetweenLongPoints = 200


# Now, we need to actually run over these bounds. I'll give you one guess how we do that. (it's a for loop!!!)
indexLat = 0 # we need a way of actually assigning the data to the correct matrix index.
indexLong = 0

dataMatrix = np.ones((inbetweenLatPoints, inbetweenLongPoints), dtype=object) # yes, we're using the Big Matrix strategy here. I don't like it either. Most efficient way to do it would be to just spit it directly into the .csv but I think the flexibility of doing it like this is a positive.
print("Matrix dimensions: {}x{}".format(inbetweenLatPoints, inbetweenLongPoints))

for latitude in np.linspace(latLongEnd[0], latLongStart[0], inbetweenLatPoints): # for every latitude in our bounds...
    for longitude in np.linspace(latLongStart[1], latLongEnd[1], inbetweenLongPoints): # for every longitude in our bounds...
        # The below code is almost all Alex boilerplate. No, I don't really understand it. 
        lla = st.PlanetUtils.LatLonAlt(latitude, longitude, 2000.0) # Latitude, then longitude, then 2000.0. Forgot what the two thousand meant.
        loc_pcpf = st.PlanetUtils.LLA_to_PCPF(lla, radius)
        ground = st.ProcPlanet.SampleGround(moon, loc_pcpf, radius, 0.0, 16) # that zero is the reference off the ground. The sixteen? No idea.
        ground = [ground[0][0], ground[0][1], ground[0][2], ground[1][0], ground[1][1], ground[1][2], np.rad2deg(latitude), np.rad2deg(longitude)] # I don't know if this is the best way of doing it. 
        dataMatrix[indexLat][indexLong] = ground
        indexLong += 1
    indexLong = 0
    indexLat += 1

print("Matrix created") # We did it!!!


# Now, I'm just going to try to make a nice graphical representation of the data. Matplotlib should do it.
#   As a test, I'm going to try to make a heightmap using a funky wireframe plot I found online. It may look readable. I hope.
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cbook, cm
from matplotlib.colors import LightSource

# Load and format data
X = np.array([])
Y = np.array([])
Z = np.array([])

indexLat = 0
indexLong = 0

# recall - moon radius is 1737400 m
# (array([  139784.47206655, -1398598.92428207,  1018217.8083578 ]), array([ 0.06167599, -0.81263999,  0.57949315]))

# Load in data using the matrix and the same for loops (for that latitude/longitude data to make it scale properly)
for latitude in np.linspace(latLongEnd[0], latLongStart[0], inbetweenLatPoints): # for every latitude in our bounds...
    for longitude in np.linspace(latLongStart[1], latLongEnd[1], inbetweenLongPoints): # for every longitude in our bounds...
        cell = dataMatrix[indexLat][indexLong] 
        altitude = np.sqrt(cell[0]**2 + cell[1]**2 + cell[2]**2) - radius
        
        X = np.append(X, cell[-2])
        Y = np.append(Y, cell[-1])
        Z = np.append(Z, altitude)
        indexLong += 1
    indexLong = 0
    indexLat += 1

X.resize(inbetweenLatPoints, inbetweenLongPoints)
Y.resize(inbetweenLatPoints, inbetweenLongPoints)
Z.resize(inbetweenLatPoints, inbetweenLongPoints)


fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot_surface(X, Y, Z, cmap=cm.managua)
ax.set_xlabel("latitude (deg)")
ax.set_ylabel("longitude (deg)")


plt.show()

# spicey py (pip-install spiceypy) for the fancy sun vector stuff psuedo-ray tracing
# Moon frame: "MOON_PA"

plt.close() 
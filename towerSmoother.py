import os, sys, time, datetime, traceback
import spaceteams as st

import numpy as np
import pandas as pd
import csv

# oh good more haversine formula how have I missed you
def havDist(lat1, lat2, long1, long2, radius): # gets dist between two lat-longs. IT NEEDS RADIANS AS INPUT. OTHERWISE YOU WON'T LIKE WHAT YOU GET OUTTA IT
    return 2*radius*np.arcsin(np.sqrt((1 - np.cos(lat2-lat1) + np.cos(lat1)*np.cos(lat2)*(1 - np.cos(long2-long1)))/2))

def hav(theta):
    return (1-np.cos(theta))/2

def archav(angle):
    return 2*np.arcsin(np.sqrt(angle))

def elevAtPoint(lat, long):
    lla = st.PlanetUtils.LatLonAlt(np.deg2rad(lat), np.deg2rad(long), 2000.0) # Latitude, then longitude, then 2000.0. Forgot what the two thousand meant.
    loc_pcpf = st.PlanetUtils.LLA_to_PCPF(lla, radius)
    ground = st.ProcPlanet.SampleGround(moon, loc_pcpf, radius, 0.0, 16) # that zero is the reference off the ground. The sixteen? No idea.
    return np.linalg.norm(ground[0])

# I have to rip off much of the datagrabber code since it's back to the elevation grind


moon = st.ProcPlanet.DataStore()
radius = 1737400.0 # Radius of moon in meters (required for absolutely everything DO NOT CHANGE)

moon.AddGeoBinAltimetryLayer(3.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m", st.ProcPlanet.GeoBin_Extra_Args())
moon.AddGeoBinAltimetryLayer(2.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPole75_30m", st.ProcPlanet.GeoBin_Extra_Args())
moon.AddGeoBinAltimetryLayer(1.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014", st.ProcPlanet.GeoBin_Extra_Args()) 

towerLocs = []

with open('tower_locations_smoother.csv', mode='r') as towerFile:
    towerReader = csv.reader(towerFile, delimiter=',')
    # there is no header
    for row in towerReader:
        towerLocs.append(row)

mergedLocs = []
prevType = towerLocs[0][2] # Merge them together 
toAppend = []
# print(towerLocs)
for row in towerLocs: # first, want to run through and calculate heights for each monorail segment.
    rowLat = float(row[0])
    rowLong = float(row[1])
    rowType = row[2]

    if prevType != rowType:
        mergedLocs.append(toAppend)
        toAppend = []
    else:
        toAppend.append(row)

    prevType = rowType

finalLocs = []

mergedLocs.pop(-1) # i can't resolve this right now sorry

# now, re-read through it and add distances. For curve (1), we need to see if we can get the thing working.
for seg in mergedLocs:
    # print(seg)
    segTemp = []
    if seg[0][2] == '0': # if it's tram
        for item in seg:
            rowLat = float(item[0])
            rowLong = float(item[1])
            rowType = item[2]
            segTemp.append([rowLat, rowLong, 0, 9])

        
        segTemp[0][3] = 8 # first and last must be eight for tram
        segTemp[-1][3] = 8

        for row in segTemp:
            finalLocs.append(row)
        
    else: # then it's a monorail
        firstLat = float(seg[0][0]) # this code was unsuccessful. still theoretically usable though
        firstLong = float(seg[0][1])
        endLat = float(seg[-1][0])
        endLong = float(seg[-1][1])
        distBetween = havDist(np.deg2rad(firstLat), np.deg2rad(endLat), np.deg2rad(firstLong), np.deg2rad(endLong), radius)
        
        elevStart = elevAtPoint(firstLat, firstLong)
        elevEnd = elevAtPoint(endLat, endLong)
        deltaElev = elevEnd-elevStart
        
        elevAngle = np.arctan(deltaElev/distBetween)
        deltaHs = []

        # print(elevStart - elevEnd, distBetween, len(seg),np.rad2deg(elevAngle))
        
        for tower in seg:
            lat = float(tower[0])
            long = float(tower[1])  
            elevation = elevAtPoint(lat, long)
            dist2Start = havDist(np.deg2rad(firstLat), np.deg2rad(lat), np.deg2rad(firstLong), np.deg2rad(long), radius)
            variableA = dist2Start*np.tan(elevAngle) # i'm out of name ideas
            deltaH = elevStart + variableA - elevation
            #print(deltaH, elevation)
            segTemp.append([rowLat, rowLong, 1, 5])

        segTemp[0][3] = 6 # first and last must be six for monorail
        segTemp[-1][3] = 6

        for row in segTemp:
            finalLocs.append(row)

data = {'latitude' : [],
        'longitude' : [],
        'type' : [],
        'elev' : []}           
for tower in finalLocs:
    #print(tower)
    data['latitude'].append(tower[0])
    data['longitude'].append(tower[1])
    data['type'].append(tower[2])
    data['elev'].append(tower[3])


df = pd.DataFrame(data)
df.to_csv('tower_locations_updated.csv',index=False, header=False)
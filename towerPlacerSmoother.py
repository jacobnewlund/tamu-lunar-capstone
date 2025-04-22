import numpy as np
import pandas as pd
import csv

# read in segment creator input.
# this consists of a series of segments with lat-longs for each segment of the segment. 
# you can kind of make a function of the segment's location over distance. 
# Plot a tower at the start and end and then see how many towers would be required to connect them. 
# and then repeat a bunch of times.

towerLocs = [] # list of tower locations, items are a lat-long
estTowers = 1 # estimated tower amount. starts at one since we don't include it in the calculations

# first, we break apart our input
with open('path_segments.csv', mode='r') as segFile,  open('A_star_Path.csv', mode='r') as pathFile:
    segReader = csv.reader(segFile, delimiter=',')
    pathReader = csv.reader(pathFile, delimiter=',')
    next(pathReader, None) # get rid of headers.
    next(segReader, None) # ""

    pathRows = [] # want to make it into a list so we can rifle through the indices
    for row in pathReader: 
        rowTemp = []
        for item in row:
            rowTemp.append(item)
        pathRows.append(rowTemp[1:]) # get rid of those yucky indices
        # print(rowTemp[1:])

    segRows = [] # want to make it into a list so we can... pass it into a function, actually
    for row in segReader:
        rowTemp = []
        for item in row:
            rowTemp.append(item)
        segRows.append(rowTemp)
        # print(rowTemp)
       

def segmentPlacer(lats, longs, dists, dirs, segType): # generates tower placements per segment spaced evenly
    segDist = sum(dists) # get total length of segment
    towersAdded = 0 # see how many towers we added. For reasons.

    if segType == "start": # if it's the start position (so, no action), make sure to plot the first tower. otherwise only plot mid until last tower
        towerLocs.append([[lats[0], longs[0]], "start"])
        print("First tower created.")
        return # ok cool let's keep going
    if segType == "end":
        typeMaxDist = 100 # ALEX OVERRIDE
        print("Last segment.") # FYI

    # now we need to create the distance function
    totalDist = np.sum(dists)

    if segType == "straight" or segType == "end":
        typeMaxDist = 100
        segType = 0
    else: # then it's a curve
        typeMaxDist = 25
        segType = 1

    dStep = 0.1 # distance step, more step means more worse data output. less step means the opposite

    reqTowers = np.ceil(totalDist/typeMaxDist)
    global estTowers
    estTowers = estTowers + reqTowers
    print("Type {}, total distance is {:.2f} m, Towers to add: {}".format(segType, totalDist, reqTowers))
    offsetDistance = (totalDist % typeMaxDist)
    print("Offset of", offsetDistance)

    # Set up lat-long functions as a function of distance stepped
    maxIndex = len(lats) # range of indices... max index is ya know, the max index
    
    latLongFunctions = [[[0, 0], 0]] # list of lat-long functions of dstep. first one is lat function, then long function, second one is distance before applicable. has to be initialized with that garbage list

    for i in range(maxIndex-1): # create functions for each sub-segment of the segment. there's maxIndex - 1 functions per entire segment. It looks forward.
        toAppend = []
        lat1 = lats[i]
        long1 = longs[i]
        lat2 = lats[i+1] 
        long2 = longs[i+1] 
        toAppend.append(latLongFunction(lat1, lat2, long1, long2, radius, dStep)) # add on lat and long functions
        toAppend.append(havDist(np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(long1), np.deg2rad(long2), radius) + latLongFunctions[-1][1]) # add on distance val

        latLongFunctions.append(toAppend) # add to list

    latLongFunctions = latLongFunctions[1:] # Trim off that yucky first list


    # Start going along the path
    lastTowerLatLongs = towerLocs[-1] # Initialize our loop. We want to keep going along path and place down towers as we go. So... last tower of last segment is our starting one. 
    
    lastLat = lastTowerLatLongs[0][0]
    lastLong = lastTowerLatLongs[0][1]
    currentLat = lastLat # because... we're starting there???
    currentLong = lastLong

    currentFunctionIndex = 0 # start on the first function
    currentFunctionDistance = latLongFunctions[currentFunctionIndex][1]

    x = - offsetDistance # iterable creep along path. it increases by one dStep. who cares if it takes forever. has to be initialized with offset distance for reasons
    while x < segDist - offsetDistance: # while we're in the segment
        if x < 0: # if we're still in offset it ain't even worth the computational expense
            x += dStep
            continue            
        
        if x > currentFunctionDistance: # if we're out of the bounds of our current function, go to the next one.
            currentFunctionIndex += 1
        if havDist(np.deg2rad(lastLat), np.deg2rad(currentLat), np.deg2rad(lastLong), np.deg2rad(currentLong), radius) > typeMaxDist: # if we're further than the max distance from a tower allowed by our type, place one down and reset out lastLat/Longs...
            towerLocs.append([[currentLat, currentLong], segType])
            towersAdded += 1
            lastLat = currentLat
            lastLong = currentLong

        currentLatFunction = latLongFunctions[currentFunctionIndex][0][0]
        currentLongFunction = latLongFunctions[currentFunctionIndex][0][1]
        currentFunctionDistance = latLongFunctions[currentFunctionIndex][1]
        # calc currentLat, currentLong along path
        currentLat += currentLatFunction*(dStep) # step along using our current lat/long function of dStep
        currentLong += currentLongFunction*(dStep)
        x += dStep
    towersAdded += 1 # Now that we're out, add the required suboptimal tower placement to make the transition happen. Assumes we don't have any non-diagonal one segment segments, which is currently correct.
    # and now we're out. Stamp down final tower at final lat-long values and get onto the next one.
    if segType == "end":
        towerLocs[-1] = [towerLocs[-1][0], 1] # retroactively change the last transition to a curve since end segment is technically a curve. don't like it? Me neither.
        towerLocs.append([[currentLat, currentLong], 1]) 
        towerLocs.append([[lats[-1], longs[-1]], "end"])
        towersAdded += 1
    else:
        towerLocs.append([[currentLat, currentLong], segType]) 
        towerLocs.append([[lats[-1], longs[-1]], 0])
        towersAdded += 1
    print("Added", towersAdded, "tower(s) this segment.")


def havDist(lat1, lat2, long1, long2, radius): # gets dist between two lat-longs. IT NEEDS RADIANS AS INPUT. OTHERWISE YOU WON'T LIKE WHAT YOU GET OUTTA IT
    return 2*radius*np.arcsin(np.sqrt((1 - np.cos(lat2-lat1) + np.cos(lat1)*np.cos(lat2)*(1 - np.cos(long2-long1)))/2))

def hav(theta):
    return (1-np.cos(theta))/2

def archav(angle):
    return 2*np.arcsin(np.sqrt(angle))

def latLongFunction(latStart, latEnd, longStart, longEnd, radius, dStep): # returns "slope" of latitude and longitude as a function of dstep. It takes DEGREE INPUT. there are no inconsistencies in this program.
    # I originally did this a truly deranged way using the haversine function and a bunch of trig but the juice wasn't worth the squeeze. this way adds uncertainty but I don't really care since we reset our instability each segment to zero by forcing the last tower placed in a segment. another dub for approximation gang
    dist = havDist(np.deg2rad(latStart), np.deg2rad(latEnd), np.deg2rad(longStart), np.deg2rad(longEnd), radius) # distance from point a to point b. dStep is the distance step along this.
    stepsReq = dist/dStep # gets how many dSteps it'll take to reach the end. this is a constant. 

    latPerStep = (latEnd-latStart)/stepsReq # in units of degrees/step. We also have distance/step. It's called dStep. boy, how could we possibly convert it to degrees/distance???
    longPerStep = (longEnd-longStart)/stepsReq

    latPerDistance = latPerStep/dStep # like this
    longPerDistance = longPerStep/dStep
    return [latPerDistance, longPerDistance]

# constants
radius = 1737400.0

# now we want to run each segment, passing the latitudes and longitudes of all points in the segment as well as distances between each lat-long to the function.
# this allows us to get the distances between each segment. and therefore... the length. and therefore... the cost. and therefore... the deliverable for this capstone.
for segment in segRows:
    lats = []
    longs = []
    segDirs = []
    segDists = []
    segType = segment[0]
    startIndex = int(segment[1])
    endIndex = int(segment[2]) # this variable is kind of fun to say
    segDist = float(segment[3])

    # if endIndex - startIndex == 1: # if it's a single-segment segment... yes, this is an edge case we got to catch <- no it isn't moron
    #     pathSnippet = pathRows[startIndex:endIndex+1]
        # print(pathSnippet)
        # lats.append(float(pathSnippet[0][0]))
        # longs.append(float(pathSnippet[0][1]))
        # lats.append(float(pathSnippet[1][0]))
        # longs.append(float(pathSnippet[1][1]))
        # pathDirs.append(pathSnippet[0][4])
        # if segType=="curve":
        #     print(lats, longs)
        #     towersRequired = np.ceil(segDist / 25) # We're using elevmonorail
        #     towerLocs.append(lats[0]) # Append the start one since we need it
        # else: #so, it's straight (this should never happen)
        #     print("uhh")

    # else: # if it's anything else...
    # now we have to get the lats and longs from the path stuff
    pathSnippet = pathRows[startIndex:endIndex+1] # specifying range...
    # print(pathSnippet, startIndex, endIndex)
    lats.append(float(pathSnippet[0][0]))
    longs.append(float(pathSnippet[0][1]))
    segDirs.append(pathSnippet[0][4])
    for row in pathSnippet[1:]:
        lat = float(row[0])
        long = float(row[1])
        segDists.append(havDist(np.deg2rad(lat), np.deg2rad(lats[-1]), np.deg2rad(long), np.deg2rad(longs[-1]), radius))
        lats.append(lat)
        longs.append(long)
        segDirs.append(row[4])

    segmentPlacer(lats, longs, segDists, segDirs, segType) # it does the work for us...

# print(towerLocs) # test print

# Assassinate start and make it a tram
towerLocs[0][1] = "0"

# Now let's csv. it all up
data = {'latitude' : [],
        'longitude' : [],
        'type' : []}           
for tower in towerLocs:
    data['latitude'].append(tower[0][0])
    data['longitude'].append(tower[0][1])
    data['type'].append(tower[1])

df = pd.DataFrame(data)
df.to_csv('tower_locations_smoother.csv',index=False, header=False)

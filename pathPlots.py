import numpy as np
import csv
import matplotlib.pyplot as plt

def havDist(lat1, lat2, long1, long2, radius):
    return 2*radius*np.arcsin(np.sqrt((1 - np.cos(lat2-lat1) + np.cos(lat1)*np.cos(lat2)*(1 - np.cos(long2-long1)))/2))

distanceAxis = [0.0]
grades = []
elevations = []

radius = 1737.4000

with open('A_star_Path.csv', mode='r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader, None) # get rid of header
    firstRow = next(reader)
    prevLat = float(firstRow[1])
    prevLong = float(firstRow[2])
    elevations.append(float(firstRow[3])/1000)
    grades.append(float(firstRow[4]))
    for row in reader:
        lat = float(row[1])
        long = float(row[2])
        elevation = float(row[3])
        grade = float(row[4])
        distBetween = havDist(np.deg2rad(prevLat), np.deg2rad(lat), np.deg2rad(prevLong), np.deg2rad(long), radius)

        distanceAxis.append(distanceAxis[-1] + distBetween)
        elevations.append(elevation/1000)
        grades.append(grade)
        prevLat = lat
        prevLong = long

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', 'box')
plt.plot(distanceAxis, elevations, color ='magenta')
plt.title("Elevation over Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Elevation (km)")
plt.figaspect(1)
# ax.legend(bbox_to_anchor=(0, 1), loc='upper left')
# ax.set_title("Weight Map with Path")
plt.savefig("ElevationMap.jpg", format='jpeg', dpi=1200)

plt.figure()
plt.plot(distanceAxis, grades)
plt.title("Grade over Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Grade (deg)")

plt.savefig("GradeMap.jpg", format='jpeg', dpi=1200)


print("Final path length using really bad straight-lines between points:", distanceAxis[-1])
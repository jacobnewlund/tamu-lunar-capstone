import pandas as pd
import numpy as np
import csv
class Segment:
    def __init__(self, type, start_ind, end_ind, length):
        self.type = type
        self.start = start_ind
        self.end = end_ind
        self.length = length #length in meters

    ''' attribute explanation:
        type            -> [straight, turn, end]
        mode            -> [-1 = neither, 0 = tramway, 1 = monorail]
        length          -> [in meters]'''
    
    def __str__(self):
        return f"type: {self.type}, start_ind: {self.start}, end_ind {self.end}, length: {self.length}"


segments = []

def havDist(lat1, lat2, long1, long2, radius):
    return 2*radius*np.arcsin(np.sqrt((1 - np.cos(lat2-lat1) + np.cos(lat1)*np.cos(lat2)*(1 - np.cos(long2-long1)))/2))


def segmentation(pathfile):
    #pathfile needs to be formatted as [index, lat, long, -, grade, action]
    df = pd.read_csv(pathfile)
    path_points = df.to_numpy()
    path = {'latitude' : [],
            'longitude': [],
            'action'   : [],
            'mode'     : []}
    
    for point in path_points:
        path['latitude'].append(point[1])
        path['longitude'].append(point[2])

        #ugly if list to make this readable in a print
        if point[5] == '(-1, 0)':
            path['action'].append('S')
        elif point[5] == '(0, 1)':
            path['action'].append('E')
        elif point[5] == '(-1, 0)':
            path['action'].append('N')
        elif point[5] == '(0, -1)':
            path['action'].append('W')
        elif point[5] == '(-1, 1)':
            path['action'].append('SE')
        elif point[5] == '(-1, -1)':
            path['action'].append('SW')
        elif point[5] == '(1, 1)':
            path['action'].append('NE')
        elif point[5] == '(1, -1)':
            path['action'].append('NW')
        else:
            path['action'].append('start')

    step_direction = path['action'][1] #store initial direction

    dist = 0
    step = 0
    type = 'start'
    while step < len(path['latitude']) - 1:
        start = step
        print(round(step/len(path['longitude'])*100,2),'% through gathering distances')
        if step + 3 < len(path['latitude']) - 1:
            while path['action'][step] == step_direction:
                dist += havDist(np.deg2rad(path['latitude'][step]),np.deg2rad(path['latitude'][step + 1]),np.deg2rad(path['longitude'][step]),np.deg2rad(path['longitude'][step + 1]),1737400.0)
                step += 1
                type = 'straight'
                #print('straight')
                if step + 3 == len(path['latitude']) - 1:
                    break
            segments.append(Segment(type=type, start_ind=start, end_ind=step, length = dist))
            start = step
            dist = 0
            while path['action'][step] != step_direction or path['action'][step + 1] != step_direction or path['action'][step + 2] != step_direction or path['action'][step + 3] != step_direction :
                dist += havDist(np.deg2rad(path['latitude'][step]),np.deg2rad(path['latitude'][step + 1]),np.deg2rad(path['longitude'][step]),np.deg2rad(path['longitude'][step + 1]),1737400.0)
                step += 1
                step_direction = path['action'][step]
                type = 'curve'
                if step + 3 == len(path['latitude']) - 1:
                    break

            segments.append(Segment(type=type, start_ind=start, end_ind=step, length = dist))
            dist = 0
            step_direction = path['action'][step]
            
        else:
            type = 'end'
            dist += havDist(np.deg2rad(path['latitude'][step]),np.deg2rad(path['latitude'][step + 1]),np.deg2rad(path['longitude'][step]),np.deg2rad(path['longitude'][step + 1]),1737400.0)
            step += 1
    segments.append(Segment(type=type, start_ind=start, end_ind=step, length=dist))
        

    print(100,'% through gathering distances')

    data = {'type' : [],
            'start' : [],
            'end' : [],
            'length' : []}                
    for seg in segments:
        data['type'].append(seg.type)
        data['start'].append(seg.start)
        data['end'].append(seg.end)
        data['length'].append(seg.length)

    return data

def mode_optimization(data):
    df = pd.DataFrame(data)
    df.to_csv('path_segments.csv',index=False)
    # this is genuinely aborrent but I don't know how to work with pandas and honestly I don't want to learn it
    newCSV = []
    
    fileRows = []    
    with open('path_segments.csv', mode='r') as file:
        fileReader = csv.reader(file, delimiter=',')
        next(fileReader, None) # get rid of header
        for row in fileReader:
            fileRows.append(row)

    fileRows.pop(1) # assassinate the one segment segment at beginning
    fileRows[1][1] = 0
    fileRows[-1][0] = "straight" # Alex Override

    # now re-read and merge together segments if possible using the Carl Optimization.
    for row in fileRows:
        if row[0] == "straight" and float(row[3]) < 1000.0: # implement CO
            row[0] = "curve"
    # Now we merge together same adjacent segment (and sum distances)
    lastRow = fileRows[0]
    newCSV.append(lastRow)
    runningDistance = 0
    runningIndex = 0
    mergeMode = False # false means we're not merging.


    for row in fileRows[1:]: # run along path and attempt to merge together adjacent paths. this is semi-hardcoded and poorly designed
        lastType = lastRow[0]
        lastStartIndex = lastRow[1]
        lastEndIndex = lastRow[2]
        lastDist = float(lastRow[3])
        
        rowType = row[0]
        rowStartIndex = row[1]
        rowEndIndex = row[2]
        rowDist = float(row[3])

        

        if lastType != rowType and mergeMode: # end of merge mode. append to list
            runningDistance += lastDist
            newCSV.append([lastType, runningIndex, rowStartIndex, runningDistance])
            mergeMode = False
            runningDistance = 0
            runningIndex = 0
        elif lastType == rowType: # then we want to merge. Hold off on appending to row
            mergeMode = True # we're now merging. Until lastType != rowType
            runningIndex = newCSV[-1][2]
            runningDistance += lastDist
        else: # if we aren't merging, just append
            newCSV.append(lastRow)
        lastRow = row
    
    newCSV.pop(0) # how'd that get in there (also, rare pop() value)
    # for row in newCSV:
    #     print(row)

    data = {'type' : [],
            'start' : [],
            'end' : [],
            'length' : []}                
    for row in newCSV:
        data['type'].append(row[0])
        data['start'].append(row[1])
        data['end'].append(row[2])
        data['length'].append(row[3])

    df = pd.DataFrame(data)
    df.to_csv('path_segments.csv',index=False)

data = segmentation('A_star_Path.csv')
mode_optimization(data)
print("Done creating .csv")
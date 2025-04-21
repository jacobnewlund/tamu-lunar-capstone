import pandas as pd
import numpy as np

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


def mode_optimization(pathfile):
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

    df = pd.DataFrame(data)
    df.to_csv('path_segments.csv',index=False)

mode_optimization('A_star_Path.csv')
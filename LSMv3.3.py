import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import heapq
from collections import deque
import time
import csv
import pandas as pd


class GridCell():
    def __init__(self, lat, long, grade, elevation, illumination, traversable, mode, goal, cost):
        self.lat = lat
        self.long = long
        self.illumination = illumination
        self.grade = grade
        self.elevation = elevation
        self.traversable = traversable
        self.goal = goal 
        self.mode = mode 
        self.cost = cost #keeps track of transportation that can be implemented

    ''' attribute explanation:
        grade           -> [float]
        elevation       -> [kept track of for backtracing]
        illumination    -> [0 = PSR, 1 = ILLUMINATED]
        traversable     -> [Bool True/False]
        goal            -> [keeps track of goal point, True/False]
        mode            -> [transportation mode: 0 = any, 1 = ONLY TRAMWAY]'''
    

    def __str__(self):
        return f"lat: {self.lat}, long: {self.long}, Illumination {self.illumination}, Grade: {self.grade}, Elevation {self.elevation}, Traversable: {self.traversable}, Mode: {self.mode}, Goal: {self.goal}, Cost: {self.cost}"
    
class State:
    # Just two attributes that represent position on grid (x, y for indexing purposes)
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"transportState({self.x}, {self.y})"

    def __eq__(self, other):
        # Check if two transportState objects are equal based on their coordinates
        if isinstance(other, State):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        # Allow State objects to be used in sets and as dictionary keys
        return hash((self.x, self.y))

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def path(self): # path from to root node to this node
        node, path_back = self, []
        while node:
            path_back.append((node.state, node.action))
            node = node.parent
        return list(reversed(path_back))
    
class Problem:
    def __init__(self, initial_state, goal_test, grid, transition_function):
        self.initial_state = initial_state
        self.goal_test = goal_test 
        self.grid = grid # this is the grid world
        self.transition_function = transition_function 
        
    def is_goal(self, state):# binary IS-GOAL(s) function from R&N
        return goal_test(state, self.grid)
    
    def transition(self, state, action):#This is the transition model, i.e., the s'=RESULT(s,a) function from R&N. It also returns the cost c(s,s',a)
        return self.transition_function(state, action, self.grid)
    
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
    
### FUNCTIONS ###

def goal_test(state,grid): # IS-GOAL function in R&N
        x, y = state.x, state.y
        return grid[x, y].goal

#DICTIONARY OF AVAILABLE ACTIONS
step = 1
Action ={"UP":(-step, 0),"DOWN":(step, 0),"LEFT":(0, -step),"RIGHT":(0, step),"UP-LEFT":(-step,-step),"UP-RIGHT":(-step,step),"DOWN-LEFT":(step,-step),"DOWN-RIGHT":(step,step)} # Logic of actions seems reversed but it's like this to facilitate plotting.

def available_actions(state, grid):
    actions = []
    x, y = state.x, state.y
    maxInd_X = np.shape(grid)[0] - 1
    maxInd_Y = np.shape(grid)[1] - 1

    ### ADJUSTABLE VARIABLES ##
    step = 1
    
    #path can only move within bounds and cells that have "reasonable danger"
    if x > 0 and grid[x-step][y].traversable:
        actions.append(Action["UP"])
    if x < maxInd_X and grid[x+step][y].traversable:
        actions.append(Action['DOWN'])
    if y > 0 and grid[x][y-step].traversable:
        actions.append(Action["LEFT"])
    if y < maxInd_Y and grid[x][y+step].traversable:
        actions.append(Action["RIGHT"])
    if x > 0 and y > 0 and grid[x-step][y-step].traversable:
        actions.append(Action["UP-LEFT"])
    if x > 0 and y < maxInd_Y and grid[x-step][y+step].traversable:
        actions.append(Action["UP-RIGHT"])
    if x < maxInd_X and y > 0 and grid[x+step][y-step].traversable:
        actions.append(Action["DOWN-LEFT"])
    if x < maxInd_X and y < maxInd_Y and grid[x+step][y+step].traversable:
        actions.append(Action["DOWN-RIGHT"])
    
    return actions 

def transition(state, action, grid):
    #BOUNDING BOX:
    maxInd_X = np.shape(grid)[0]
    maxInd_Y = np.shape(grid)[1]
    #grade = grid[state.x][state.y].grade
    cost = grid[state.x,state.y].cost 
    
    new_x = state.x + action[0]
    new_y = state.y + action[1]

    if 0 <= new_x < maxInd_X and 0 <= new_y < maxInd_Y:
        return State(new_x, new_y), cost
    return state, cost  

class AStarSearch:
    def __init__(self, problem, available_actions):
        self.problem = problem
        self.available_actions = available_actions
        self.heuristic = manhattan_heuristic  

    def search(self):
        frontier = [(0, id(Node(self.problem.initial_state)), Node(self.problem.initial_state))] 
        reached = {self.problem.initial_state: 0}  
        while frontier:
            current_f, _, node = heapq.heappop(frontier)  

            if self.problem.is_goal(node.state):
                #return node.path()  
                return node

            for child in self.expand(node):
                g_cost = node.path_cost + child.path_cost  # Actual path cost g(n)
                h_cost = self.heuristic(child.state, self.problem)  # Heuristic cost
                f_cost = g_cost + h_cost  # Total cost f(n)

                if child.state not in reached or g_cost < reached[child.state]:
                    reached[child.state] = g_cost
                    heapq.heappush(frontier, (f_cost, id(child), child))  # Push with updated f(n)

        return None  
    
    def expand(self, node):
        """Generate child nodes for the given node."""
        s = node.state
        for action in self.available_actions(s, self.problem.grid):  
            s_prime, cost_action = self.problem.transition(s, action)  
            yield Node(state=s_prime, parent=node, action=action, path_cost=node.path_cost + cost_action)

def manhattan_heuristic(state, problem):
    return abs(state.x - goalX) + abs(state.y - goalY)

def havDist(lat1, lat2, long1, long2, radius):
    return 2*radius*np.arcsin(np.sqrt((1 - np.cos(lat2-lat1) + np.cos(lat1)*np.cos(lat2)*(1 - np.cos(long2-long1)))/2))


pointA = [-85.292466, 36.920242] #start point
pointB = [-84.7906, 29.1957]     #end point
#boundingBox
minLat, maxLat, minLong, maxLong = -85.5, -84, 28, 38


#IMPORT BOUND BOX DATA AND GENERATE GRID
lat_points = 1820 
lon_points = 1108
radius = 1737.4000
minHavStart = 1000.0 # intialize value
minHavEnd = 1000.0
grid = np.empty((lat_points,lon_points), dtype=object)
# Read CSV file
print("Reading Data")
step = 0

# import os
# script_dir = os.path.dirname(__file__)  # folder where the script is
# file_path = os.path.join(script_dir, 'megathingy.csv')

with open('megathingy.csv', mode='r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader, None)
    i = 0
    j = 0
    for row in reader:
        lat = float(row[0])
        long = float(row[1])

        #decrease bounding box:
        # if round(long,2) < (round(pointB[1],2) - 1): 
        #     continue

        #initialize variables
        ill = float(row[2])
        grade = float(row[3])
        elev = float(row[4])
        goal = False
        mode = 0
        cost = 0

        #Store initial state
        if round(lat,2) == round(pointA[0],2) and round(long,2) == round(pointA[1],2):
            if havDist(np.deg2rad(pointA[0]), np.deg2rad(lat), np.deg2rad(pointA[1]), np.deg2rad(long), radius) < minHavStart:
                minHavStart = havDist(np.deg2rad(pointA[0]), np.deg2rad(lat), np.deg2rad(pointA[1]), np.deg2rad(long), radius)
                initX = i
                initY = j
                #print(lat,long, "dist", havDist(pointA[0], lat, pointA[1], long, radius)) 

        #set goal state using science
        if round(lat,2) == round(pointB[0],2) and round(long,2) == round(pointB[1],2):
            if havDist(np.deg2rad(pointB[0]), np.deg2rad(lat), np.deg2rad(pointB[1]), np.deg2rad(long), radius) < minHavEnd:
                minHavEnd = havDist(np.deg2rad(pointB[0]), np.deg2rad(lat), np.deg2rad(pointB[1]), np.deg2rad(long), radius)
                goalX = i
                goalY = j
                # print(lat,long, "dist", havDist(pointB[0], lat, pointB[1], long, radius)) 

        #Danger Equation Implemented
        if grade >= 20 or ill == 0:
            traversable = False
        
        else:
            traversable = True
            cost = np.exp(grade/20)
        

        #Transportation modes available (ADJUST NUMBER !!!)
        if grade > 10:
            mode = 2 #only tramway possible


        #INITIALIZE GRIDCELL
        
        grid[i,j] = GridCell(lat=lat, long=long, illumination=ill, grade=grade, elevation=elev, traversable=traversable, goal = goal, mode=mode, cost=cost)
        #print(grid[i,j])

        #index
        j += 1

        if j == lon_points:
            j = 0
            i += 1

grid[goalX,goalY].goal = True

print("done reading !!!")

prob = Problem(State(initX,initY), goal_test, grid, transition)

def LLA_to_PCPF(lat, lon, alt, radius):
    r = radius + alt
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])


lat_minus = np.deg2rad(-85.5)
lat_plus = np.deg2rad(-84.0)
lon_minus = np.deg2rad(28.0)
lon_plus = np.deg2rad(38.0)

#MODE OPTIMIZATION
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

    segments = []
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
                if step + 3 == len(path['latitude']) - 1:
                    break
            segments.append(Segment(type=type, start_ind=start, end_ind=step, length = dist))
            start = step

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
            step += 1

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

    return segments

#VISUALIZATION
def Indices_to_Stereo(i_, j_, n_lat, n_lon):
    lat = lat_minus + i_ * (lat_plus - lat_minus) / n_lat
    lon = lon_minus + j_ * (lon_plus - lon_minus) / n_lon
    xyz = LLA_to_PCPF(lat, lon, 0.0, 1737400.0)
    return [xyz[0] / (1 - xyz[2]), xyz[1] / (1 - xyz[2])]

def visualize_problem(problem: Problem):
    grid = problem.grid
    n_X = np.shape(grid)[0]
    n_Y = np.shape(grid)[1]
    # We create a color map for the grid based on danger values
    color_grid_trev = np.zeros((n_X, n_Y, 3))
    color_grid_non = np.zeros((n_X, n_Y, 3))  # RGB color grid
    # color_grid_stereo = np.zeros((n, n, 3))

    x = np.linspace(0, n_X, n_X)
    y = np.linspace(0, n_Y, n_Y)

    # x := longitude, y := latitude
    X, Y = np.meshgrid(x, y)

    x_stereo = np.zeros((n_X, n_Y))
    y_stereo = np.zeros((n_X, n_Y))
    for i in range(n_X):
        for j in range(n_Y):
            # lat = lat_minus + i * (lat_plus - lat_minus) / n
            # lon = lon_minus + j * (lon_plus - lon_minus) / n
            # xyz = LLA_to_PCPF(lat, lon, 0.0, 1737400.0)
            xy_stereo = Indices_to_Stereo(i, j, n_X, n_Y)
            # print(xyz)
            x_stereo[i, j] = xy_stereo[0]
            y_stereo[i, j] = xy_stereo[1]

    # transform to south pole stereographic xy coordinates
    Z = np.zeros((n_X, n_Y))
    
    for i in range(n_X):
        for j in range(n_Y):
            cell = grid[i][j]
            #print(i, j)
            if cell.goal:  
                color_grid_trev[i, j] = [0, 1, 0]  # Green for goal 
                goalX = i
                goalY = j
            elif cell.traversable == False:
                color_grid_non[i,j] = [1,0,0]
            else:
                safety = 1 - np.exp(cell.grade/20)/np.e # Now, danger=1 means safe, danger=0 means dangerous
                color_grid_trev[i, j] = [1, safety, safety]  # Red fades to white as safety increases
                # PSR = cell.illumination
                # color_grid[i,j] = [1, PSR, PSR]
                Z[i, j] = safety
    
    init_state = problem.initial_state
    color_grid_trev[init_state.x,init_state.y] = [0,0,1]

    df = pd.read_csv('A_star_Path.csv')
    path_lats = df['latitude'].to_numpy()
    path_longs = df['longitude'].to_numpy()
    
    path_stero = np.zeros((len(path_longs),2))
    for i in range(len(path_lats)):
        pcpf = LLA_to_PCPF(np.deg2rad(path_lats[i]),np.deg2rad(path_longs[i]),0.0,1737400.0)
        stereo = [pcpf[0] / (1 - pcpf[2]), pcpf[1] / (1 - pcpf[2])]
        for j in range(2):
            path_stero[i][j] = stereo[j]

    #path segments
    segments = mode_optimization('A_star_Path.csv')
    tramX = []
    tramY = []
    monoX = []
    monoY = []
    for seg in segments:
        i = int(seg.start)
        j = int(seg.end)

        if seg.type == 'straight':
            tramX.append(path_stero[i:j,1])
            tramY.append(path_stero[i:j,0])
        elif seg.type == 'curve':
            monoX.append(path_stero[i:j,1])
            monoY.append(path_stero[i:j,0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmesh = ax.pcolormesh(y_stereo, x_stereo, Z, cmap='RdBu')
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    xy_init = Indices_to_Stereo(initX + 0.5, initY + 0.5, n_X, n_Y)
    xy_goal = Indices_to_Stereo(goalX + 0.5, goalY + 0.5, n_X, n_Y)
    plt.scatter(xy_init[1], xy_init[0], s=50, facecolors='none', edgecolors='lime', linewidths=2, label='Landing Site')
    plt.scatter(xy_goal[1], xy_goal[0], s=50, facecolors='none', edgecolors='gold', linewidths=2, label='IM-2 (ISRU)')
    #plt.plot(path_stero[:,1],path_stero[:,0],color ='magenta',label = 'Path')
    # plt.plot(tramX, tramY, color = 'magenta', label='Tramway')
    # plt.plot(monoX, monoY, color='green', label='MONORAIL')

    # Prevent duplicate labels
    plotted_labels = {'Tramway': False, 'MONORAIL': False}

    for x, y in zip(tramX, tramY):
        label = 'Tramway' if not plotted_labels['Tramway'] else None
        plt.plot(x, y, color='magenta', label=label)
        plotted_labels['Tramway'] = True

    for x, y in zip(monoX, monoY):
        label = 'MONORAIL' if not plotted_labels['MONORAIL'] else None
        plt.plot(x, y, color='green', label=label)
        plotted_labels['MONORAIL'] = True

    # Add colorbar
    cbar = fig.colorbar(cmesh, ax=ax)
    cbar.set_label("Cost")  # Set label for colorbar


    ax.legend(bbox_to_anchor=(0, 1), loc='upper left')
    ax.set_title("Cost Map with Path")
    plt.savefig("LSMpathMap.jpg", format='jpeg', dpi=1200)


ast = AStarSearch(prob, available_actions)
print('Running A-star search algorithm')
start_time = time.time()

solution = ast.search()

path = {
    'latitude' : [],
    'longitude' : [],
    'elevation' : [],
    'grade' : [],
    'action' : []
}

if solution:
    print("Solution found:")
    # print(solution.path())
    for item in solution.path():
        state = item[0]
        path["latitude"].append(prob.grid[state.x][state.y].lat)
        path["longitude"].append(prob.grid[state.x][state.y].long)
        path["elevation"].append(prob.grid[state.x][state.y].elevation)
        path["grade"].append(prob.grid[state.x][state.y].grade)
        path['action'].append(item[1])
        
    print("solution cost=", solution.path_cost) #prints coordinate of solution
else:
    print("No solution found.")

end_time = time.time()
running = end_time - start_time
print('runtime', running)

df = pd.DataFrame(path)
df.to_csv('A_star_Path.csv',index=True) 

visualize_problem(prob)

def operations(path):
    grid_size = 100 #[m]
    tot_cost = 0
    tot_power = 0
    tot_mass = 0

    #variables
    rail_mass = 0
    monoMass = 33 #kg/m of rail
    tramMass = 2.75 #kg/m rail
    railCost = 3.50 # $/m
    #assuming 6m towers
    towerCount = 1 #initialize with 1
    towerMass = 1050
    towerCost = 3 # $/m
 
    for row in df:
        mode = row[2]

        if mode == 0: #Monorail
            rail_mass += (grid_size*monoMass)
            towerCount += 3
        elif mode == 1: #Tramway
            rail_mass.append(grid_size*tramMass)
            towerCount += 1

    #material cost
    mat_cost = rail_mass*railCost + towerCount*towerMass*towerCost

    #Power

    #Deployment
    capacity = 17*907.185 #kg/launch
    launches = np.ceil(towerCount*towerMass/capacity + rail_mass/capacity)
    
    return mat_cost, launches

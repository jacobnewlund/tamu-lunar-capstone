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
    def __init__(self, lat, long, grade, elevation, illumination, traversable, mode, goal):
        self.lat = lat
        self.long = long
        self.illumination = illumination
        self.grade = grade
        self.elevation = elevation
        self.traversable = traversable
        self.goal = goal 
        self.mode = mode 
        self.cost = -np.log(1-grade/20) #keeps track of transportation that can be implemented

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
            path_back.append(node.state)
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

def goal_test(state,grid): # IS-GOAL function in R&N
        x, y = state.x, state.y
        return grid[x, y].goal

#DICTIONARY OF AVAILABLE ACTIONS
step = 1
Action ={"UP":(-step, 0),"DOWN":(step, 0),"LEFT":(0, -step),"RIGHT":(0, step)} # Logic of actions seems reversed but it's like this to facilitate plotting.

def available_actions(state, grid):
    actions = []
    x, y = state.x, state.y
    maxInd = len(grid)

    ### ADJUSTABLE VARIABLES ##
    step = 1
    
    #path can only move within bounds and cells that have "reasonable danger"
    if x > 0 and grid[x-step][y].traversable:
        actions.append(Action["UP"])
    if x < maxInd and grid[x+step][y].traversable:
        actions.append(Action['DOWN'])
    if y > 0 and grid[x][y-step].traversable:
        actions.append(Action["LEFT"])
    if y < maxInd and grid[x][y+step].traversable:
        actions.append(Action["RIGHT"])
    
    return actions 

def transition(state, action, grid):
    #BOUNDING BOX:
    maxInd = len(grid) - 1
    #grade = grid[state.x][state.y].grade
    cost = grid.cost 
    
    new_x = state.x + action[0]
    new_y = state.y + action[1]

    if 0 <= new_x < maxInd and 0 <= new_y < maxInd:
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
    goal_positions = [(i, j) for i in range(len(problem.grid)) for j in range(len(problem.grid[0]))
                    if problem.grid[i, j].goal]
    return min(abs(state.x - gx) + abs(state.y - gy) for gx, gy in goal_positions)

pointA = [-85.292466, 36.920242] #start point
pointB = [-84.7906, 29.1957]     #end point
#boundingBox
minLat, maxLat, minLong, maxLong = -85.5, -84, 28, 38

#IMPORT BOUND BOX DATA AND GENERATE GRID
yippee = 500
grid = np.empty((yippee,yippee), dtype=object)
# Read CSV file
with open('terrain_data.csv', mode='r') as file:
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

        #Store initial state
        if round(lat,2) == round(pointA[0],2) and round(long,2) == round(pointA[1],2):
            initX = i
            initY = j

        #set goal state using science
        if round(lat,2) == round(pointB[0],2) and round(long,2) == round(pointB[1],2):
            goal = True

        #Danger Equation Implemented
        if grade >= 20 or ill == 0:
            traversable = False
        
        else:
            traversable = True
        

        #Transportation modes available (ADJUST NUMBER !!!)
        if grade > 10:
            mode = 2 #only tramway possible


        #INITIALIZE GRIDCELL
        grid[i,j] = GridCell(lat=lat, long=long, illumination=ill, grade=grade, elevation=elev, traversable=traversable, goal = goal, mode=mode)
        #print(grid[i,j])

        #index
        j += 1
        if j == yippee:
            j = 0
            i += 1

            if i == 250:
                print('progress !!')

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



#VISUALIZATION
def Indices_to_Stereo(i_, j_, n):
    lat = lat_minus + i_ * (lat_plus - lat_minus) / n
    lon = lon_minus + j_ * (lon_plus - lon_minus) / n
    xyz = LLA_to_PCPF(lat, lon, 0.0, 1737400.0)
    return [xyz[0] / (1 - xyz[2]), xyz[1] / (1 - xyz[2])]

def visualize_problem(problem: Problem):
    grid = problem.grid
    n = len(grid) 
    # We create a color map for the grid based on danger values
    color_grid_trev = np.zeros((n, n, 3))
    color_grid_non = np.zeros((n, n, 3))  # RGB color grid
    # color_grid_stereo = np.zeros((n, n, 3))

    x = np.linspace(0, n, n)
    y = np.linspace(0, n, n)

    # x := longitude, y := latitude
    X, Y = np.meshgrid(x, y)

    x_stereo = np.zeros((n, n))
    y_stereo = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # lat = lat_minus + i * (lat_plus - lat_minus) / n
            # lon = lon_minus + j * (lon_plus - lon_minus) / n
            # xyz = LLA_to_PCPF(lat, lon, 0.0, 1737400.0)
            xy_stereo = Indices_to_Stereo(i, j, n)
            # print(xyz)
            x_stereo[i, j] = xy_stereo[0]
            y_stereo[i, j] = xy_stereo[1]

    # transform to south pole stereographic xy coordinates

    Z = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            cell = grid[i][j]
            #print(i, j)
            if cell.goal:  
                color_grid_trev[i, j] = [0, 1, 0]  # Green for goal 
                goalX = i
                goalY = j
            elif cell.traversable == False:
                color_grid_non[i,j] = [1,0,0]
            else:
                safety = 1 - cell.grade/20  # Now, danger=1 means safe, danger=0 means dangerous
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



    # Plot the grid
    # plt.imshow(color_grid, extent=[0, n, 0, n], origin='lower')
    # plt.pcolormesh(X, Y, Z, cmap='RdBu')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmesh = ax.pcolormesh(y_stereo, x_stereo, Z, cmap='RdBu')
    ax.set_aspect('equal', 'box')

    xy_init = Indices_to_Stereo(initX + 0.5, initY + 0.5, n)
    xy_goal = Indices_to_Stereo(goalX + 0.5, goalY + 0.5, n)

    plt.plot(xy_init[1], xy_init[0], 'y*', markersize=20, markeredgecolor='black', label='Start Point')
    plt.plot(xy_goal[1], xy_goal[0], 'g*', markersize=20, markeredgecolor='black', label='End Point')
    plt.plot(path_stero[:,1],path_stero[:,0], 'k--',label = 'path')

    # Add colorbar
    cbar = fig.colorbar(cmesh, ax=ax)
    cbar.set_label("Traversability")  # Set label for colorbar

    ax.grid(True)
    ax.legend()
    ax.set_title("Weight Map")
    plt.savefig("LSMpathMap.jpg", format='jpeg', dpi=1200)

visualize_problem(prob)






ast = AStarSearch(prob, available_actions)
print('Running A-star search algorithm')
start_time = time.time()

solution = ast.search()

path = {
    'latitude' : [],
    'longitude' : []
}

if solution:
    print("Solution found:")
    for state in solution.path():
        path["latitude"].append(prob.grid[state.x][state.y].lat)
        path["longitude"].append(prob.grid[state.x][state.y].long)
        print(state.__str__())
    print("solution cost=", solution.path_cost) #prints coordinate of solution
else:
    print("No solution found.")

end_time = time.time()
running = end_time - start_time
print('runtime', running)

df = pd.DataFrame(path)
df.to_csv('A_star_Path.csv',index=False) 

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
    launches = towerCount*towerMass/capacity

    
    return mat_cost, launches
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import heapq
from collections import deque
import time
import csv

#CLASSES
class GridCell():
    def __init__(self, lat, long, grade, elevation, illumination, science, danger, mode):
        self.lat = lat
        self.long = long
        self.illumination = illumination
        self.grade = grade
        self.elevation = elevation
        self.danger = danger
        self.science = science #used to set goal point
        self.mode = mode #keeps track of transportation that can be implemented

    ''' attribute explanation:
        grade           -> []
        elevation       -> [kept track of for backtracing]
        illumination    -> [0 = PSR, 1 = ILLUMINATED]
        danger          -> [scale 0-1, 1 = GOOD]
        science         -> [keeps track of goal point, 1 = end of path]
        mode            -> [transportation mode: 0 = any, 1 = ONLY TRAMWAY]'''
    
    def __str__(self):
        return f"lat: {self.lat}, long: {self.long}, Illumination {self.illumination}, Grade: {self.grade}, Elevation {self.elevation}, Science: {self.science}, Danger: {self.danger}, Mode: {self.mode}"

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

#DICTIONARY OF AVAILABLE ACTIONS
step = 1
Action ={"UP":(-step, 0),"DOWN":(step, 0),"LEFT":(0, -step),"RIGHT":(0, step)} # Logic of actions seems reversed but it's like this to facilitate plotting.

#FUNCTIONS
def goal_test(state,grid): # IS-GOAL function in R&N
        x, y = state.x, state.y
        return grid[x, y].science > 0.95

def available_actions(state, grid):
    actions = []
    x, y = state.x, state.y
    maxInd = len(grid)

    ### ADJUSTABLE VARIABLES ###
    safety = 0
    step = 1
    
    #path can only move within bounds and cells that have "reasonable danger"
    if x > 0 and grid[x-step][y].danger > safety:
        actions.append(Action["UP"])
    if x < maxInd and grid[x+step][y].danger > safety:
        actions.append(Action['DOWN'])
    if y > 0 and grid[x][y-step].danger > safety:
        actions.append(Action["LEFT"])
    if y < maxInd and grid[x][y+step].danger > safety:
        actions.append(Action["RIGHT"])
    
    return actions 

# Transition model
def transition(state, action, grid):
    #BOUNDING BOX:
    maxInd = len(grid)
    cost = 1
    
    new_x = state.x + action[0]
    new_y = state.y + action[1]
     
    if 0 <= new_x < maxInd and 0 <= new_y < maxInd:
        return State(new_x, new_y), cost
    return state, cost  # Return the same state if out of bounds  

def transition_with_danger(state, action, grid):
    x, y = state.x, state.y
    new_state,_ = transition(state, action, grid)
    
    danger_level = grid[new_state.x][new_state.y].danger
    cost = 0.1 + 3 * (1-danger_level)
    
    return new_state, cost





##### A STAR #####
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
                    if problem.grid[i, j].science > 0.95]
    return min(abs(state.x - gx) + abs(state.y - gy) for gx, gy in goal_positions)

#CONSTRAINT: tram can not be more than 200 m






# ACTUAL CODE STARTS HERE !!!!!!!!!!!!!!!!!!!!!!
pointA = [-85.292466, 36.920242] #start point
pointB = [-84.7906, 29.1957]     #end point
#boundingBox
minLat, maxLat, minLong, maxLong = -85.5, -84, 28, 38

# #test point
# pointA = [1.0,2.0]
# pointB = [4.0,4.0]

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
        #print(row)
        #initialize variables
        # lat = float(row[1])
        # long = float(row[2])
        # ill = float(row[3])
        # grade = float(row[4])
        # elev = float(row[5])
        
        lat = float(row[0])
        long = float(row[1])
        ill = float(row[2])
        grade = float(row[3])
        elev = float(row[4])
        sci = 0

        #Danger Equation Implemented
        danger = (1-grade/20)*ill
        if danger < 0:
            danger = 0

        #Store initial state
        if round(lat,2) == round(pointA[0],2) and round(long,2) == round(pointA[1],2):
            initX = i
            initY = j

        #set goal state using science
        if round(lat,2) == round(pointB[0],2) and round(long,2) == round(pointB[1],2):
            sci = 1

        grid[i,j] = GridCell(lat=lat, long=long, illumination=ill, grade=grade, elevation=elev, danger=danger, science=sci, mode=0)
        #print(grid[i,j])
        #index
        j += 1
        #print(j)
        if j == yippee:
            j = 0
            #print(i)
            i += 1

            if i == 250:
                print('halfway !!')

print("done reading !!!")

# GENERATE PROBLEM 
prob = Problem(State(initX,initY), goal_test, grid, transition_with_danger)


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


def Indices_to_Stereo(i_, j_, n):
    lat = lat_minus + i_ * (lat_plus - lat_minus) / n
    lon = lon_minus + j_ * (lon_plus - lon_minus) / n
    xyz = LLA_to_PCPF(lat, lon, 0.0, 1737400.0)
    return [xyz[0] / (1 - xyz[2]), xyz[1] / (1 - xyz[2])]


### VISULAIZATION ###
def visualize_problem(problem: Problem):
    grid = problem.grid
    n = len(grid) 
    # We create a color map for the grid based on danger values
    color_grid = np.zeros((n, n, 3))  # RGB color grid
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
            if cell.science >= 0.95:  
                color_grid[i, j] = [0, 1, 0]  # Green for goal 
                goalX = i
                goalY = j
            else:
                safety = cell.danger  # Now, danger=1 means safe, danger=0 means dangerous
                color_grid[i, j] = [1, safety, safety]  # Red fades to white as safety increases
                # PSR = cell.illumination
                # color_grid[i,j] = [1, PSR, PSR]
                Z[i, j] = safety

    init_state = problem.initial_state
    color_grid[init_state.x,init_state.y] = [0,0,1]

    # Plot the grid
    # plt.imshow(color_grid, extent=[0, n, 0, n], origin='lower')
    # plt.pcolormesh(X, Y, Z, cmap='RdBu')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(y_stereo, x_stereo, Z, cmap='RdBu')
    ax.set_aspect('equal', 'box')

    xy_init = Indices_to_Stereo(initX + 0.5, initY + 0.5, n)
    xy_goal = Indices_to_Stereo(goalX + 0.5, goalY + 0.5, n)

    # plt.plot(initY + 0.5, initX + 0.5, 'b*', markersize=12, markeredgecolor='black')
    plt.plot(xy_init[1], xy_init[0], 'y*', markersize=24, markeredgecolor='black', label='Start Point')
    # plt.text(xy_init[1], xy_init[0], "   Start Point", verticalalignment='center')
    # plt.plot(goalY + 0.5, goalX + 0.5, 'g*', markersize=12, markeredgecolor='black', label='End Point')
    plt.plot(xy_goal[1], xy_goal[0], 'g*', markersize=24, markeredgecolor='black', label='End Point')
    # plt.text(xy_goal[1], xy_goal[0], "   End Point", verticalalignment='center')
    ax.grid(True)
    ax.legend()
    ax.set_title("Problem Visualization")
    plt.show()

    print('pass')

visualize_problem(prob)



# GENERATE SOLUTION WITH A STAR
# ast = AStarSearch(prob, available_actions)
# solution = ast.search()
# if solution:
#     print("Solution found:")
#     for state in solution.path():
#         print(state.__str__())
#     print("solution cost=", solution.path_cost) #prints coordinate of solution
# else:
#     print("No solution found.")

#backtrace / check for rough elevation changes


#calculate power and budget and operations




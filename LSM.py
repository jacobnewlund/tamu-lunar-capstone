import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq
from collections import deque
import time
"""
AERO 489-689 Homework 1 Provided Code
Original code by David Fornos Fernandez 
Heavily edited by Lunar-Sensitivity-Model-Team
"""
# Data structure for a grid cell
class GridCell():
    def __init__(self, science, danger, grade, elevation, roughness):
        self.science = science
        self.grade = grade
        self.elevation = elevation
        self.roughness  = roughness
        self.danger = danger

    def __str__(self):
        return f"Science: {self.science}, Grade: {self.grade}, Elevation {self.elevation}, Roughness {self.roughness}, Danger: {self.danger}"

# Data structure for the state
class transportState:
    # Just two attributes that represent position x, y in the grid
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"transportState({self.x}, {self.y})"

    def __eq__(self, other):
        # Check if two transportState objects are equal based on their coordinates
        if isinstance(other, transportState):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        # Allow transportState objects to be used in sets and as dictionary keys
        return hash((self.x, self.y))

# Data structure for a Node in the search algorithms
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

# Data structure for a basic search problem
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

# Available actions (ACTIONS(s) function from R&N)
transAction ={"UP":(-1, 0),"DOWN":(1, 0),"LEFT":(0, -1),"RIGHT":(0, 1)} # Logic of actions seems reversed but it's like this to facilitate plotting.

def available_actions(state, grid):
    actions = []
    x, y = state.x, state.y
    height, width = len(grid), len(grid[0])
    
    # Check if movement in each direction is within bounds and the target cell is safe
    if x > 0 and grid[x-1][y].danger < 0.5:
        actions.append(transAction["UP"])
    if x < height - 1 and grid[x+1][y].danger < 0.5:
        actions.append(transAction["DOWN"])
    if y > 0 and grid[x][y-1].danger < 0.5:
        actions.append(transAction["LEFT"])
    if y < width - 1 and grid[x][y+1].danger < 0.5:
        actions.append(transAction["RIGHT"])

    return actions

def available_actions_cost_model(state, grid): 
    actions = []
    x, y = state.x, state.y
    height, width = len(grid), len(grid)
    if x > 0:
        actions.append(transAction["UP"])
    if x < height - 1:
        actions.append(transAction["DOWN"])
    if y > 0:
        actions.append(transAction["LEFT"])
    if y < width - 1:
        actions.append(transAction["RIGHT"])

    return actions

# Transition model (RESULT(s,a) function in R&N)
def transition(state, action, grid):
    height, width = len(grid),len(grid)
    cost = 1
    
    new_x = state.x + action[0]
    new_y = state.y + action[1]
     
    if 0 <= new_x < height and 0 <= new_y < width:
        return transportState(new_x, new_y), cost
    return state, cost  # Return the same state if out of bounds  

def transition_with_danger(state, action, grid):
    x, y = state.x, state.y
    new_state,_ = transition(state, action, grid)
    
    danger_level = grid[new_state.x][new_state.y].danger
    cost = 0.1 + 3 * danger_level
    
    return new_state, cost

def goal_test(state,grid): # IS-GOAL function in R&N
        x, y = state.x, state.y
        return grid[x, y].science > 0.95


def visualize_problem(problem):
    grid = problem.grid
    n = len(grid) 
    # We create a color map for the grid based on danger values
    color_grid = np.zeros((n, n, 3))  # RGB color grid

    for i in range(n):
        for j in range(n):
            cell = grid[i][j]
            danger = cell.danger
            if cell.science >=0.95:  
                color_grid[i, j] = [0, 1, 1]  # Green for goal
            # elif cell.danger > 0,5:
            #     color_grid[i, j] = [1, 1-]
            else:  # Safe cell
                color_grid[i, j] = [1, 1-danger, 1-danger]  # Blue for safe cells
    init_state = problem.initial_state
    color_grid[init_state.x,init_state.y] = [0,1,0]

    fig, ax = plt.subplots()

    # Show RGB grid
    ax.imshow(color_grid, origin='lower', extent=[0, n, 0, n])

    # Overlay invisible danger map for colorbar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])  # required for colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Danger Level")

    # Plot the grid
    plt.imshow(color_grid, extent=[0, n, 0, n])
    plt.grid(True)
    plt.title("Problem Visualization")


def visualize_solution(problem, path):
    grid = problem.grid
    n = len(grid)

    visualize_problem(problem)
    # Adjust for the flipped y-axis in matplotlib vs numpy
    path_x = [state.y + 0.5 for state in path]  # Flip x and y because matplotlib expects (row, col)
    path_y = [n - state.x - 0.5 for state in path]  # Flip y-axis to match numpy indexing

    plt.plot(path_x, path_y, color='black', linewidth=2, label="Path")
    plt.scatter(0.5, 0.5, color='green', label='point A')
    plt.scatter(9.5, 9.5, color='blue', label='point B')
    plt.title("EXAMPLE A*")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'Solution{len(problem.grid)}')
    plt.close()

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

ill = [[0, 0, 0, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 0, 0, 0, 0]]

gr = [[1, 0.8, 0.9, 0.9, 0.7],
      [0.6, 1, 0.7, 0.7, 0.7],
      [0.4, 0.4, 0.4, 0.5, 0.6],
      [0, 0.2, 0.3, 0, 0],
      [0, 0, 0, 0, 0]]

elev = [[0.7, 0, 0.7, 0.7, 0.3],
        [0.7, 0.7, 0.8, 0.9, 0.1],
        [0.2, 0.3, 0.1, 0.1, 0],
        [0.2, 0, 0, 0.8, 0.7],
        [0, 0, 0, 1, 0.9]]

rou = [[0, 0, 0.1, 0.5, 0.5],
       [0, 0, 0.1, 0, 0],
       [0.2, 1, 0.2, 0.2, 0.2],
       [0.3, 1, 1, 1, 0.3],
       [0, 0, 1, 1, 0.1]]

tot = np.zeros((5,5))

print(len(ill))
for i in range(0,len(ill)):
    for j in range(0,len(ill)):
        tot[i][j] = round(((gr[i][j] + elev[i][j] + rou[i][j])/3), 2)
print(tot)

def create_path(pointA, pointB):
    grid = np.empty((10, 10), dtype=object)
    for i in range(10):
        for j in range(10):
            science = 0
            # grade = gr[i][j]
            # elevation = elev[i][j]
            # roughness = rou[i][j]
            danger = random.random()
            grid[i, j] = GridCell(science=science, grade=0, elevation=0, roughness=0, danger=danger)
    Ay = int(pointA[1])
    Ax = int(pointA[3])
    By = int(pointB[1])
    Bx = int(pointB[3])
    grid[Bx][By].science = 1
    initial_state = transportState(Ax, Ay)

    prob = Problem(initial_state, goal_test, grid, transition_with_danger)

    ast= AStarSearch(prob, available_actions_cost_model)
    solution = ast.search()
    if solution:
        print("Solution found:")
        for state in solution.path():
            print(state.__str__())
        print("solution cost=", solution.path_cost)
        visualize_solution(prob, solution.path())
    else:
        print("No solution found.")

    return solution

##### USER INTERACTION #####
# A = input('Enter Location A (x,y) Coordinates: ')
# B = input('Enter Location B (x,y) Coordinates: ')
create_path('(0,9)', '(9,0)')

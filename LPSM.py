import numpy as np
import LSM_funcs as LSM

def cost(grade,LTV_grade,mono_grade,mode,return_max = False):
    if mode == 'mono':
        max_grade = min(LTV_grade,mono_grade)
    else:
        max_grade = LTV_grade

    if return_max == False:
        return np.exp(grade/max_grade)
    elif return_max == True:
        return np.e

def help(input = True,output = True):
    if input == True:
        print(
        'LPSM takes input in the following format:\n',
        "data_file,start,stop,bounds,grid_dimensions,LTV_grade,mono_grade,cost_func = cost,radius = 1737.4,mode='multi',PSR_travel='False',Plot_Legend = False,heuristic = 'manhattan'\n",
        '\n',
        'data_file is a csv file with latitudes (float degrees), longitudes (float degrees), elevations (float meters), grades (float degrees), and illumination (integer, 0 if PSR, 1 else) in that order.\n',
        'start is a tuple consisting of a latitude and longitude (lat,lon) (tuples, degrees).\n',
        'stop is a tuple consisting of a latitude and longitude (lat,lon) (tuples degrees).\n',
        'bounds is a list of tuples containing the bounding box for the A* search area [(lat,lat),(lon,lon)] (list of tuples of floats, degrees).\n',
        'grid_dimensions is a tuple containg the # of points that the terrain data grid contains in the latitudinal and longitudinal directions (# points in lat direction, # points in lon direction) (integers).\n',
        'LTV_grade is the highest acceptable grade of the LTV to construct the path (float degrees).\n',
        'mono_grade is the highest elevation traversable by the monorail transportation mode. 8 percent is the default input (float, radians).\n',
        'cost_func is a cost function for A* to use while searching. Exponential by default. Cost functions should have the following inputs in the following order:\n(grade,LTV_grade,mono_grade,mode,return_max = False) (function)',
        'radius is the radius of the planetary body that is being pathed over in kilometers. The default is 1737.4 km for the Moon (float, km).\n',
        "mode is the type of transportation mode to be used when searching, set to 'multi' by default.\n'tram' and 'mono' are accepted inputs for a purely ropeway or monorail path respectively. (string)\n",
        'PSR_travel is to mark whether A* is allowed to search through permanently shadowed regions or not. Set to False by default (boolean).\n',
        'Plot_legend is to mark whether the plotting function should include a legend. This is set to False by default (boolean)\n',
        "heuristic is a function for A* to use. The heuristic is set to 'manhattan' by default. 'Euclidian' is also an accepteble input. (string)\n\n\n"
        )

    if output == True:
        print(
            'The LPSM outputs a list of points the path travels along in the following format:\n',
            '(index, latitude, longitude, elevation, grade, action).\n',
            'action is a parameter used in the visualization when modes are being fit to the path.\n',
            'The path_segments csv is used in the mode selection process.\n'

            'The LPSM also outputs a .jpg of the terrain in sterographic coordinates.\n',
            'Traversability is a percent difference from the highest accepteble cost.\n',
            'The path will also be plotted over the map along with a legend if Plot_legend is marked True in the input.\n'
        )

    if output == input == False:
        print('Please mark at least one input as True.')
    

def LPSM(data_file,start,stop,bounds,grid_dimensions,LTV_grade,mono_grade = np.asin(.08),cost_func = cost,radius = 1737.4,mode='multi',PSR_travel='False',Plot_Legend = False,heuristic = 'manhattan'):
    '''
    data_file = .csv with lat,lon,elev,grade,illumination
    start = (lat,lon)
    stop = (lat,lon)
    bounds = [(low lat, high lat), (low lon, high lon)]
    grid_dimensions = (# points in lat direction, # points in lon direction)
    LTV_grade = degrees
    mode = 'multi' or 'tram' or 'mono'
    PSR_travel = True/False
    '''
    lat_points = grid_dimensions[0]
    lon_points = grid_dimensions[1]
    grid = np.empty((lat_points,lon_points), dtype=object)
    startX, startY, goalX, goalY = LSM.read_data(data_file, grid, grid_dimensions, LTV_grade, mono_grade, start, stop, cost_func, radius, mode, PSR_travel)

    prob = LSM.Problem(LSM.State(startX,startY), LSM.goal_test, grid, LSM.transition, startX, startY, goalX, goalY)

    LSM.run_Astar(prob,goalX,goalY,heuristic)
    LSM.visualize_problem(bounds,prob,cost,LTV_grade,mono_grade,mode = mode, legend = Plot_Legend)



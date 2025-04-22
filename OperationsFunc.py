import numpy as np
from tabulate import tabulate

#Note this function builds off of the towerPlacer.py 



def operations(curveTowers, straightTowers, tTowers, curveDist, straightDist, vehicleChoice):
    '''INPUT EXPLAINED:
        Towers = # of towers for type
        Distance is in METERS'''
    tot_dist = curveDist + straightDist
    tot_towers = curveTowers + straightTowers + tTowers

    # NOTE ADJUST FOLLOWING FOR DIFFERENT MATERIAL/PRICING
    monoRailMass = 33 #kg/m of rail
    tramRailMass = 2.75 #kg/m rail
    genRailCost = 3.50 # $/m
    #TOWERS
    monoTowerMass1 = 140 #kg [4m tower]
    monoTowerMass2 = 187 #kg [6m tower]
    tramTowerMass = 456 #kg  [10m tower]
    towerCost = 3   #$/kg
 
    #mass calc
    monoMass1 = monoRailMass*curveDist + monoTowerMass1*curveTowers
    monoMass2 = monoRailMass*curveDist + monoTowerMass2*curveTowers
    tramMass = tramRailMass*straightDist + tramTowerMass*straightTowers
    transitionMass = tTowers*tramTowerMass*2 
    tot_mass1 = monoMass1 + tramMass + transitionMass
    tot_mass2 = monoMass2 + tramMass + transitionMass

    #material cost
    towerCost1 = (monoTowerMass1*curveTowers + tramTowerMass*straightTowers + transitionMass)*towerCost
    towerCost2 = (monoTowerMass2*curveTowers + tramTowerMass*straightTowers + transitionMass)*towerCost
    railCost = (monoRailMass*curveDist + tramRailMass*straightDist)*genRailCost
    mat_cost1 = towerCost1 + railCost
    mat_cost2 = towerCost2 + railCost
    mono_cost1 = tot_dist*monoRailMass*genRailCost + tot_towers*monoTowerMass1*towerCost
    mono_cost2 = tot_dist*monoRailMass*genRailCost + tot_towers*monoTowerMass2*towerCost
    tram_cost = tot_dist*tramRailMass*genRailCost + tot_towers*tramTowerMass*towerCost
    print('Material cost calculations complete !!')

    print('The total material cost of the system will be in between ${:.2f} and ${:.2f} depending on the number of 4m vs. 6m towers'.format(mat_cost1, mat_cost2))
    print('The total cost of a purely 4m tower monorail system is ${:.2f}'.format(mono_cost1))
    print('The total cost of a purely 6m tower monorail system is ${:.2f}'.format(mono_cost2))
    print('The total cost of a purely tramway system is ${:.2f}'.format(tram_cost))
    print()

    #MASS TABLE
    print(tabulate([['4m and 10m Towers', mat_cost1], 
                    ['6m and 10m Towers', mat_cost2], 
                    ['Purely 4m Towers', mono_cost1], 
                    ['Purely 6m Towers', mono_cost2], 
                    ['Purely 10m Towers', tram_cost]], 
                   headers=['System', 'Cost [$]'], tablefmt='orgtbl'))
    print()

    #### DEPLOYMENT SECTION ###
    if vehicleChoice == 0:
        print('Your launch company of choice is SpaceX')
        capacity = 17*907.185 #kg/launch
        launchCost = 5
    elif vehicleChoice == 1:
        print('Your launch company of choice is BlueOrigin')
        capacity = 3*907.185 #kg/launch
        launchCost = 6
    launches1 = np.ceil(tot_mass1/capacity)
    launches2 = np.ceil(tot_mass2/capacity)

    #LTV delivery schedule made with the following ASSUMPTIONS
    #1. 25m between each tower
    #2. one trip per tower
    LTVcapacity = 1600 #kg [MAX capacity. nominal is 800]
    LTVspeed = 6000 #m/day
    LTVlimit = 20 #km before recharge necessary 
    LTVtime = 1
    LTVtravel = 0

    print('figuring out LTV STUFF')
    print()
    for i in range(1, tot_towers+1):
        if LTVtravel == LTVspeed:
            LTVtime += 1
            LTVtravel = 0
        LTVtravel += 25*i*2

    months = launches2 + np.floor(LTVtime/28)
    leftover_days = LTVtime%28 

    tot_time = launches2 + LTVtime/28 #months

    print('The total deployment of the system will take approximately {} months'.format(months, leftover_days))
    print('Assuming 1 launch per month, it will take {} months to deliver system to the Lunar South Pole'.format(launches1))
    print('It will take the LTV roughly {} days to deliver construction material along the path'.format(LTVtime))
    print()

    tot_cost1 = mat_cost1+launches1*launchCost
    print('The total cost (material + launch) is ${:.2f}'.format(tot_cost1))
    
    return mat_cost1, tot_time, tot_cost1

operations(3, 5, 1, 125, 125, 0)
import pandas as pd
import numpy as np

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
    #assuming 6m towers
    monoTowerMass = 187 #kg
    tramTowerMass = 456 #kg
    transTowerMass = 455 #kg #NOTE THIS NUMBER IS JUST ASSUMING HEAVIER TOWER MASS
    towerCost = 3   #$/kg
 
    #mass calc
    monoMass = monoRailMass*curveDist + monoTowerMass*curveTowers
    tramMass = tramRailMass*straightDist + tramTowerMass*straightTowers
    transitionMass = tTowers*transTowerMass
    tot_mass = monoMass + tramMass + transitionMass
    print('Mass calculations complete')

    #material cost
    towerCost = (monoTowerMass*curveTowers + tramTowerMass*straightTowers + transTowerMass*tTowers)*towerCost
    railCost = (monoRailMass*curveDist + tramRailMass*straightDist)*genRailCost
    mat_cost = towerCost + railCost
    print('Material cost calculations complete !!')

    print('The total material cost of the system is ${:.2f}'.format(mat_cost))
    print('The total cost of a purely monorail system is ${:.2f}'.format(tot_dist*monoRailMass*genRailCost + tot_towers*monoTowerMass*towerCost))
    print('The total cost of a purely tramway system is ${:.2f}'.format(tot_dist*tramRailMass*genRailCost + tot_towers*tramTowerMass*towerCost))
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
    launches = np.ceil(tot_mass/capacity)

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

    months = launches + np.floor(LTVtime/28)
    leftover_days = LTVtime%28 

    tot_time = launches + LTVtime/28 #months

    print('The total deployment of the system will take approximately {} months'.format(months, leftover_days))
    print('Assuming 1 launch per month, it will take {} months to deliver system to the Lunar South Pole'.format(launches))
    print('It will take the LTV roughly {} days to deliver construction material along the path'.format(LTVtime))
    print()

    tot_cost = mat_cost+launches*launchCost
    print('The total cost (material + launch) is ${:.2f}'.format(tot_cost))
    
    return mat_cost, tot_time, tot_cost

operations(5, 5, 1, 125, 125, 0)
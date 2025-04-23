import numpy as np
from tabulate import tabulate

#Note this function builds off of the towerPlacer.py 



def operations(smallTower, medTower, largeTower, transTower, monoDist, tramDist, vehicleChoice):
    '''INPUT EXPLAINED:
        Towers = # of towers for type
        Distance is in METERS'''
    
    tot_dist = monoDist + tramDist
    tot_tower = int(smallTower + medTower + largeTower + transTower)

    # NOTE ADJUST FOLLOWING FOR DIFFERENT MATERIAL/PRICING
    monoRailMass = 40 #kg/m of rail
    tramRailMass = 7.3 #kg/m rail
    genRailCost = 3.50 # $/m
    #TOWERS
    monoTowerMass1 = 140 #kg [4m tower]
    monoTowerMass2 = 187 #kg [6m tower]
    tramTowerMass = 456 #kg  [10m tower]
    genTowerCost = 3   #$/kg
    

    print()
    print('Beginning material calculations')
    #mass calc of multi-mode system
    monoMass = monoRailMass*monoDist + monoTowerMass1*smallTower + monoTowerMass2*medTower 
    tramMass = tramRailMass*tramDist + tramTowerMass*largeTower
    transitionMass = transTower*tramTowerMass*2 
    tot_mass = monoMass + tramMass + transitionMass

    #material cost
    towerCost = (smallTower*monoTowerMass1 + medTower*monoTowerMass2 + (largeTower + transTower*2)*tramTowerMass)*genTowerCost
    railCost = (monoRailMass*monoDist + tramRailMass*tramDist)*genRailCost
    mat_cost = towerCost + railCost
    #purely one system calculations:
    mono_cost1 = tot_dist*monoRailMass*genRailCost + (tot_dist/25)*monoTowerMass1*genTowerCost
    mono_cost2 = tot_dist*monoRailMass*genRailCost + (tot_dist/25)*monoTowerMass2*genTowerCost
    tram_cost = tot_dist*tramRailMass*genRailCost + (tot_dist/100)*tramTowerMass*genTowerCost
    print('--> Material cost calculations complete !!')
    # print('The total material cost of the system will be ${:.2f}'.format(mat_cost))
    # print('The total cost of a purely 4m tower monorail system is ${:.2f}'.format(mono_cost1))
    # print('The total cost of a purely 6m tower monorail system is ${:.2f}'.format(mono_cost2))
    # print('The total cost of a purely tramway system is ${:.2f}'.format(tram_cost))

    #### DEPLOYMENT SECTION ###
    print('Beginning deployment calculations')
    if vehicleChoice == 0:
        print('--> Your launch company of choice is SpaceX')
        capacity = 17*907.185 #kg/launch
        launchCost = 100000000
    elif vehicleChoice == 1:
        print('Your launch company of choice is BlueOrigin')
        print('There is no launch cost available for BlueOrigin...this is a gap')
        capacity = 3*907.185 #kg/launch
        launchCost = 0
    launches = np.ceil(tot_mass/capacity)
    mono1_launches = np.ceil((tot_dist*monoRailMass + (tot_dist/25)*monoTowerMass1)/capacity)
    mono2_launches = np.ceil((tot_dist*monoRailMass + (tot_dist/25)*monoTowerMass2)/capacity)
    tram_launches = np.ceil((tot_dist*tramRailMass + (tot_dist/100)*tramTowerMass)/capacity)

    #LTV delivery schedule made with the following ASSUMPTIONS
    #1. 25m between each tower
    #2. one trip per tower
    LTVcapacity = 1600 #kg [MAX capacity. nominal is 800]
    LTVlimit = 20 #km before recharge necessary
    LTVspeed = 15000 #m/hr 
    LTVtime = 1
    LTVtravel = 0

    for i in range(1, tot_tower+1):
        if LTVtravel > LTVspeed:
            LTVtime += 1
            LTVtravel = 0
        LTVtravel += 25*i*2
    
    LTVtimeMono = 0
    LTVtravel = 0
    for i in range(1, int(tot_dist/25)+1):
        if LTVtravel > LTVspeed:
            LTVtimeMono += 1
            LTVtravel = 0
        LTVtravel += 25*i*2

    LTVtimeTram = 0
    LTVtravel = 0
    for i in range(1, int(tot_dist/100)+1):
        if LTVtravel > LTVspeed:
            LTVtimeTram += 1
            LTVtravel = 0
        LTVtravel += 25*i*2

    tot_time = launches + np.ceil(LTVtime/730) #months
    tot_launchCost = launches*launchCost
    mono_launchCost1 = mono1_launches*launchCost
    mono_launchCost2 = mono2_launches*launchCost 
    tram_launchCost = tram_launches*launchCost

    tot_cost = mat_cost+launches*launchCost

    print('--> Deployment calculations complete !!')

    # print('The total deployment of the system will take approximately {} months'.format(months, leftover_days))
    # print('Assuming 1 launch per month, it will take {} months to deliver system to the Lunar South Pole'.format(launches))
    # print('It will take the LTV roughly {} days to deliver construction material along the path'.format(LTVtime))
    # print()

    print('THE TOTAL COST OF THE SYSTEM IS ${:,.2f}'.format(tot_cost))
    print()

    #OUTPUT TABLE
    print(tabulate([['Mixture of Transportation', '{:,.2f}'.format(mat_cost), '{:,.2f}'.format(tot_launchCost), '{:,.2f}'.format(tot_cost), tot_time], 
                    ['Purely Monorail [4m Towers]', '{:,.2f}'.format(mono_cost1), '{:,.2f}'.format(mono_launchCost1), '{:,.2f}'.format(mono_cost1 + mono_launchCost1), np.ceil(mono1_launches + LTVtimeMono/720)], 
                    ['Purely Monorail [6m Towers]', '{:,.2f}'.format(mono_cost2), '{:,.2f}'.format(mono_launchCost2), '{:,.2f}'.format(mono_cost2 + mono_launchCost2),np.ceil(mono2_launches + LTVtimeMono/720)], 
                    ['Purely Tramway [10m Towers]', '{:,.2f}'.format(tram_cost), '{:,.2f}'.format(tram_launchCost), '{:,.2f}'.format(tram_cost + tram_launchCost), np.ceil(tram_launches + LTVtimeTram/720)]], 
                   headers=['System', 'Mat Cost [$]', 'Launch Cost [$]', 'Total Cost [$]', 'Deployment Time [months]'], tablefmt='orgtbl'))
    print()
    print(tot_tower)
    print(tot_dist/25)

    return mat_cost, tot_time, tot_cost

operations(0, 262, 208, 14, 6458.39, 21306.93, 0)
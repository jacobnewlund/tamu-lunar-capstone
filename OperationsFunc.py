import numpy as np
from tabulate import tabulate

#Note this function builds off of the towerPlacer.py 



def operations(smallTower, medTower, largeTower, transTower, monoDist, tramDist, vehicleChoice):
    '''INPUT EXPLAINED:
        smallTower = # of 4m Towers
        medTower = # of 6m Towers
        largeTower = # of 10m Towers
        transTowers = # of transition Towers
        Distance values are in METERS
        vehicldChoice = 0 (SpaceX) or 1 (BlueOrigin)'''
    
    tot_dist = monoDist + tramDist
    tot_tower = int(smallTower + medTower + largeTower + transTower)

    # NOTE ADJUST FOLLOWING FOR DIFFERENT MATERIAL + PRICING
    monoRail_mass = 40 #kg/m of rail
    tramRail_mass = 7.3 #kg/m rail
    genRailCost = 3.50 # $/m
    #TOWERS
    smallMass = 140 #kg [4m tower]
    medMass = 187 #kg [6m tower]
    largeMass = 456 #kg  [10m tower]
    genTowerCost = 3   #$/kg
    
    print('Beginning material calculations')
    #mass calc of multi-mode system
    monoMass = monoRail_mass*monoDist + smallMass*smallTower + medMass*medTower 
    tramMass = tramRail_mass*tramDist + largeMass*largeTower
    transitionMass = transTower*largeMass*2 
    tot_mass = monoMass + tramMass + transitionMass
    print(monoMass)
    print(tramMass)
    print(tot_mass)

    pureSmallMass = monoRail_mass*tot_dist + smallMass*tot_tower
    pureMedMass = monoRail_mass*tot_dist + medMass*tot_tower
    pureLargeMass = tramRail_mass*tot_dist + largeMass*tot_tower

    #material cost
    towerCost = (smallTower*smallMass + medTower*medMass + (largeTower + transTower*2)*largeMass)*genTowerCost
    railCost = (monoRail_mass*monoDist + tramRail_mass*tramDist)*genRailCost
    mat_cost = towerCost + railCost
    #purely one system calculations:
    mono_cost1 = tot_dist*monoRail_mass*genRailCost + (tot_dist/25)*smallMass*genTowerCost
    mono_cost2 = tot_dist*monoRail_mass*genRailCost + (tot_dist/25)*medMass*genTowerCost
    tram_cost = tot_dist*tramRail_mass*genRailCost + (tot_dist/100)*largeMass*genTowerCost
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
    mono1_launches = np.ceil((tot_dist*monoRail_mass + (tot_dist/25)*smallMass)/capacity)
    mono2_launches = np.ceil((tot_dist*monoRail_mass + (tot_dist/25)*medMass)/capacity)
    tram_launches = np.ceil((tot_dist*tramRail_mass + (tot_dist/100)*largeMass)/capacity)

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
    print(tabulate([['Mixture of Transportation', '{:,.0f}'.format(tot_mass), '{:,.2f}'.format(mat_cost), '{:,.2f}'.format(tot_launchCost), '{:,.2f}'.format(tot_cost), tot_time], 
                    ['Purely Monorail [4m Towers]', '{:,.0f}'.format(pureSmallMass), '{:,.2f}'.format(mono_cost1), '{:,.2f}'.format(mono_launchCost1), '{:,.2f}'.format(mono_cost1 + mono_launchCost1), np.ceil(mono1_launches + LTVtimeMono/720)], 
                    ['Purely Monorail [6m Towers]', '{:,.0f}'.format(pureMedMass), '{:,.2f}'.format(mono_cost2), '{:,.2f}'.format(mono_launchCost2), '{:,.2f}'.format(mono_cost2 + mono_launchCost2),np.ceil(mono2_launches + LTVtimeMono/720)], 
                    ['Purely Tramway [10m Towers]', '{:,.0f}'.format(pureLargeMass), '{:,.2f}'.format(tram_cost), '{:,.2f}'.format(tram_launchCost), '{:,.2f}'.format(tram_cost + tram_launchCost), np.ceil(tram_launches + LTVtimeTram/720)]], 
                   headers=['System', 'Mass [kg]', 'Mat Cost [$]', 'Launch Cost [$]', 'Total Cost [$]', 'Deployment Time [months]'], tablefmt='orgtbl'))
    print()

    return mat_cost, tot_time, tot_cost

operations(0, 262, 208, 14, 6458.39, 21306.93, 0)
# THIS COMMENT LINE SHOULD BE THE FIRST LINE OF THE FILE! 

# DON'T CHANGE ANY OF THE BELOW; NECESSARY FOR JOINING SIMULATION 

import os, sys, time, datetime, traceback 

import spaceteams as st 

# def custom_exception_handler(exctype, value, tb): 

#     error_message = "".join(traceback.format_exception(exctype, value, tb)) 

#     st.logger_fatal(error_message) 

#     exit(1) 

# sys.excepthook = custom_exception_handler 

# st.connect_to_sim(sys.argv) 

import numpy as np 

# DON'T CHANGE ANY OF THE ABOVE; NECESSARY FOR JOINING SIMULATION 

################################################################ 

 

moon = st.ProcPlanet.DataStore() 

 

moon.AddGeoBinAltimetryLayer(1.0,  

                             "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014",  

                             st.ProcPlanet.GeoBin_Extra_Args()) 

moon.AddGeoBinAltimetryLayer(2.0,  

                             "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPole75_30m",  

                             st.ProcPlanet.GeoBin_Extra_Args()) 

moon.AddGeoBinAltimetryLayer(3.0,  

                             "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m",  

                             st.ProcPlanet.GeoBin_Extra_Args()) 

 

 

radius = 1737400.0 

lla = st.PlanetUtils.LatLonAlt(np.deg2rad(-85.292466), np.deg2rad(36.920242), 2000.0) # Latitude, then longitude, then 2000.0. Forgot what the two thousand meant. 

loc_pcpf = st.PlanetUtils.LLA_to_PCPF(lla, radius) 

ground = st.ProcPlanet.SampleGround(moon, loc_pcpf, radius, 0.0, 16) # that zero is the reference off the ground 

 

# ground [0] = position at surface in cartesian planet-centered planet-fixed (PCPF) frame 

# ground [1] = normal vector (also in same frame) 

print(ground) 


# spicey py (pip-install spiceypy) for the fancy sun vector stuff psuedo-ray tracing
# Moon frame: "MOON_PA"

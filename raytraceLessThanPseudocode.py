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


# moon.AddGeoBinAltimetryLayer(3.0, "../SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m", st.ProcPlanet.GeoBin_Extra_Args())
# moon.AddGeoBinAltimetryLayer(2.0, "../SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPole75_30m", st.ProcPlanet.GeoBin_Extra_Args())
# moon.AddGeoBinAltimetryLayer(1.0, "../SharedData/PlanetData/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014", st.ProcPlanet.GeoBin_Extra_Args())

# uncomment the below lines if you couldn't stuff the STPRO files into your main drive
moon.AddGeoBinAltimetryLayer(3.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPolar875_10m", st.ProcPlanet.GeoBin_Extra_Args())
moon.AddGeoBinAltimetryLayer(2.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Local/SouthPole/LRO_LOLA_DEM_SPole75_30m", st.ProcPlanet.GeoBin_Extra_Args())
moon.AddGeoBinAltimetryLayer(1.0, "B:/SpaceTeamsPro_0.30.0/SharedData/PlanetData/Moon/Global/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014", st.ProcPlanet.GeoBin_Extra_Args()) 

radius = 1737400.0

def altFinder(lat, long):
    lla = st.PlanetUtils.LatLonAlt(lat, long, 2000.0)
    loc_pcpf = st.PlanetUtils.LLA_to_PCPF(lla, radius)
    ground = st.ProcPlanet.SampleGround(moon, loc_pcpf, radius, 0.0, 16)
    return np.sqrt(ground[0][0]**2 + ground[0][1]**2 + ground[0][2]**2) - radius

startingLatLong = [-85, 35]

lla = st.PlanetUtils.LatLonAlt(np.deg2rad(85), np.deg2rad(35), 2000.0)
loc_pcpf = st.PlanetUtils.LLA_to_PCPF(lla, radius)
ground = st.ProcPlanet.SampleGround(moon, loc_pcpf, radius, 0.0, 16)

print(lla)
print(loc_pcpf)

print(st.PlanetUtils.PCPF_to_LLA(loc_pcpf, radius) )
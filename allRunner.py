# Goal is to run everything from one program since we got a billion of the things flying around right now.
# You can just comment out which ones are not being used, I guess.

# Currently it does not expect you to have SpaceTeamsPro past the first step. Will be required for regenerating terrain data.
# Also does not have function-like inputs where you only have to specify things once. That will change before next Monday. 

import os
import subprocess

import numpy as np

bounds = [(np.deg2rad(-84), np.deg2rad(28)), (np.deg2rad(-85.5), np.deg2rad(38))] # Start bounds and then end bounds. Goes latitude first, then longitude. Want to start from "top left corner" of bounding box.

import datagrabber as dg



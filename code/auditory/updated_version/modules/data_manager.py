# here we handle all things related to the outputs of the scripts
#This means we need a class that handles: 
# creation of a new folder that will go inside the parent folder that was defined in config.py as base_output_path (line 52)
# extrapolate date/time of the experiment
# prompt user for mouse info (This will be handled later in main.py - should be the equivalent of collect_metadata)
# create df and csv with mouse metadata
# get the stimulus string
# create visitation log with roi and stimulus info
#
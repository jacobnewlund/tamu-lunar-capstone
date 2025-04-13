import pandas as pd


def data_stitcher(file1,file2,output = 'terrain_data.csv'):
    #file 1 is the illumination csv file, formated as [latitude, longitude, illumination]
    #^make sure all three are fully spelt out
    #
    #file 2 is the grade, elevation csv file, formated as [latitude, longitude, grade, elevation]
    #^grade and elevation need to be fully spelt out
    #
    #output is the output file name


    #read csv files in as pandas data-frames
    df_1 = pd.read_csv(file1)
    df_2 = pd.read_csv(file2)

    #initialize output data dictionary
    data = {
    "latitude" : [],
    "longitude" : [],
    "illum" : [],
    "grade" : [],
    "elev" :[]
}
    #convert data-frames ot dictionaries
    dict_1 = df_1.to_dict()
    dict_2 = df_2.to_dict()

    #loop through each row in the csvs, and collect data in the output dictionary
    for row in range(len(dict_1['latitude'])):
        data['latitude'].append(dict_1['latitude'][row])
        data['longitude'].append(dict_1['longitude'][row]) 
        data['illum'].append(dict_1['illumination'][row]) 
        data['grade'].append(dict_2['grade'][row]) 
        data['elev'].append(dict_2['elevation'][row])
    
    #convert output dictionary to pandas data-frame, and output csv
    df = pd.DataFrame(data)
    df.to_csv(output, index=False)




combo_dict = data_stitcher('illumination.csv','datagrabbed.csv')

print("we're done... yippee!!!")
### Custom definitions and classes if any ###


import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy

ipl_data=pd.read_csv("edited.csv")
batsmen_data=pd.read_csv("batting_avg.csv")
bowler_data=pd.read_csv("bowling_avg.csv")
teambat_data=pd.read_csv("teambatting_avg.csv")
teambowl_data=pd.read_csv("teambowling_avg.csv")
venue_data=pd.read_csv("venue.csv")
def data_prepare(input):
    try:
        input.loc[0,'venue'] = venue_data.loc[venue_data.loc[:, 'venue'] == input.loc[0,'venue']].average.to_list()[0]
    except IndexError:
        input.loc[0,'venue'] = 47
    input.loc[0,'batting_team'] = teambat_data.loc[teambat_data.loc[:, 'batting_team'] == input.loc[0, 'batting_team']].average.to_list()[0]
    input.loc[0,'bowling_team'] = teambowl_data.loc[teambowl_data.loc[:, 'bowling_team'] == input.loc[0,'bowling_team']].average.to_list()[0]

    names = input.loc[0,'batsmen']
    names = names.split(",")
    sum = 0
    for name in names:
        try:
            sum += batsmen_data.loc[batsmen_data.loc[:, 'Players'] == name].batting_average.to_list()[0]
        except IndexError:
            sum+=5.04
    input.loc[0,'batsmen'] = sum / len(names)

    names = input.loc[0,'bowlers']
    names=names.split(",")
    sum = 0
    for name in names:
        try:
            sum += bowler_data.loc[bowler_data.loc[:, 'bowler'] == name].rpb.to_list()[0]
        except:
            sum+=1
    input.loc[0,'bowlers'] = (sum * 36) / len(names)
    return input



def predictRuns(testInput):
    prediction = 0
    with open('randomforest_model.joblib','rb') as f:
        reg_model=joblib.load(f)
    input=pd.read_csv(testInput)
    try:
        input.loc[0,'venue'] = venue_data.loc[venue_data.loc[:, 'venue'] == input.loc[0,'venue']].average.to_list()[0]
    except IndexError:
        input.loc[0,'venue'] = 47
    try:
        input.loc[0,'batting_team'] = teambat_data.loc[teambat_data.loc[:, 'batting_team'] == input.loc[0, 'batting_team']].average.to_list()[0]
    except IndexError:
        input.loc[0, 'batting_team']=46.73
    try:
        input.loc[0,'bowling_team'] = teambowl_data.loc[teambowl_data.loc[:, 'bowling_team'] == input.loc[0,'bowling_team']].average.to_list()[0]
    except IndexError:
        input.loc[0, 'bowling_team'] =46.40377
    names = input.loc[0,'batsmen']
    names = names.split(",")
    sum = 0
    for name in names:
        try:
            sum += batsmen_data.loc[batsmen_data.loc[:, 'Players'] == name].batting_average.to_list()[0]
        except IndexError:
            sum+=13
    input.loc[0,'batsmen'] = sum / len(names)

    names = input.loc[0,'bowlers']
    names=names.split(",")
    sum = 0
    for name in names:
        try:
            sum += bowler_data.loc[bowler_data.loc[:, 'bowler'] == name].rpb.to_list()[0]
        except:
            sum+=1.38
    input.loc[0,'bowlers'] = (sum * 36) / len(names)
    

    return (int(round(reg_model.predict(input)[0])))



    ### Your Code Here ###


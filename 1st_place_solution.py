import numpy as np 
import pandas as pd 
import tensorflow as tf
import gc
import optuna
from lightgbm import LGBMRegressor

train= pd.read_csv('/kaggle/input/tabular-playground-series-mar-2022/train.csv')
test =  pd.read_csv('/kaggle/input/tabular-playground-series-mar-2022/test.csv')
ss = pd.read_csv('/kaggle/input/tabular-playground-series-mar-2022/sample_submission.csv')


TARGET = 'congestion'
test[TARGET] = 0

#Turning time to DateTime
train['time'] = pd.to_datetime(train['time'])
test['time'] = pd.to_datetime(test['time'])
ss['time'] = test['time']

#Combining all location features 
for df in [train, test]:
    df['xydir'] = df['x'].astype(str) + '_' + df['y'].astype(str) + train['direction']
    df['xy'] = df['x'].astype(str) + '_' + df['y'].astype(str) 
ss['xydir'] = test['xydir']
##################################################################################################
##################################################################################################
#                      Feature Creation                                                          #
##################################################################################################
##################################################################################################

train['test'] = False
test['test'] = True
data = pd.concat([train, test]).reset_index(drop=True)
del train, test; gc.collect()

data['day'] = data.time.dt.weekday
data['hour'] =data.time.dt.hour
data['minute'] = data.time.dt.minute
data['dhm'] = data['day'].astype(str) + '_' + data['hour'].astype(str) + '_' + data['minute'].astype(str)
data['hm'] = data['hour'].astype(str) + '_' + data['minute'].astype(str)
data['hm_xydir'] = data['hm'] + data['xydir']
data['hm_xy'] = data['hm'] + data['xydir']

################
#Global stats
################
#Find the min, max, median, variance, and mean for both 
#the x-y-direction and x-y for each unique hour-minute of the day.
#Modified from https://www.kaggle.com/code/packinman/tps-mar-2022-automl-pycaret-regression
FEATURES = []
for location_time in ['hm_xydir', 'hm_xy']:
    for stat in ['min','max','median','var','mean']:
        name = f'global_{location_time}_{stat}'
        stat = data.loc[data.test==False, [location_time, TARGET]].groupby([location_time]).agg(stat).to_dict()[TARGET]
        data[name] = data[location_time].map(stat)
        FEATURES.append(name)
        
###############################
#Lag Stats
###############################
#I found the mean, var, median, min, max, and 1 interval shift
# for 3/5/10 interval windows and expanding windows.  The intervals
# were every week and every day.
EXPANDING = ['roll', 'expand']
STATS = ['mean','var','median', 'min','max','shift']
TIMES = ['dhm','hm']
LOCATIONS = ['xydir'] 
LENGTHS = [3,5, 10]
i=0
for stat in STATS:
    for time in TIMES:
        for length in LENGTHS:
            for expanding in EXPANDING:
                print(i, end=", ")
                name = f'{time}_{stat}_{length}_{expanding}' 
                if stat == 'shift':
                    #Only shift once per time
                    if length==3:
                        name = f'{time}_{stat}_1'
                        data[name] = data.groupby([time, 'xydir'])[TARGET].apply(lambda x: x.shift())
                elif expanding =='roll':
                    data[name] = data.groupby([time, 'xydir'])[TARGET].apply(lambda x: x.shift().rolling(length,min_periods=1).agg(stat))
                else:
                    #Only use the expanding 1x per loop
                    if length==3:
                        name = f'{time}_{stat}_{expanding}'
                        data[name] = data.groupby([time, 'xydir'])[TARGET].apply(lambda x: x.shift().expanding(min_periods=1).agg(stat))
                FEATURES.append(name)
                i+=1
                
                
##################################################################################################
##################################################################################################
#                      Feature Search                                                            #
##################################################################################################
##################################################################################################            
for feat in ['xydir','xy','dhm','hm','direction', 'hm_xy','hm_xydir']:
    data[feat] = df[feat].astype('category')

bad_features = ['row_id','time',TARGET, 'test']
POSSIBLE_FEATURES = [feat for feat in data.columns if feat not in bad_features]

val_times = data[data.test].time.unique() - pd.Timedelta(days=7)
val = data[data.time.isin(val_times)].reset_index(drop=True).copy()
train = data[data.time<val_times[0]].reset_index(drop=True).copy()

def objective(trial):
    ###################################
    # Generate our trial model.
    ###################################
    FEATURES = []
    for feat in POSSIBLE_FEATURES:
        select_feat = trial.suggest_categorical(feat, [True, False])
        if select_feat:
            FEATURES.append(feat)
    model = LGBMRegressor()
    
    #Masks for day and hours 
    only_test_day = trial.suggest_categorical('only_test_day',[True,False])
    if only_test_day:
        msk_day = train.time.dt.weekday.isin([0])
    else:
        msk_day = pd.Series([True for i in range(train.shape[0])])
        
    only_test_hours = trial.suggest_categorical('only_test_hours',[True,False])
    if only_test_hours:
        msk_hm = train.hm.isin(hm)
    else:
        msk_hm = pd.Series([True for i in range(train.shape[0])])
    model.fit(train.loc[msk_day & msk_hm, FEATURES], train.loc[msk_day & msk_hm, TARGET])
    
    
    #Val Score
    val_preds = model.predict(val[FEATURES])
    score = np.mean(np.abs(val[TARGET].values - val_preds))    
    return score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

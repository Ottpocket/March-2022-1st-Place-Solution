# March-2022-1st-Place-Solution
Code and Write Up for my March 2022 TPS Solution

## Problem
For this competition, we predict traffic congestion for 65 different intersection and traffic direction combinations.  The data is given every 20 minutes from April 1st to noon of September 30th of 1991.  We are tasked with predicting the traffic congestions at 20 minute intervals for each of the 65 intersection/direction pairs from noon to midnight of September 30th.  

## Data
The only features given are the x-y coordinates of the light, the direction of the traffic, and the time.  Between April 1st and September 30th, very few 20 minute time stamps were missing.  

## Validation
I validated on the noon to midnight period 7 days previous to the test data.

## Solution Overview
The winning score was a single lgbm.  First, I created a substantial amount of likelihood encodings and lag features. After creating the features, I found the best subset of the features using Optuna.

### Feature Creation
For the lagged features, I found means, variances, medians, minimums, maximums, and 1 interval shifts. I took this for every x-y-direction combination on both the day and weekday. I used both 3,5, and 10 day rolling windows and expanding windows.

The likelihood encodings took the minimus, maximums, medians, variances, and means for every xy and x-y-direction combination at all hour-minute combinations.
The whole process takes around 5-10 minutes to run.

### Features as Hyperparameters
After creating the above features, I had too many features to hand select the best ones.  Given the surplus of features, I used Optuna to pick the best features.  The insight is that each feature is essentially a switch to turn on or off.   Looked at from this perspective, the features are now hyperparameters: inclusion or exclusion from a model is something needing validation to determine.  Given the ~90 features created above, I didn't have time to run 2^90 models to find the best subset of features.  Instead, I used optuna to find the best selection.  After 300 trials, Optuna got very good at finding the best features.  After the Optuna trials, I just picked the best features and submitted the notebook.

## Failures
Given the spatial relationship of the data, it would make sense to train a regressor on all the spatial points at a given time simultaneously.  I created a neural network to taken in the data for each timestamp and output predictions for each location/direction in the timestampe.  This approach never worked well, unfortunately.

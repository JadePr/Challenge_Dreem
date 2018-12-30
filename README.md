# Dreem Deep Sleep Challenge

The aim of this challenge is to better understand when to stimulate the brain with the Dreem headband. To achieve this, we have some data from the Dreem headband. During this challenge, we focus on the few seconds before a stimulation and we want to determine whether or not it is a good moment for a stimulation. To answer that multiple informations are given on those few seconds before a stimulation, including EEG signal and Accelerometer signal. The impact of the stimulation is computed by comparing delta-wave energy after and before the stim. We want to predict this impact of stimulation to be able to stimulate at the best moment.
The evalutation uses Root Mean Square Error (MSE) to compute distance between predicted stimulation impact and real stimulation impact.


### Prerequisites

You first need to download the data from the challenge : https://www.kaggle.com/c/dreem-deep-sleep-power-increase/data



### Data Description  
### File descriptions  
The zip download includes 3 files :   

    train.csv.zip - the training set. Needs to be unzip before reading.  
    test.csv.zip - the test set. Needs to be unzip before reading.  
    test_solution_sample.csv - A valid sample submission with stimulation impact = 1 everywhere.  

### Data fields  
Here are the features we have to make the prediction :  
    index: Index of the stimulation  
    eeg: 8 seconds of EEG signal before the stimulation (250 Hz)  
    respiration_x: 8 seconds of Accelerometer signal on x axis before the stimulation (50 Hz)  
    respiration_y: 8 seconds of Accelerometer signal on y axis before the stimulation (50 Hz)  
    respiration_z: 8 seconds of Accelerometer signal on z axis before the stimulation (50 Hz)  
    time_previous: time between current stimulation and previous stimulation (-1 means no previous stimulation)  
    number_previous: number of previous stimulations  
    time: time elapsed from beginning of the night  
    user: user id  
    night: night id  
    power_increase: value to predict: impact of the stimulation measured on EEG signal after stimulation.  

## Download the code
To download the code, click the zip download button and then go to your Downloads directory.  
In a terminal, type  
cd ~/Downloads  
python solution.py

## Preprocessing

The first step of our work was to do the preprocessing on the data.   
We identified outliers, that are EEG signals which seemed not to represent the sleep wave but probably a damage with the headband hardware. We identified the maximum (resp. minimum) threshold to exclude these outliers to be 400 (resp. -400) for the EEG signal.

## Fourier transform
To densify the information in the time series signal we computed the Fourier transform for the time series (respiration and EEG), however the respiration signal did not help performing better results (being with or without the Fourier transform).
In the end, we finally removed it from the analysis.  
From the Fourier coefficients, we computed the energy of the signal by adding these coefficients.  

For the EEG, we also computed the mean, standard deviation and sum of the signal.  

The metadata also proved to have a positive influence on the results, so we kept them and finally concatenate our features.

### Results

The best results were obtained using Gradient Boosting Regression, which helped us perform a prediction with a Root Mean Squared Error of 0.53. However, we had previously tried linear regression techniques, and because our first model was overfitting the data, we added regularisation (lasso then ridge), tried Kernel and Support Vector Machines, to then use Gradient Boosting, which performed best concerning the overfitting issue.

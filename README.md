# HST-efficiency-map-with-speed-and-mass-regression
Introduces two classes: hudro-static transmission (HST) and regression models (RegressionModel). The former serves to initilize an HST object for a certain displacement
and other design pararmeters. Included methods allow to size, to calculate efficiencies as well as perfromance characteristics for a given operational regime and to create, 
print and save an efficiency map - a contour or a surface plot for ranges of speeds and torques required. The latter class loads the collected catalogue data of displacements, 
speeds and masses of axial-piston machines from the data.csv file. Then it fits regression models to the data in order to provide inter- and extrapolating predictions. 
The data and the regression models could then be printed and saved.

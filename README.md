# Effmap

Effmap is the python library, which introduces two classes: hydro-static transmission (HST) and regression models (Regressor). The former serves to initilize an HST object for a certain displacement, oil and an engine. Included methods allow to size, to calculate efficiencies as well as perfromance characteristics for a given operational regime and to create, print and save an efficiency map - a contour or a surface plot for ranges of speeds and torques required. 

The latter class loads the collected catalogue data of displacements, speeds and masses of axial-piston machines from the data.csv file. Then it fits regression models to the data in order to provide inter- and extrapolating predictions. The data and the regression models could then be printed and saved.

## Installation
```
pip install effmap
```

## Usage
```
from effmap.hst import HST
from effmap.regressor import Regressor
```
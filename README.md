# Effmap

Effmap is the python library, which introduces two classes: hydro-static transmission (HST) and regression models (Regressor). The former serves to initialize an HST object for a certain displacement, oil and an engine. Included methods allow to size, to calculate efficiencies as well as performance characteristics for a given operational regime and to create, print and save an efficiency map - a contour or a surface plot for ranges of speeds and torques required. 

The latter class loads the collected catalogue data of displacements, speeds and masses of axial-piston machines from the data.csv file. Then it fits regression models to the data in order to provide inter- and extrapolating predictions. The data and the regression models could then be printed and saved.

## Installation
```
pip install effmap
```

## Usage
```
from effmap.hst import HST
from effmap.regressor import Regressor

hst = HST(displ)
hst.compute_sizes()
hst.compute_speed_limit(models['pump_speed'])
hst.load_oil()
hst.plot_oil()
hst.compute_eff(input_speed, pressure_discharge)
hst.compute_loads(pressure_discharge)
hst.plot_eff_maps(max_speed, max_pressure)
```

## Web app
https://effmap.herokuapp.com/

Code for the web app: https://github.com/ivanokhotnikov/effmap_demo
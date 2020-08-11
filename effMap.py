import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import lxml.html as lh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression


def load_engines_dict():
    """Loads the dictionary of available engines.

    For each key - engine name, the value is a dictionary with a performance curve and pivot speeds. A performance curve is in a form lists of engine speed in rpm, torque in Nm and power in kW.
    """
    return {'engine_1': {'speed': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000],
                         'torque': [1350, 1450, 1550, 1650, 1800, 1975, 2200, 2450, 2750, 3100, 3100, 3100, 3100, 3022, 2944, 2849, 2757, 2654, 2200, 1800, 0],
                         'power': [141.372, 167.028, 194.779, 224.624, 263.894, 310.232, 368.614, 436.158, 518.363, 616.799, 649.262, 681.726, 714.189, 727.865, 739.908, 745.866, 750.652, 750.401, 645.074, 546.637, 0],
                         'pivot speed': 2700},
            'engine_2': {'speed': [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
                         'torque': [1000, 1100, 1450, 1750, 2100, 2400, 2600, 2950, 3100, 3300, 3400, 3500, 3400, 3300, 3200, 3000, 2800, 2600, 0],
                         'power': [62.8319, 80.634, 121.475, 164.934, 219.911, 276.46, 326.726, 401.6, 454.484, 518.363, 569.675, 623.083, 640.885, 656.593, 670.206, 659.734, 645.074, 626.224, 0],
                         'pivot speed': 2200}}


class RegressionModel(BaseEstimator):
    """Creates the rergession model object.

    The object performs curve fitting to the provided data using the `reg_func` target function with the ordinary least square method.

    Attributes
    ----------
    data: DataFrame
        The pd.DataFrame object containing the pump and motor speed and mass data.
    fitted: boolean
        The flag to indicate whether fitting was performed.
    machine_type: {'pump', 'motor'}, optional
        The string specifying a machine type under consideration, default 'pump'.
    data_type: {'speed', 'mass'}, optional
        The string specifying a data type, deafult 'speed'.
    """

    def __init__(self, data, machine_type='pump', data_type='speed'):
        self.data = data[data['type'] ==
                         f'{machine_type.capitalize()}']
        self.fitted = False
        self.model_ = self.reg_func
        self.coefs = None
        self.cov = None
        self.machine_type = machine_type
        self.data_type = data_type
        self.r2_ = None
        self.rmse_ = None

    def reg_func(self, x, *args):
        """Specifies the regression target function.

        Based on the data_type in the `data_type` class attribute, the function chooses between linear-exponential or imple linear models.

        Parameters
        ----------
        x: ndarray
            The function variable.
        *args: tuple
            The tuple with the coefficients of the regression model.

        Returns
        -------
        function
            The target regression function.
        """
        if self.data_type == 'speed':
            return args[0] * np.exp(-args[1] * x) + args[2]
        if self.data_type == 'mass':
            return args[0] * x + args[1]

    def fit(self):
        """Fits the target regression model to the provided data with the ordinary least square method nd updated the class attributes accordingly.

        Fitting quality is defined by the R^2 and root-mean-square metrics.
        """
        data = self.data[self.data['type'] ==
                         f'{self.machine_type.capitalize()}'].sample(frac=1)
        x = data['displacement'].to_numpy(dtype='float64')
        y = data[self.data_type].to_numpy(dtype='float64')
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2,
            random_state=np.random.RandomState(np.random.randint(1000)),
            shuffle=True)
        guess_speed = [1e3, 1e-3, 1e3]
        guess_mass = [1, 1]
        if self.data_type == 'speed':
            self.coefs, self.cov = curve_fit(
                self.reg_func, x_train, y_train, guess_speed)
        elif self.data_type == 'mass':
            self.coefs, self.cov = curve_fit(
                self.reg_func, x_train, y_train, guess_mass)
        self.r2_ = r2_score(y_test, self.reg_func(x_test, *self.coefs))
        self.rmse_ = np.sqrt(mean_squared_error(
            y_test, self.reg_func(x_test, *self.coefs)))
        self.fitted = True

    def plot(self, show_figure=False, save_figure=False, format='pdf'):
        """Plots and optionally saves the catalogue data alonoe or the cataloue data and the regression model if fitting was performed.

        Parameters
        ----------
        show_figure: bool, optional
            The flag for saving the figure, default False.
        save_figure: bool, optional
            The flag for saving the figure, default False.
        format : str, optional
            The file extension in which the figure will be saved, default 'pdf'.
        in_app: bool, optional
            The flag allowing to show the plots in a browser.
        """
        fig = go.Figure()
        for idx, i in enumerate(self.data['manufacturer'].unique()):
            fig.add_scatter(
                x=self.data['displacement'][self.data['manufacturer'] == i],
                y=self.data[self.data_type][self.data['manufacturer'] == i],
                mode='markers',
                name=i,
                marker_symbol=idx,
                marker=dict(
                    size=7,
                    line=dict(
                        color='black',
                        width=.5
                    )
                )
            )
        fig.update_layout(
            title=f'{self.machine_type.capitalize()} {self.data_type} data',
            width=800,
            height=600,
            xaxis=dict(
                title=f'{self.machine_type.capitalize()} displacement, cc/rev',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
                range=[0, round(1.1 * max(self.data['displacement']), -2)]
            ),
            yaxis=dict(
                title=f'{self.machine_type.capitalize()} {self.data_type}, rpm' if self.data_type == 'speed' else f'{self.machine_type.capitalize()} {self.data_type}, kg',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
                range=[0, round(1.1 * max(self.data[self.data_type]), -2)] if self.data_type == 'mass' else [
                    round(.9 * min(self.data[self.data_type]), -2), round(1.1 * max(self.data[self.data_type]), -2)]
            ),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,0)',
            showlegend=True,
        )
        if self.fitted:
            data = self.data[self.data['type'] ==
                             f'{self.machine_type.capitalize()}']
            x = data['displacement'].values
            x_cont = np.linspace(.2 * np.amin(x), 1.2 * np.amax(x), num=100)
            for i in zip(('Regression model', 'Upper limit', 'Lower limit'), (0, self.rmse_, -self.rmse_)):
                fig.add_scatter(
                    x=x_cont,
                    y=self.model_(x_cont, *self.coefs)+i[1],
                    mode='lines',
                    name=i[0],
                    line=dict(
                        width=1,
                        dash='dash'),
                )
        if save_figure:
            if not os.path.exists('Images'):
                os.mkdir('Images')
            fig.write_image(
                f'Images/{self.machine_type}_{self.data_type}.{format}')
        if show_figure:
            fig.show()
        return fig


class HST:
    """Creates the HST object.

    Attributes
    ----------
    disp: int
        The displacement of an axial-piston machine in cc/rev
    swash: int, optional
        The maxi swash angle of the axial piston machine in degrees, default 18 degrees when optional.
    pistons: int, optional
        The number of piston in a machine, default 9 when optional.
    oil: {'15w40', '5w30', '10w40'}, optional
        The oil choice from the dictionary of available oils, default '15w40'. Each oil is a dictionary with the following structure: {'visc_kin': float, 'density': float, 'visc_dyn': float, 'bulk': float}. Here 'visc_kin' is the kinematic viscosity of the oil in cSt, 'density' is its density in kg/cub.m, 'visc_dyn' is the dynamic viscosity in Pa s, 'bulk' is the oil bulk modulus in bar. All properties are at 100C.
    engine: {'engine_1', 'engine_2'}, optional
        The engine choice from the dictionary of engines, default 'engine_1'. Each engine is a dictionary with the following structure: {'speed': list, 'torque': list, 'power': list}. Lists must be of the same length.
    input_gear_ratio: float, optional
        The gear ratio of a gear train connecting the HST with an engine, default 0.75, which corresponds to a reduction gear set.
    max_power_input: int, optional
        The maximum mechanical power in kW the HST is meant to transmit, i.e. to take as an input, default 682 kW.
    """

    def __init__(self, disp, swash=18, pistons=9, oil='SAE 15W40', engine='engine_1', input_gear_ratio=.75, max_power_input=680, oil_temp=100):
        self.displ = disp
        self.swash = swash
        self.pistons = pistons
        self.oil = oil
        self.oil_temp = oil_temp
        self.oil_data = None
        self.oil_bulk = 15000
        self.sizes = {}
        self.efficiencies = {}
        self.performance = {}
        self.leaks = {}
        self.engine = engine
        self.input_gear_ratio = input_gear_ratio
        self.max_power_input = max_power_input
        self.pump_speed_limit = None
        self.no_load = None
        self.no_load_coef = None
        self.no_load_intercept = None
        self.import_oils()

    def import_oils(self):
        url = 'https://wiki.anton-paar.com/uk-en/engine-oil/'
        page = requests.get(url)
        doc = lh.fromstring(page.content)
        data = doc.xpath('//tr')
        oils = [i.text_content().rstrip().lstrip().replace('-', '')
                for i in doc.xpath('//h3')[:-2]]
        col = []

        for t in data[0]:
            name = t.text_content().rstrip().lstrip()
            name = name[:name.rfind(' ')]
            col.append((name, []))

        for j in data[1:]:
            i = 0
            for t in j.iterchildren():
                local_data = t.text_content().lstrip().rstrip()
                try:
                    local_data = int(
                        local_data) if i == 0 else float(local_data)
                except:
                    continue
                col[i][1].append(local_data)
                i += 1

        df = pd.DataFrame(
            {(oil, title): column[i * 11: i * 11 + 11]
             for i, oil in enumerate(oils) for (title, column) in col[1:]},
            index=col[0][1][:11])
        df.index.name = index_name = col[0][0]
        self.oil_data = df[self.oil]

    def sizing(self, k1=.75, k2=.91, k3=.48, k4=.93, k5=.91):
        """Defines the basic sizes of the pumping group of an axial piston machine in metres. Updates the `sizes` attribute.

        Parameters
        ----------
        k1, k2, k3, k4, k5: float, optional
            Design balances, default k1 = .75, k2 = .91, k3 = .48, k4 = .93, k5 = .91

        """
        dia_piston = (4 * self.displ * 1e-6 * k1 /
                      (self.pistons ** 2 * np.tan(np.radians(self.swash)))) ** (1 / 3)
        area_piston = np.pi * dia_piston ** 2 / 4
        pcd = self.pistons * dia_piston / (np.pi * k1)
        stroke = pcd * np.tan(np.radians(self.swash))
        min_engagement = 1.4 * dia_piston
        kidney_area = k3 * area_piston
        kidney_width = 2 * (np.sqrt(dia_piston ** 2 + (np.pi - 4) *
                                    kidney_area) - dia_piston) / (np.pi - 4)
        land_width = k2 * self.pistons * area_piston / \
            (np.pi * pcd) - kidney_width
        rad_ext_int = (pcd + kidney_width) / 2
        rad_ext_ext = rad_ext_int + land_width
        rad_int_ext = (pcd - kidney_width) / 2
        rad_int_int = rad_int_ext - land_width
        area_shoe = k4 * area_piston / np.cos(np.radians(self.swash))
        rad_ext_shoe = np.pi * pcd * k5 / (2 * self.pistons)
        rad_int_shoe = np.sqrt(rad_ext_shoe ** 2 - area_shoe / np.pi)
        self.sizes = {'d': dia_piston, 'D': pcd, 'h': stroke, 'eng': min_engagement,
                      'rbo': rad_ext_int, 'Rbo': rad_ext_ext, 'Rbi': rad_int_ext, 'rbi': rad_int_int, 'rs': rad_int_shoe, 'Rs': rad_ext_shoe}

    def predict_speed_limit(self, RegModel):
        """Defines the pump speed limit."""
        self.pump_speed_limit = [RegModel.reg_func(
            self.displ, *RegModel.coefs)+i for i in (-RegModel.rmse_, 0, +RegModel.rmse_)]

    def efficiency(self, speed_pump, pressure_discharge, pressure_charge=25.0, A=.17, Bp=1.0, Bm=.5, Cp=.001, Cm=.005, D=125, h1=15e-6, h2=15e-6, h3=25e-6, eccentricity=1):
        """Defines efficiencies and performance characteristics of the HST made of same-displacement axial-piston machines.

        Parameters
        ----------
        speed_pump: int
            The HST input, or pump, speed in rpm.
        pressure_discharge: int
            The discharge pressures in bar.
        pressure_charge: int, optional
            The charge pressure in bar, default 25 bar.
        A, Bp, Bm, Cp, Cm, D: float, optional
            Coefficients in the efficiency model, default A = .17, Bp = 1.0, Bm = .5, Cp = .001, Cm = .005, D = 125.
        h1, h2, h3: float, optional
            Clearances in m, default h1 = 15e-6, h2 = 15e-6, h3 = 25e-6.
        eccentricity: float, optional
            Eccentricity ratio of a psiton in a bore, default 1.

        Returns
        --------
        out: dict
            The dictionary containing as values efficiencies of each machine as well as of HST in per cents. The dictionary structure is as following:
            {'pump': {'volumetric': float, 'mechanical': float, 'total': float},
            'motor': {'volumetric': float, 'mechanical': float, 'total': float},
            'hst': {'volumetric': float, 'mechanical': float, 'total': float}}

        """
        leak_block = np.pi * h1 ** 3 * 0.5 * (pressure_discharge * 1e5 + pressure_charge * 1e5) * (
            1 / np.log(self.sizes['Rbo'] / self.sizes['rbo']) + 1 / np.log(self.sizes['Rbi'] / self.sizes['rbi'])) / (6 * self.oil_data.loc[self.oil_temp]['Dyn. Viscosity'] * 1e-3)
        leak_shoes = (self.pistons * np.pi * h2 ** 3 * 0.5 * (pressure_discharge * 1e5 + pressure_charge * 1e5) / (
            6 * self.oil_data.loc[self.oil_temp]['Dyn. Viscosity'] * 1e-3 * np.log(self.sizes['Rs'] / self.sizes['rs'])))
        leak_piston = np.array([self.pistons * np.pi * self.sizes['d'] * h3 ** 3 * 0.5 * (pressure_discharge * 1e5 + pressure_charge * 1e5) * (
            1 + 1.5 * eccentricity ** 3) * (1 / (self.sizes['eng'] + self.sizes['h'] * np.sin(np.pi * (ii) / self.pistons))) / (12 * self.oil_data.loc[self.oil_temp]['Dyn. Viscosity'] * 1e-3)
            for ii in np.arange(self.pistons)])
        leak_pistons = sum(leak_piston)
        leak_total = sum((leak_block, leak_shoes, leak_pistons))
        th_flow_rate_pump = speed_pump * self.displ / 6e7
        vol_pump = (1 - (pressure_discharge - pressure_charge) / self.oil_bulk
                    - leak_total / th_flow_rate_pump) * 100
        vol_motor = (1 - leak_total / th_flow_rate_pump) * 100
        vol_hst = vol_pump * vol_motor * 1e-2
        mech_pump = (1 - A * np.exp(
            - Bp * self.oil_data.loc[self.oil_temp]['Dyn. Viscosity'] * speed_pump / (self.swash*(pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5))
            - Cp * np.sqrt(
            self.oil_data.loc[self.oil_temp]['Dyn. Viscosity'] * speed_pump / (self.swash*(pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5))
            - D / (self.swash * (pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5)) * 100
        mech_motor = (1 - A * np.exp(
            - Bm * self.oil_data.loc[self.oil_temp]['Dyn. Viscosity'] * speed_pump * vol_hst * 1e-2 / (self.swash * (pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5))
            - Cm * np.sqrt(
            self.oil_data.loc[self.oil_temp]['Dyn. Viscosity'] * speed_pump*vol_hst * 1e-2 / (self.swash*(pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5))
            - D / (self.swash * (pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5)) * 100
        mech_hst = mech_pump * mech_motor * 1e-2
        total_pump = vol_pump * mech_pump * 1e-2
        total_motor = vol_motor * mech_motor * 1e-2
        total_hst = total_pump * total_motor * 1e-2
        torque_pump = (pressure_discharge * 1e5 - pressure_charge * 1e5) * \
            self.displ * 1e-6 / (2 * np.pi * mech_pump * 1e-2)
        torque_motor = (pressure_discharge * 1e5 - pressure_charge * 1e5) * self.displ * \
            1e-6 / (2 * np.pi * mech_pump * 1e-2) * (mech_hst * 1e-2)
        power_pump = torque_pump * speed_pump * np.pi / 30 * 1e-3
        power_motor = power_pump * total_hst * 1e-2
        speed_motor = speed_pump * vol_hst * 1e-2
        self.performance = {'pump': {'speed': speed_pump, 'torque': torque_pump, 'power': power_pump},
                            'motor': {'speed': speed_motor, 'torque': torque_motor, 'power': power_motor},
                            'delta': {'speed': speed_pump - speed_motor, 'torque': torque_pump - torque_motor, 'power': power_pump - power_motor},
                            'charge pressure': pressure_charge, 'discharge pressure': pressure_discharge}
        self.efficiencies = {'pump': {'volumetric': vol_pump, 'mechanical': mech_pump, 'total': total_pump},
                             'motor': {'volumetric': vol_motor, 'mechanical': mech_motor, 'total': total_motor},
                             'hst': {'volumetric': vol_hst, 'mechanical': mech_hst, 'total': total_hst}}
        return self.efficiencies

    def no_load_test_data(self, *args):
        X, Y = np.reshape([], (-1, 1)), np.reshape([], (-1, 1))
        for speed, pressure in args:
            X = np.r_[X, np.reshape(speed, (-1, 1))]
            Y = np.r_[Y, np.reshape(pressure, (-1, 1))]
        lin_reg = LinearRegression()
        lin_reg.fit(X, Y)
        self.no_load = (X, Y)
        self.no_load_intercept = lin_reg.intercept_
        self.no_load_coef = lin_reg.coef_

    def plot_oil(self):
        fig = go.Figure()
        fig.add_scatter(
            mode='lines+markers',
            x=self.oil_data.index,
            y=self.oil_data.loc[:]['Dyn. Viscosity'],
            yaxis='y1',
            name='Dynamic viscosity, mPa s',
            line=dict(
                width=1,
                color='indianred',
                shape='spline'
            ),
            connectgaps=True,
        )
        fig.add_scatter(
            mode='lines+markers',
            x=self.oil_data.index,
            y=self.oil_data.loc[:]['Kin. Viscosity'],
            yaxis='y1',
            name='Kinematic viscosity, cSt',
            line=dict(
                width=1,
                color='navy',
                shape='spline'
            ),
            connectgaps=True,
        )
        fig.add_scatter(
            mode='lines+markers',
            x=self.oil_data.index,
            y=self.oil_data.loc[:]['Density']*1e3,
            yaxis='y2',
            name='Density, kg/cub.m',
            line=dict(
                width=1,
                color='orange'
            )
        )
        fig.update_layout(
            title=f'Viscosity and density of {self.oil}',
            width=800,
            height=500,
            xaxis=dict(
                title='Oil temperature, degrees',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
                range=[min(self.oil_data.index), max(self.oil_data.index)]
            ),
            yaxis=dict(
                title='Viscosity',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
                range=[0, 1500]
            ),
            yaxis2=dict(
                title='Density',
                linecolor='black',
                mirror=True,
                linewidth=0.5,
                overlaying='y',
                side='right',
            ),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,0)',
            showlegend=True,
            legend_orientation='h',
            legend=dict(x=0, y=-.2)
        )
        return fig

    def plot_eff_maps(self, max_speed_pump, max_pressure_discharge, min_speed_pump=1000, min_pressure_discharge=75, pressure_charge=25.0, pressure_lim=480, res=100, show_figure=True, save_figure=False, format='pdf'):
        """Plots and optionally saves the HST efficiency maps.

        Parameters
        ----------
        max_speed_pump: int
            The upper limit of the input(pump) speed range on the map in rpm.
        max_pressure_discharge: int
            The upper limit of the discharge pressure range on the map in bar.
        min_speed_pump: int, optional
            The lower limit of the input speed range on the map in rpm, default nmin = 1000 rpm.
        min_pressure_discharge: int, optional
            The lower limit of the discharge pressure range on the map in bar, default pmin = 100 bar.
        pressure_lim: int, optional
            The torque limiter setting in bar, default 480 bar.
        res: float, optional
            The resolution of the map. The number of efficiency samples calculated per axis, default = 100.
        show_figure: bool, optional
            The flag for saving the figure, default True.
        save_figure: bool, optional
            The flag for saving the figure, default True.
        format : str, optional
            The file extension in which the figure will be saved, default 'pdf'.
        in_app: bool, optional
            The flag allowing to show the plots in a browser.
        """
        speed = np.linspace(min_speed_pump, max_speed_pump, res)
        pressure = np.linspace(min_pressure_discharge,
                               max_pressure_discharge, res)
        eff_hst = np.array(
            [[self.efficiency(i, j, pressure_charge=pressure_charge)['hst']['total']
              for i in speed]
             for j in pressure]
        )
        mech_eff_pump = np.array(
            [[self.efficiency(i, j, pressure_charge=pressure_charge)['pump']['mechanical']
              for i in speed]
             for j in pressure]
        )
        torque_pump = self.displ * 1e-6 * \
            (pressure - pressure_charge) * 1e5 / \
            (2 * np.pi * np.amax(mech_eff_pump, axis=0) * 1e-2)
        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                z=eff_hst,
                x=speed,
                y=pressure,
                colorscale='Portland',
                showscale=False,
                contours_coloring='lines',
                name='Total HST efficiency, %',
                contours=dict(
                    coloring='lines',
                    start=50,
                    end=90,
                    size=1,
                    showlabels=True,
                    labelfont=dict(
                        size=8,
                        color='black'))
            ),
        )
        fig.add_scatter(
            mode='lines',
            x=speed,
            y=np.reshape(self.no_load_coef, (-1,))*speed +
            np.reshape(self.no_load_intercept, (-1,)),
            yaxis='y1',
            name='No-load test limit',
            line=dict(
                width=1,
                dash='dash',
                color='purple',
            )
        )
        fig.update_layout(
            title=f'HST efficiency map and the engine torque curve {self.oil} at {self.oil_temp}C',
            width=800,
            height=700,
            xaxis=dict(
                title='HST input speed, rpm',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
            ),
            yaxis=dict(
                title='HST discharge pressure, bar',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
                range=[min(pressure), max(pressure)]
            ),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,0)',
            showlegend=True,
            legend_orientation='h',
            legend=dict(x=0, y=-.1)
        )
        if self.engine:
            ENGINES = load_engines_dict()
            pressure_pivot = self.max_power_input * 1e3 * 30 / np.pi / \
                ENGINES[self.engine]['pivot speed'] / self.input_gear_ratio * 2 * np.pi / \
                self.displ / 1e-6 / 1e5 * \
                np.amax(mech_eff_pump) * 1e-2 + pressure_charge
            _ = self.efficiency(
                ENGINES[self.engine]['pivot speed'] * self.input_gear_ratio, pressure_pivot)
            performance_pivot = self.performance
            fig.add_scatter(
                x=self.input_gear_ratio * np.asarray(
                    ENGINES[self.engine]['speed']),
                y=np.asarray(ENGINES[self.engine]
                             ['torque']) / self.input_gear_ratio,
                name='Engine torque',
                mode='lines+markers',
                marker=dict(
                    size=3
                ),
                line=dict(
                    color='indianred',
                    width=1),
                yaxis='y2',
            )
            fig.add_scatter(
                x=speed,
                y=self.max_power_input * 1e3 * 30 /
                (np.pi * speed),
                name='Torque at max power',
                mode='lines',
                line=dict(
                    color='steelblue',
                    width=1),
                yaxis='y2'
            )
            fig.add_scatter(
                x=[self.input_gear_ratio *
                    ENGINES[self.engine]['pivot speed']],
                y=[performance_pivot['pump']['torque']],
                name='Pivot turn',
                mode='markers',
                marker=dict(
                    color='steelblue',
                    size=7,
                    line=dict(
                        color='navy',
                        width=1
                    ),
                ),
                yaxis='y2'
            )
            fig.add_scatter(
                x=[np.amin(speed), np.amax(speed)],
                y=[pressure_lim, pressure_lim],
                mode='lines',
                name='Limiter setting',
                line=dict(
                    color='darkseagreen',
                    dash='dash',
                    width=1),
                yaxis='y1',
            )
            fig.update_layout(
                xaxis=dict(
                    dtick=200,
                    range=[min_speed_pump, max_speed_pump],
                ),
                yaxis2=dict(
                    title='HST input torque, Nm',
                    range=[np.amin(torque_pump), np.amax(torque_pump)],
                    overlaying='y',
                    side='right',
                    showline=True,
                    linecolor='black',
                ),
                yaxis=dict(
                    range=[np.amin(pressure), np.amax(pressure)]),
            )
        if self.pump_speed_limit:
            for i in zip(self.pump_speed_limit, ('Min rated speed', 'Rated speed', 'Max rated speed'), ('green', 'orange', 'red')):
                fig.add_scatter(
                    x=[i[0], i[0]],
                    y=[np.amin(pressure), np.amax(pressure)],
                    mode='lines',
                    name=i[1],
                    line=dict(
                        dash='dash',
                        width=1,
                        color=i[2],
                    ),
                    yaxis='y1',
                )
        if save_figure:
            if not os.path.exists('Images'):
                os.mkdir('Images')
            fig.write_image(f'Images/eff_map_{self.displ}.{format}')
        if show_figure:
            fig.show()
        return fig


def plot_catalogues():
    st.title('Catalogue data and regressions')
    df = pd.read_csv('data.csv', index_col='#')
    models = {}
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                        shared_yaxes=True, vertical_spacing=0.1, horizontal_spacing=0.07,
                        subplot_titles=[
                            'Pump speed data', 'Pump mass data', 'Motor speed data', 'Motor mass data']
                        )
    for i, machine_type in enumerate(('pump', 'motor')):
        for j, data_type in enumerate(('speed', 'mass')):
            data = df[df['type'] ==
                      f'{machine_type.capitalize()}']
            model = RegressionModel(
                data,
                machine_type=machine_type,
                data_type=data_type)
            model.fit()
            rmse = np.round(model.rmse_, decimals=2)
            x = data['displacement'].values
            x_cont = np.linspace(.2 * np.amin(x), 1.2 * np.amax(x), num=100)
            for l in zip(('Regression model', 'Upper limit', 'Lower limit'), (0, model.rmse_, -model.rmse_)):
                fig.add_scatter(
                    x=x_cont,
                    y=model.model_(x_cont, *model.coefs)+l[1],
                    mode='lines',
                    name=l[0],
                    line=dict(
                        width=1,
                        dash='dash'),
                    row=j + 1,
                    col=i + 1,
                )
            for idx, k in enumerate(data['manufacturer'].unique()):
                fig.add_scatter(
                    x=data['displacement'][data['manufacturer'] == k],
                    y=data[data_type][data['manufacturer'] == k],
                    mode='markers',
                    name=k,
                    marker_symbol=idx,
                    marker=dict(
                        size=7,
                        line=dict(
                            color='black',
                            width=.5)),
                    row=j + 1,
                    col=i + 1)
            fig.update_xaxes(
                title_text=f'{machine_type.capitalize()} displacement, cc/rev',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
                range=[0, round(1.1 * max(data['displacement']), -2)],
                row=j + 1,
                col=i + 1
            )
            fig.update_yaxes(
                title_text=f'{data_type.capitalize()}, rpm' if data_type == 'speed' else f'{data_type.capitalize()}, kg',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
                range=[0, round(1.2 * max(data[data_type]), -2)] if data_type == 'mass' else [
                    round(.7 * min(data[data_type]), -2), round(1.2 * max(data[data_type]), -2)],
                row=j + 1,
                col=i + 1
            )
            models['_'.join((machine_type, data_type))] = model
    fig.update_layout(
        width=800,
        height=800,
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,0)',
        showlegend=False,
    )
    st.write(fig)
    return models


def set_sidebar():
    st.sidebar.markdown('Setting up the efficiency map')
    oil = st.sidebar.selectbox(
        'Select an oil:',
        ('SAE 15W40', 'SAE 10W40', 'SAE 10W60', 'SAE 5W40', 'SAE 0W30', 'SAE 30'))
    oil_temp = st.sidebar.slider(
        'Select oil temperature',
        min_value=0,
        max_value=100,
        value=100,
        step=10)
    max_displ = st.sidebar.slider(
        'Select a displcement',
        min_value=100,
        max_value=800,
        value=440,
        step=5)
    max_power = st.sidebar.slider(
        'Select a max power',
        min_value=400,
        max_value=800,
        value=685,
        step=5)
    gear_ratio = st.sidebar.slider(
        'Select an input gear ratio',
        min_value=.5,
        max_value=2.,
        value=.75,
        step=.25)
    max_speed = st.sidebar.slider(
        'Select a max speed',
        min_value=1000,
        max_value=4000,
        value=2400,
        step=100)
    max_pressure = st.sidebar.slider('Select the max pressure',
                                     min_value=100,
                                     max_value=800,
                                     value=650,
                                     step=50)
    pressure_lim = st.sidebar.slider('Select the pressure limiter setting',
                                     min_value=300,
                                     max_value=800,
                                     value=480,
                                     step=10)
    return oil, oil_temp, max_displ, max_power, gear_ratio, max_speed, max_pressure, pressure_lim


def run():
    models = plot_catalogues()
    oil, oil_temp, max_displ, max_power, gear_ratio, max_speed, max_pressure, pressure_lim = set_sidebar()
    hst = HST(max_displ, oil=oil, oil_temp=oil_temp, max_power_input=max_power,
              input_gear_ratio=gear_ratio)
    hst.sizing()
    hst.predict_speed_limit(models['pump_speed'])
    hst.no_load_test_data((1800, 140), (2025, 180))
    st.title(f'Mechanical properties of oil')
    st.write(hst.plot_oil())
    st.title('Efficiency map')
    st.write(hst.plot_eff_maps(max_speed, max_pressure, pressure_lim=pressure_lim,
                               show_figure=False,
                               save_figure=False))


if __name__ == '__main__':
    run()

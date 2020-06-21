import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_oils():
    return {'15w40': {'visc_kin': 13.65, 'density': 829.1, 'visc_dyn': 13.65 * 829.1 / 1e6, 'bulk': 15000},
            '5w30': {'visc_kin': 12.08, 'density': 820, 'visc_dyn': 12.08 * 820 / 1e6, 'bulk': 15000},
            '10w40': {'visc_kin': 14.61, 'density': 804.5, 'visc_dyn': 14.61 * 804.5 / 1e6, 'bulk': 15000}}


def load_engines():
    return {'engine_1': {'speed': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000],
                         'torque': [1350, 1450, 1550, 1650, 1800, 1975, 2200, 2450, 2750, 3100, 3100, 3100, 3100, 3022, 2944, 2849, 2757, 2654, 2200, 1800, 0],
                         'power': [141.372, 167.028, 194.779, 224.624, 263.894, 310.232, 368.614, 436.158, 518.363, 616.799, 649.262, 681.726, 714.189, 727.865, 739.908, 745.866, 750.652, 750.401, 645.074, 546.637, 0],
                         'pivot speed': 2700},
            'engine_2': {'speed': [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
                         'torque': [1000, 1100, 1450, 1750, 2100, 2400, 2600, 2950, 3100, 3300, 3400, 3500, 3400, 3300, 3200, 3000, 2800, 2600, 0],
                         'power': [62.8319, 80.634, 121.475, 164.934, 219.911, 276.46, 326.726, 401.6, 454.484, 518.363, 569.675, 623.083, 640.885, 656.593, 670.206, 659.734, 645.074, 626.224, 0],
                         'pivot speed': 2200}}


def load_catalogues():
    df = pd.read_csv('data.csv', index_col='#')
    all_pumps = df[df['Type'] == 'Pump']
    all_motors = df[df['Type'] == 'Motor']
    return all_pumps, all_motors


class SpeedModel:
    def __init__(self):
        self.pump_data = load_catalogues()[0]
        self.motor_data = load_catalogues()[1]

    def plot_speed(self, show_figure=True, save_figure=False):
        fig = go.Figure()
        for idx, i in enumerate(set(self.pump_data['Manufacturer'])):
            fig.add_trace(go.Scatter(
                x=self.pump_data['Displacement'][self.pump_data['Manufacturer'] == i],
                y=self.pump_data['Speed'][self.pump_data['Manufacturer'] == i],
                mode='markers',
                name=i,
                marker_symbol=idx,
                marker=dict(
                    size=8,
                    # color=np.random.randn(1000),
                    colorscale='Portland',
                    line=dict(
                        color='black',
                        width=.5
                    )
                )
            ))
        fig.update_layout(
            title='EDA of the pump speed data',
            width=700,
            height=500,
            xaxis=dict(
                title='Pump displacement, cc/rev',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
                range=[0, 800]
            ),
            yaxis=dict(
                title='Pump rated speed, rpm',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
            ),
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            showlegend=True,
        )
        if save_figure:
            if not os.path.exists('images'):
                os.mkdir('images')
            fig.write_image(f'images/eda_speed_model.pdf')
        if show_figure:
            fig.show()


class HSU:
    """Creates the HSU object.

    Parameters
    ----------
    disp : int
        The displacement of an axial-piston machine in cc/rev
    swash : int, optional
        The maxi swash angle of the axial piston machine in degrees, default 18 degrees when optional.
    pistons : int, optional
        The number of piston in a machine, default 9 when optional.
    oil : {'15w40', '5w30', '10w40'}, optional
        The oil choice from the dictionary of available oils , default '15w40'. Each oil is a dictionary with the following structure: {'visc_kin': float, 'density': float, 'visc_dyn': float, 'bulk': float}. Here 'visc_kin' is the kinematic viscosity of the oil in cSt, 'density' is its density in kg/cub.m, 'visc_dyn' is the dynamic viscosity in Pa s, 'bulk' is the oil bulk modulus in bar. All properties are at 100C.
    engine : {'engine_1', 'engine_2'}, optional
        The engine choice from the dictionary of engines, default 'engine_1'. Each engine is a dictionary with the following structure: {'speed': list, 'torque': list, 'power': list}. Lists must be of the same length.
    """

    def __init__(self, disp, swash=18, pistons=9, oil='15w40', engine='engine_1', input_gear_ratio=.75, max_power_input=682):
        self.displ = disp
        self.swash = swash
        self.pistons = pistons
        self.oil = oil
        self.sizes = {}
        self.efficiencies = {}
        self.performance = {}
        self.leaks = {}
        self.engine = engine
        self.input_gear_ratio = input_gear_ratio
        self.max_power_input = max_power_input
        self.pump_speed_limit = None
        self.sizing()
        # self.speed_limit()

    def sizing(self, k1=.75, k2=.91, k3=.48, k4=.93, k5=.91):
        """Defines the basic sizes of the pumping group of an axial piston machine in metres. Updates the `sizes` attribute.

        Parameters
        ----------
        k1, k2, k3, k4, k5 : float, optional
            Design balances, default k1=.75, k2=.91, k3=.48, k4=.93, k5=.91

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

    def predict_speed_limit(self):
        """Defines the pump speed limit."""
        pass

    def efficiency(self, speed_pump, pressure_discharge, pressure_charge=25.0, A=.17, Bp=1.0, Bm=.5, Cp=.001, Cm=.005, D=125, h1=15e-6, h2=15e-6, h3=25e-6, eccentricity=1):
        """Defines efficiencies and performance characteristics of the HSU made of same-displacement axial-piston machines.

        Parameters
        ----------
        speed_pump: int
            The HSU input, or pump, speed in rpm.
        pressure_discharge : int
            The discharge pressures in bar.
        pressure_charge : int, optional
            The charge pressure in bar, default 25 bar.
        A, Bp, Bm, Cp, Cm, D : float, optional
            Coefficients in the efficiency model, default A=.17, Bp=1.0, Bm=.5, Cp=.001, Cm=.005, D=125.
        h1, h2, h3 : float, optional
            Clearances in m, default h1=15e-6, h2=15e-6, h3=25e-6.
        eccentricity : float, optional
            Eccentricity ratio of a psiton in a bore, default 1.

        """
        oils = load_oils()
        leak_block = np.pi * h1 ** 3 * 0.5 * (pressure_discharge * 1e5 + pressure_charge * 1e5) * (
            1 / np.log(self.sizes['Rbo'] / self.sizes['rbo']) + 1 / np.log(self.sizes['Rbi'] / self.sizes['rbi'])) / (6 * oils[self.oil]['visc_dyn'])
        leak_shoes = (self.pistons * np.pi * h2 ** 3 * 0.5 * (pressure_discharge * 1e5 + pressure_charge * 1e5) / (
            6 * oils[self.oil]['visc_dyn'] * np.log(self.sizes['Rs'] / self.sizes['rs'])))
        leak_piston = np.array([self.pistons * np.pi * self.sizes['d'] * h3 ** 3 * 0.5 * (pressure_discharge * 1e5 + pressure_charge * 1e5) * (
            1 + 1.5 * eccentricity ** 3) * (1 / (self.sizes['eng'] + self.sizes['h'] * np.sin(np.pi * (ii) / self.pistons))) / (12 * oils[self.oil]['visc_dyn'])
            for ii in np.arange(self.pistons)])
        leak_pistons = sum(leak_piston)
        leak_total = sum((leak_block, leak_shoes, leak_pistons))
        th_flow_rate_pump = speed_pump * self.displ / 6e7
        vol_pump = (1 - (pressure_discharge - pressure_charge) / oils[self.oil]['bulk']
                    - leak_total / th_flow_rate_pump) * 100
        vol_motor = (1 - leak_total / th_flow_rate_pump) * 100
        vol_hsu = vol_pump * vol_motor * 1e-2
        mech_pump = (1 - A * np.exp(
            - Bp * oils[self.oil]['visc_dyn'] * 1e3 * speed_pump / (self.swash*(pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5))
            - Cp * np.sqrt(
            oils[self.oil]['visc_dyn'] * 1e3 * speed_pump / (self.swash*(pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5))
            - D / (self.swash * (pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5)) * 100
        mech_motor = (1 - A * np.exp(
            - Bm * oils[self.oil]['visc_dyn'] * 1e3 * speed_pump * vol_hsu * 1e-2 / (self.swash * (pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5))
            - Cm * np.sqrt(
            oils[self.oil]['visc_dyn'] * 1e3 * speed_pump*vol_hsu * 1e-2 / (self.swash*(pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5))
            - D / (self.swash * (pressure_discharge * 1e5 - pressure_charge * 1e5) * 1e-5)) * 100
        mech_hsu = mech_pump * mech_motor * 1e-2
        total_pump = vol_pump * mech_pump * 1e-2
        total_motor = vol_motor * mech_motor * 1e-2
        total_hsu = total_pump * total_motor * 1e-2
        torque_pump = (pressure_discharge * 1e5 - pressure_charge * 1e5) * \
            self.displ * 1e-6 / (2 * np.pi * mech_pump * 1e-2)
        torque_motor = (pressure_discharge * 1e5 - pressure_charge * 1e5) * self.displ * \
            1e-6 / (2 * np.pi * mech_pump * 1e-2) * (mech_hsu * 1e-2)
        power_pump = torque_pump * speed_pump * np.pi / 30 * 1e-3
        power_motor = power_pump * total_hsu * 1e-2
        speed_motor = speed_pump * vol_hsu * 1e-2
        self.performance = {'pump': {'speed': speed_pump, 'torque': torque_pump, 'power': power_pump},
                            'motor': {'speed': speed_motor, 'torque': torque_motor, 'power': power_motor},
                            'delta': {'speed': speed_pump - speed_motor, 'torque': torque_pump - torque_motor, 'power': power_pump - power_motor},
                            'charge pressure': pressure_charge, 'discharge pressure': pressure_discharge}
        # self.leaks = {'Machine': {'Block': leak_block, 'Shoes': leak_shoes, 'Pistons': leak_pistons, 'Total': leak_total},
        #               'HSU': {'Block': 2*leak_block, 'Shoes': 2*leak_shoes, 'Pistons': 2*leak_pistons, 'Total': 2*leak_total}}
        self.efficiencies = {'pump': {'volumetric': vol_pump, 'mechanical': mech_pump, 'total': total_pump},
                             'motor': {'volumetric': vol_motor, 'mechanical': mech_motor, 'total': total_motor},
                             'hsu': {'volumetric': vol_hsu, 'mechanical': mech_hsu, 'total': total_hsu}}
        return self.efficiencies

    def plot_eff_map(self, max_speed_pump, max_pressure_discharge, min_speed_pump=1000, min_pressure_discharge=75, pressure_charge=25.0, pressure_comp=480, res=100, show_figure=True, save_figure=False):
        """Plots and optionally saves the HSU efficiency map.

        Parameters
        ----------
        max_speed_pump : int
            The upper limit of the input (pump) speed range on the map in rpm.
        max_pressure_discharge : int
            The upper limit of the discharge pressure range on the map in bar.
        min_speed_pump : int, optional
            The lower limit of the input speed range on the map in rpm, default nmin = 1000 rpm.
        min_pressure_discharge : int, optional
            The lower limit of the discharge pressure range on the map in bar, default pmin = 100 bar.
        pressure_comp : int, optional
            The torque limiter, or compensator, setting in bar, default 480 bar.
        res : float, optional
            The resolution of the map. The number of efficiency samples calculated per axis, default = 100.
        save_figure : bool, optional
            The flag for saving the figure, default True.
        show_figure : bool, optional
            The flag for saving the figure, default False.
        """
        speed = np.linspace(min_speed_pump, max_speed_pump, res)
        pressure = np.linspace(min_pressure_discharge,
                               max_pressure_discharge, res)
        eff_hsu = np.array(
            [[self.efficiency(i, j, pressure_charge=pressure_charge)['hsu']['total']
              for i in speed]
             for j in pressure]
        )
        mech_eff_pump = np.array(
            [[self.efficiency(i, j, pressure_charge=pressure_charge)['pump']['mechanical']
              for i in speed]
             for j in pressure]
        )
        torque_pump = self.displ * 1e-6 * (pressure - pressure_charge) * 1e5 / \
            (2 * np.pi * np.amax(mech_eff_pump, axis=0) * 1e-2)
        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                z=eff_hsu,
                x=speed,
                y=pressure,
                colorscale='Portland',
                showscale=False,
                contours_coloring='lines',
                name='Total HSU efficiency, %',
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
        fig.update_layout(
            title='HSU efficiency map and the engine torque curve',
            width=700,
            height=600,
            xaxis=dict(
                title='HSU input speed, rpm',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
            ),
            yaxis=dict(
                title='HSU discharge pressure, bar',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
            ),
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
        )
        if self.engine:
            engines = load_engines()
            pressure_pivot = self.max_power_input * 1e3 * 30 / np.pi / \
                engines[self.engine]['pivot speed'] / self.input_gear_ratio * 2 * np.pi / \
                self.displ / 1e-6 / 1e5 * \
                np.amax(mech_eff_pump) * 1e-2 + pressure_charge
            _ = self.efficiency(
                engines[self.engine]['pivot speed'] * self.input_gear_ratio, pressure_pivot)
            performance_pivot = self.performance
            fig.add_trace(
                go.Scatter(
                    x=self.input_gear_ratio * np.asarray(
                        engines[self.engine]['speed']),
                    y=np.asarray(engines[self.engine]
                                 ['torque']) / self.input_gear_ratio,
                    name='Engine torque',
                    mode='lines+markers',
                    marker=dict(
                        size=3
                    ),
                    line=dict(
                        color='indianred',
                        width=1),
                    yaxis='y2'
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=speed,
                    y=self.max_power_input * 1e3 * 30 /
                    (np.pi * speed),
                    name='Torque at max power',
                    mode='lines',
                    line=dict(
                        color='steelblue',
                        width=1),
                    yaxis='y2'
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[self.input_gear_ratio *
                       engines[self.engine]['pivot speed']],
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
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[np.amin(speed), np.amax(speed)],
                    y=[pressure_comp, pressure_comp],
                    mode='lines',
                    name='Compensator setting',
                    line=dict(
                        color='darkseagreen',
                        dash='dash',
                        width=1),
                    yaxis='y1',
                )
            )
            fig.update_layout(
                xaxis=dict(
                    dtick=200,
                    range=[min_speed_pump, max_speed_pump],
                ),
                yaxis2=dict(
                    title='HSU input torque, Nm',
                    range=[np.amin(torque_pump), np.amax(torque_pump)],
                    overlaying='y',
                    side='right',
                ),
                yaxis=dict(
                    range=[np.amin(pressure), np.amax(pressure)]),
                showlegend=True,
                legend_orientation='h',
                legend=dict(x=0, y=-.1))
        if save_figure:
            if not os.path.exists('images'):
                os.mkdir('images')
            fig.write_image(f'images/eff_map_{self.displ}.pdf')
        if show_figure:
            fig.show()


if __name__ == '__main__':
    hsu = HSU(440)
    hsu.plot_eff_map(2300, 650,
                     save_figure=True
                     )
    speed = SpeedModel()
    speed.plot_speed(save_figure=True)

import numpy as np
import plotly.graph_objects as go


class HSU:
    """Creates the HSU object.

    Parameters
    ----------
    disp : int
        The displacement of an axial-piston machine in cc/rev
    sa : int, optional
        The maxivisc_dynm swash angle of the axial piston machine in degrees, default 18 degrees when optional.
    z : int, optional
        The number of piston in a machine, default 9 when optional.
    oil : {'15w40', '5w30', '10w40'}, optional
        Choices of available oils, default '15w40'. Each is a dictionary with the following structure: {'visc_kin': float, 'density': float, 'bulk': float}.
        Here 'visc_kin' is the kinematic viscosity of the oil in cSt, 'density' is its density in kg/cub.m, 'bulk' is the bulk modulus in bar. All properties are at 100C.
    """
    oils = {'15w40': {'visc_kin': 13.65, 'density': 829.1, 'visc_dyn': 13.65 * 829.1 / 1e6, 'bulk': 15000},
            '5w30': {'visc_kin': 12.08, 'density': 820, 'visc_dyn': 12.08 * 820 / 1e6, 'bulk': 15000},
            '10w40': {'visc_kin': 14.61, 'density': 804.5, 'visc_dyn': 14.61 * 804.5 / 1e6, 'bulk': 15000}}

    def __init__(self, disp, swash=18, pistons=9, oil='15w40'):
        self.displ = disp
        self.swash = swash
        self.pistons = pistons
        self.oil = oil
        self.sizes = {}
        self.efficiencies = {}
        self.performance = {}
        self.leaks = {}
        self.size()

    def size(self, k1=.75, k2=.91, k3=.48, k4=.93, k5=.91):
        """Calculates the basic sizes of the pumping group of an axial piston machine in metres. Updates the `sizes` attribute.

        Parameters
        ----------
        k1, k2, k3, k4, k5 : float, optional
            Design balances, default k1=.75, k2=.91, k3=.48, k4=.93, k5=.91

        """
        dia_pist = (4 * self.displ * 1e-6 * k1 /
                    (self.pistons ** 2 * np.tan(np.radians(self.swash)))) ** (1 / 3)
        area_pist = np.pi * dia_pist ** 2 / 4
        pcd = self.pistons * dia_pist / (np.pi * k1)
        stroke = pcd * np.tan(np.radians(self.swash))
        min_engagement = 1.4 * dia_pist
        kidney_area = k3 * area_pist
        kidney_width = 2 * (np.sqrt(dia_pist ** 2 + (np.pi - 4) *
                                    kidney_area) - dia_pist) / (np.pi - 4)
        land_width = k2 * self.pistons * area_pist / \
            (np.pi * pcd) - kidney_width
        rad_ext_int = (pcd + kidney_width) / 2
        rad_ext_ext = rad_ext_int + land_width
        rad_int_ext = (pcd - kidney_width) / 2
        rad_int_int = rad_int_ext - land_width
        area_shoe = k4 * area_pist / np.cos(np.radians(self.swash))
        rad_ext_shoe = np.pi * pcd * k5 / (2 * self.pistons)
        rad_int_shoe = np.sqrt(rad_ext_shoe ** 2 - area_shoe / np.pi)
        self.sizes = {'d': dia_pist, 'D': pcd, 'h': stroke, 'eng': min_engagement,
                      'rbo': rad_ext_int, 'Rbo': rad_ext_ext, 'Rbi': rad_int_ext, 'rbi': rad_int_int, 'rs': rad_int_shoe, 'Rs': rad_ext_shoe}

    def efficiency(self, speed_pump, pres_discharge, pres_charge=25.0, A=.17, Bp=1.0, Bm=.5, Cp=.001, Cm=.005, D=125, h1=15e-6, h2=15e-6, h3=25e-6, eccentricity=1):
        """Calculates efficiencies and performance characteristics of the HSU made of same-displacement axial-piston machines.

        Parameters
        ----------
        speed_pump: float
            The HSU input, or pump, speed in rpm.
        pres_discharge : float
            The discharge pressures in bar.
        pres_charge : float, optional
            The charge pressure in bar, default 25 bar.
        A, Bp, Bm, Cp, Cm, D : float, optional
            Coefficients in the efficiency model, default A=.17, Bp=1.0, Bm=.5, Cp=.001, Cm=.005, D=125.
        h1, h2, h3 : float, optional
            Clearances in m, default h1=15e-6, h2=15e-6, h3=25e-6.
        eccentricity : float, optional
            Eccentricity ratio of a psiton in a bore, default 1.

        """
        leak_block = np.pi * h1 ** 3 * 0.5 * (pres_discharge * 1e5 + pres_charge * 1e5) * (
            1 / np.log(self.sizes['Rbo'] / self.sizes['rbo']) + 1 / np.log(self.sizes['Rbi'] / self.sizes['rbi'])) / (6 * self.oils[self.oil]['visc_dyn'])
        leak_shoes = (self.pistons * np.pi * h2 ** 3 * 0.5 * (pres_discharge * 1e5 + pres_charge * 1e5) / (
            6 * self.oils[self.oil]['visc_dyn'] * np.log(self.sizes['Rs'] / self.sizes['rs'])))
        leak_piston = np.array([self.pistons * np.pi * self.sizes['d'] * h3 ** 3 * 0.5 * (pres_discharge * 1e5 + pres_charge * 1e5) * (
            1 + 1.5 * eccentricity ** 3) * (1 / (self.sizes['eng'] + self.sizes['h'] * np.sin(np.pi * (ii) / self.pistons))) / (12 * self.oils[self.oil]['visc_dyn'])
            for ii in np.arange(self.pistons)])
        leak_pistons = sum(leak_piston)
        leak_total = sum((leak_block, leak_shoes, leak_pistons))
        th_flow_rate_pump = speed_pump * self.displ / 6e7
        vol_pump = (1 - (pres_discharge - pres_charge) / self.oils[self.oil]['bulk']
                    - leak_total / th_flow_rate_pump) * 100
        vol_motor = (1 - leak_total / th_flow_rate_pump) * 100
        vol_hsu = vol_pump * vol_motor * 1e-2
        mech_pump = (1 - A * np.exp(
            - Bp * self.oils[self.oil]['visc_dyn'] * 1e3 * speed_pump / (self.swash*(pres_discharge * 1e5 - pres_charge * 1e5) * 1e-5))
            - Cp * np.sqrt(
            self.oils[self.oil]['visc_dyn'] * 1e3 * speed_pump / (self.swash*(pres_discharge * 1e5 - pres_charge * 1e5) * 1e-5))
            - D / (self.swash * (pres_discharge * 1e5 - pres_charge * 1e5) * 1e-5)) * 100
        mech_motor = (1 - A * np.exp(
            - Bm * self.oils[self.oil]['visc_dyn'] * 1e3 * speed_pump*vol_hsu * 1e-2 / (self.swash*(pres_discharge * 1e5 - pres_charge * 1e5) * 1e-5))
            - Cm * np.sqrt(
            self.oils[self.oil]['visc_dyn'] * 1e3 * speed_pump*vol_hsu * 1e-2 / (self.swash*(pres_discharge * 1e5 - pres_charge * 1e5) * 1e-5))
            - D / (self.swash * (pres_discharge * 1e5 - pres_charge * 1e5) * 1e-5)) * 100
        mech_hsu = mech_pump * mech_motor * 1e-2
        total_pump = vol_pump * mech_pump * 1e-2
        total_motor = vol_motor * mech_motor * 1e-2
        total_hsu = total_pump * total_motor * 1e-2
        torque_pump = (pres_discharge * 1e5 - pres_charge * 1e5) * \
            self.displ * 1e-6 / (2 * np.pi * mech_pump * 1e-2)
        torque_motor = (pres_discharge * 1e5 - pres_charge * 1e5) * self.displ * \
            1e-6 / (2 * np.pi * mech_pump * 1e-2) * (mech_hsu * 1e-2)
        power_pump = torque_pump * speed_pump * np.pi / 30 * 1e-3
        power_motor = power_pump * total_hsu * 1e-2
        speed_motor = speed_pump * vol_hsu * 1e-2
        self.performance = {'Pump': {'Speed': speed_pump, 'Torque': torque_pump, 'Power': power_pump},
                            'Motor': {'Speed': speed_motor, 'Torque': torque_motor, 'Power': power_motor},
                            'Delta': {'Speed': speed_pump - speed_motor, 'Torque': torque_pump - torque_motor, 'Power': power_pump - power_motor},
                            'Charge pressure': pres_charge, 'Discharge pressure': pres_discharge}
        self.efficiencies = {'Pump': {'Volumetric': vol_pump, 'Mechanical': mech_pump, 'Total': total_pump},
                             'Motor': {'Volumetric': vol_motor, 'Mechanical': mech_motor, 'Total': total_motor},
                             'HSU': {'Volumetric': vol_hsu, 'Mechanical': mech_hsu, 'Total': total_hsu}}
        self.leaks = {'Machine': {'Block': leak_block, 'Shoes': leak_shoes, 'Pistons': leak_pistons, 'Total': leak_total},
                      'HSU': {'Block': 2*leak_block, 'Shoes': 2*leak_shoes, 'Pistons': 2*leak_pistons, 'Total': 2*leak_total}}
        return self.efficiencies

    def plot_eff_map(self, max_speed_pump, max_pressure, min_speed_pump=1000, min_pressure=100, res=100):
        """Plots and saves the HSU efficiency map.

        Parameters
        ----------
        max_speed_pump : float
            The upper limit of the input (pump) speed range on the map in rpm.
        max_pressure : float
            The upper limit of the discharge pressure range on the map in bar.
        min_speed_pump : float, optional
            The lower limit of the input speed range on the map in rpm, default nmin = 1000 rpm.
        min_pressure : float, optional
            The lower limit of the discharge pressure range on the map in bar, default pmin = 100 bar.
        res : float, optional
            The resolution of the map. The number of efficiency samples calculated per axis, default = 100.
        """
        speed = np.linspace(min_speed_pump, max_speed_pump, res)
        pressure = np.linspace(min_pressure, max_pressure, res)
        eff_hsu = [[self.efficiency(speed[i], pressure[j])['HSU']['Total']
                    for i in range(len(speed))]
                   for j in range(len(pressure))]
        return eff_hsu


if __name__ == '__main__':
    NGLAV = HSU(440)
    NGLAV.size()
    map = NGLAV.plot_eff_map(3000, 500)
    fig = go.Figure(
        data=go.Contour(z=map,
                        contours=dict(coloring='heatmap',
                                      showlabels=True,                                         labelfont=dict(size=12, color='white',)
                                      )
                        )
    )
    fig.show()

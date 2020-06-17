import numpy as np


class HSU:
    """
    def __init__(self, V):
        self.V = V
        self.sa = 18
        self.z = 9
        self.k1 = .75
        self.k2 = .91
        self.k3 = .48
        self.k4 = .93
        self.k5 = .91

    def size(self):
        """Calculates the basic sizes of the pumping group of an axial piston pump.
        Parameters
        ----------
        V: int
            The displacement of an axial-piston machine in cc/rev
        sa: int, optional
            The maximum swash angle of the axial piston machine in degrees, default 18 degrees when optional.
        z: int, optional
            The number of piston in a machine, default 9 when optional.
        k1: float, optional
            The PCD balance, default 0.75 when optional.
        k2: float, optional
            The cylinder block balance, default 0.91 when optional.
        k3: float, optional
            The kidney area balance, default 0.48 when optional.
        k4: float, optional
            The shoe balance, default 0.93 when optional.
        k5: float, optional
            PCD filling with shoes, default 0.91 when optional.

        Returns
        ------
        dict{'V': int, 'sa': int, 'd': float, 'D': float, 'rbo': float, 'Rbo': float, 'Rbi': float,  'rbi': float, 'rs': float, 'Rs': float}
            The dictionary of main sizes of a pumping group. The keys signify:

        """
        dia_pist = (4 * self.V * 1e-6 * self.k1 /
                    (self.z ** 2 * np.tan(np.radians(self.sa)))) ** (1 / 3)
        area_pist = np.pi * dia_pist ** 2 / 4
        pcd = self.z * dia_pist / (np.pi * self.k1)
        stroke = pcd * np.tan(np.radians(self.sa))
        min_engagement = 1.4 * dia_pist
        kidney_area = self.k3 * area_pist
        kidney_width = 2 * (np.sqrt(dia_pist ** 2 + (np.pi - 4) *
                                    kidney_area) - dia_pist) / (np.pi - 4)
        land_width = self.k2 * self.z * area_pist / \
            (np.pi * pcd) - kidney_width
        rad_ext_int = (pcd + kidney_width) / 2
        rad_ext_ext = rad_ext_int + land_width
        rad_int_ext = (pcd - kidney_width) / 2
        rad_int_int = rad_int_ext - land_width
        area_shoe = self.k4 * area_pist / np.cos(np.radians(self.sa))
        rad_ext_shoe = np.pi * pcd * self.k5 / (2 * self.z)
        rad_int_shoe = np.sqrt(rad_ext_shoe ** 2 - area_shoe / np.pi)
        return {'d': dia_pist, 'D': pcd, 'h': stroke, 'eng': min_engagement,
                'rbo': rad_ext_int, 'Rbo': rad_ext_ext, 'Rbi': rad_int_ext, 'rbi': rad_int_int, 'rs': rad_int_shoe, 'Rs': rad_ext_shoe}

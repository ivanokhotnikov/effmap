import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize
from sklearn import metrics
plt.rcParams.update({'font.size': 8})


def imp():
    """
    Imports and returns the calculation parameters. Takes data either from the params.xlsx, or data.json. 

    Returns
    ---
    - dict{"z": int, "h": list, "k": list, "e": float, "A": float, "Bm": float, "Bp": int, "Cm": float, "Cp": float, "D": int, "mu": float, "b": float, "nE": list, "tE": list, "pE": list}

        The dictionary of generic calculation parameters of an axial piston pump as the number of pistons, eccentricity ratio, design balances, leakage clearances and coefficients in the mechanical efficiency model. 

    """
    if os.path.isfile('data.json'):
        with open("data.json", "r") as read_file:
            data = json.load(read_file)
        EngSpeed = np.array(data["Engine"]["Speed"])
        EngTorque = np.array(data["Engine"]["Torque"])
        EngPower = np.array(data["Engine"]["Power"])
        z = int(data["HSU"]["Size"]["z"])
        e = float(data["Parameters"]["e"])
        A = float(data["Parameters"]["A"])
        Bm = float(data["Parameters"]["Bm"])
        Bp = int(data["Parameters"]["Bp"])
        Cm = float(data["Parameters"]["Cm"])
        Cp = float(data["Parameters"]["Cp"])
        D = int(data["Parameters"]["D"])
        mu = float(data["Oil"]["mu"])
        b = float(data["Oil"]["b"])
        h, k = [float(data["Parameters"]["h"][i]) for i in np.arange(3)], [
            float(data["Parameters"]["k"][i]) for i in np.arange(5)]
    else:
        with pd.ExcelFile("params.xlsx") as xls:
            auxSh = xls.parse("Aux")
        engine = xls.parse("Engine")
        val = auxSh["Value"].values
        symb = auxSh["Symbol"].values
        EngSpeed = engine["n"].values
        EngTorque = engine["t"].values
        EngPower = engine["p"].values
        z = int(val[symb == "z"])
        e = float(val[symb == "e"])
        A = float(val[symb == "A"])
        Bm = float(val[symb == "Bm"])
        Bp = int(val[symb == "Bp"])
        Cm = float(val[symb == "Cm"])
        Cp = float(val[symb == "Cp"])
        D = int(val[symb == "D"])
        mu = float(val[symb == "mu"])
        b = float(val[symb == "b"])
        h, k = [float(val[symb == "h" + str(i + 1)]) for i in np.arange(3)
                ], [float(val[symb == "k" + str(i + 1)]) for i in np.arange(5)]
    data = {"HSU": {"Size": {"z": z}}, "Engine": {"Speed": EngSpeed.tolist(), "Torque": EngTorque.tolist(), "Power": EngPower.tolist()},
            "Parameters": {"h": h, "k": k, "e": e, "A": A,
                           "Bm": Bm, "Bp": Bp, "Cm": Cm, "Cp": Cp, "D": D},
            "Oil": {"mu": mu, "b": b}}
    if not os.path.isfile("data.json") or not os.path.isfile("fit.png"):
        fit()
    with open("data.json", "r") as read_file:
        inJson = json.load(read_file)
    inJson.update(data)
    with open("data.json", "w") as write_file:
        json.dump(inJson, write_file)
    return {"z": z, "h": h, "k": k, "e": e, "A": A, "Bm": Bm, "Bp": Bp,
            "Cm": Cm, "Cp": Cp, "D": D, "mu": mu, "b": b, "nE": EngSpeed, "tE": EngTorque, "pE": EngPower}


def catExceltoJson():
    """Reads the catalogue data from params.xlsx and saves it into dataCat.json."""
    with pd.ExcelFile("params.xlsx") as xls:
        cat = xls.parse("Catalogues")

    DPumps, nPumps, mPumps = np.array([]), np.array([]), np.array([])
    DMotors, nMotors, mMotors = np.array([]), np.array([]), np.array([])
    col = cat.columns.values
    man = np.array(['Rexroth', 'Danfoss', 'Eaton',
                    'Poclain', 'Linde', 'Kawasaki', 'Parker', 'Hydac', 'Brevini'])
    dataCat = {}
    dataCat["Pumps"] = {}
    dataCat["Motors"] = {}
    for i in np.arange(cat.columns.size):
        if "Pump" in cat.values[0, i] and "D, cc/rev" in cat.values[1, i]:
            DPumps, nPumps, mPumps = np.concatenate(
                (DPumps, cat.values[2:, i][~pd.isnull(cat.values[2:, i])])), np.concatenate(
                (nPumps, cat.values[2:, i + 1][~pd.isnull(cat.values[2:, i + 1])])), np.concatenate(
                (mPumps, cat.values[2:, i + 2][~pd.isnull(cat.values[2:, i + 2])]))
            for j in np.arange(len(man)):
                if man[j] in col[i]:
                    dataCat["Pumps"].update({man[j]: {"D": cat.values[2:, i][~pd.isnull(cat.values[2:, i])].tolist(),
                                                      "n": cat.values[2:, i + 1]
                                                      [~pd.isnull(
                                                          cat.values[2:, i + 1])].tolist(),
                                                      "m": cat.values[2:, i + 2][
                                                          ~pd.isnull(cat.values[2:, i + 2])].tolist()}})
        elif "Motor" in cat.values[0, i] and "D, cc/rev" in cat.values[1, i]:
            DMotors, nMotors, mMotors = np.concatenate(
                (DMotors, cat.values[2:, i][~pd.isnull(cat.values[2:, i])])), np.concatenate(
                (nMotors, cat.values[2:, i + 1][~pd.isnull(cat.values[2:, i + 1])])), np.concatenate(
                (mMotors, cat.values[2:, i + 2][~pd.isnull(cat.values[2:, i + 2])]))
            for j in np.arange(len(man)):
                if man[j] in col[i]:
                    dataCat["Motors"].update({man[j]: {"D": cat.values[2:, i][~pd.isnull(cat.values[2:, i])].tolist(),
                                                       "n": cat.values[2:, i + 1]
                                                       [~pd.isnull(
                                                           cat.values[2:, i + 1])].tolist(),
                                                       "m": cat.values[2:, i + 2][
                                                           ~pd.isnull(cat.values[2:, i + 2])].tolist()}})

    with open("dataCat.json", "w") as write_file:
        json.dump(dataCat, write_file)


def fit():
    """
    Loads the catalogue data either from dataCat.json or runs the catEXceltoJson() converter. Using the ordinary least square method implements the exponential regression model on the speed data and the linear regression on the mass data for pumps and motors. Saves the results of the regression, fit metrics, into data.json in form of coefficients, root-mean-square errors and R-squared of the model.
    """
    if not os.path.isfile("dataCat.json"):
        catExceltoJson()
    with open("dataCat.json", "r") as read_file:
        dataCat = json.load(read_file)
    DPumps, nPumps, mPumps = np.array([], dtype=float), np.array(
        [], dtype=float), np.array([], dtype=float)
    DMotors, nMotors, mMotors = np.array([], dtype=float), np.array(
        [], dtype=float), np.array([], dtype=float)
    man = np.array(['Rexroth', 'Danfoss', 'Eaton', 'Poclain',
                    'Linde', 'Kawasaki', 'Parker', 'Hydac', 'Brevini'])
    for i in np.arange(len(man)):
        DPumps = np.concatenate((DPumps, dataCat["Pumps"][man[i]]["D"]))
        nPumps = np.concatenate((nPumps, dataCat["Pumps"][man[i]]["n"]))
        mPumps = np.concatenate((mPumps, dataCat["Pumps"][man[i]]["m"]))
    for i in np.arange(len(man) - 1):
        DMotors = np.concatenate((DMotors, dataCat["Motors"][man[i]]["D"]))
        nMotors = np.concatenate((nMotors, dataCat["Motors"][man[i]]["n"]))
        mMotors = np.concatenate((mMotors, dataCat["Motors"][man[i]]["m"]))

    MassPump, MassMotor = sm.OLS(
        mPumps, DPumps).fit(), sm.OLS(mMotors, DMotors).fit()
    rmseMassPump, rmseMassMotor = np.sqrt(
        MassPump.mse_resid), np.sqrt(MassMotor.mse_resid)

    def regModel(x, a, b, c):
        return a * np.exp(-b * x) + c

    def res(a, b, c):
        return sum((regModel(DPumps, a, b, c) - nPumps) ** 2)

    guessSpeeds = [1e2, 1e-2, 1e2]
    cSpeedPumps, covSpeedPumps = optimize.curve_fit(
        regModel, DPumps, nPumps, guessSpeeds)
    cSpeedMotors, covSpeedMotors = optimize.curve_fit(
        regModel, DMotors, nMotors, guessSpeeds)
    rmseSpeedPump = np.sqrt(res(
        cSpeedPumps[0], cSpeedPumps[1], cSpeedPumps[2]) / (len(DPumps) - len(cSpeedPumps)))
    rmseSpeedMotor = np.sqrt(res(
        cSpeedMotors[0], cSpeedMotors[1], cSpeedMotors[2]) / (len(DMotors) - len(cSpeedMotors)))
    r2SpeedPumps = metrics.r2_score(
        regModel(DPumps, cSpeedPumps[0], cSpeedPumps[1], cSpeedPumps[2]), nPumps)
    r2SpeedMotors = metrics.r2_score(
        regModel(DMotors, cSpeedMotors[0], cSpeedMotors[1], cSpeedMotors[2]), nMotors)
    fit_metrics = {"Fits metrics": {"Coefficients": {"MassPump": float(MassPump.params), "MassMotor": float(MassMotor.params),
                                                     "SpeedPump": cSpeedPumps.tolist(), "SpeedMotor": cSpeedMotors.tolist()},
                                    "RMSE": {
        "MassPump": rmseMassPump, "MassMotor": rmseMassMotor, "SpeedPump": rmseSpeedPump,
        "SpeedMotor": rmseSpeedMotor},
        "R squared": {"MassPump": float(MassPump.rsquared), "MassMotor": float(MassMotor.rsquared),
                      "SpeedPump": r2SpeedPumps, "SpeedMotor": r2SpeedMotors}}}
    if os.path.isfile("data.json"):
        with open("data.json", "r") as read_file:
            data = json.load(read_file)
        data.update(fit_metrics)
        with open("data.json", "w") as write_file:
            json.dump(data, write_file)
    else:
        with open("data.json", "w") as write_file:
            json.dump(fit_metrics, write_file)
    DP = np.linspace(np.amin(DPumps), np.amax(DPumps))
    nPHatLow = regModel(
        DP, cSpeedMotors[0], cSpeedMotors[1], cSpeedMotors[2]) - rmseSpeedPump
    nPHat = regModel(DP, cSpeedMotors[0], cSpeedMotors[1], cSpeedMotors[2])
    nPHatUp = regModel(
        DP, cSpeedMotors[0], cSpeedMotors[1], cSpeedMotors[2]) + rmseSpeedPump
    plt.figure(num=1, dpi=200)
    plt.plot(DP, nPHat, label="Rated speed",
             linewidth=0.75, linestyle="dashed")
    plt.plot(DP, nPHatLow, label="Lower rated speed",
             linewidth=0.75, linestyle="dashed")
    plt.plot(DP, nPHatUp, label="Upper rated speed",
             linewidth=0.75, linestyle="dashed")
    markerStyle = ['o', 'v', '^', '<', '>', 'h', 's', 'p', '*']
    colours = cm.Paired(np.linspace(0, 1, len(markerStyle)))
    for i in np.arange(len(man)):
        plt.scatter(dataCat["Pumps"][man[i]]["D"],
                    dataCat["Pumps"][man[i]]["n"], s=10,
                    label=man[i], marker=markerStyle[i])
    plt.grid(True, 'both', 'both', linewidth=0.1, linestyle='dashed')
    plt.xlabel('Pump displacement, cc/rev')
    plt.ylabel('Pump speed, rpm')
    plt.legend()
    plt.savefig("fit.png")
    plt.cla()


def sizing(V, aux, sa=18):
    '''
    Implements sizing of the pumping group of an axial piston pump. Reads from data.json the engine torque curve in order to calculate the rated speed. Saves to data.json.

    Parameters
    ---
    - V: float
        The displacement of an axial-piston machine in cc/rev.

    - aux: dict
        The dictionary of auxiliary parameters imported with .imp().

    - sa: float
        The maximum swash angle of the axial piston machine in degrees, default sa = 18 degrees.

    Returns
    ---
    - dict{V": float, "sa": float, "d": float, "D": float, "h": float, "eng": float,
           "rbo": float, "Rbo": float, "Rbi": float, "rbi": float, "rs": float, "Rs": float}

        The dictionary of the main sizes of the pumping group.

    '''
    d = float((4 * V * 1e-6 * aux["k"][0] / (aux["z"]
                                             ** 2 * np.tan(np.radians(sa)))) ** (1 / 3))
    Ap = np.pi * d ** 2 / 4
    pcd = float(aux["z"] * d / (np.pi * aux["k"][0]))
    h = float(pcd * np.tan(np.radians(sa)))
    eng = 1.4 * d
    Ak = float(aux["k"][2] * Ap)
    w = float(2 * (np.sqrt(d ** 2 + (np.pi - 4) * Ak) - d) / (np.pi - 4))
    t = float(aux["k"][1] * aux["z"] * Ap / (np.pi * pcd) - w)
    rbo = (pcd + w) / 2
    Rbo = rbo + t
    Rbi = (pcd - w) / 2
    rbi = Rbi - t
    As = float(aux["k"][3] * Ap / np.cos(np.radians(sa)))
    Rs = float(np.pi * pcd * aux["k"][4] / (2 * aux["z"]))
    rs = float(np.sqrt(Rs ** 2 - As / np.pi))
    out = {"V": V, "sa": sa, "d": d, "D": pcd, "h": h, "eng": eng,
           "rbo": rbo, "Rbo": Rbo, "Rbi": Rbi, "rbi": rbi, "rs": rs, "Rs": Rs}
    with open("data.json", "r") as read_file:
        data = json.load(read_file)
    cSpeedPump = data["Fits metrics"]["Coefficients"]["SpeedPump"]
    nLimLow = cSpeedPump[0] * np.exp(-cSpeedPump[1] * V) + \
        cSpeedPump[2] - data["Fits metrics"]["RMSE"]["SpeedPump"]
    nLim = cSpeedPump[0] * np.exp(-cSpeedPump[1] * V) + cSpeedPump[2]
    nLimUp = cSpeedPump[0] * np.exp(-cSpeedPump[1] * V) + \
        cSpeedPump[2] + data["Fits metrics"]["RMSE"]["SpeedPump"]
    data["HSU"]["Size"].update(out)
    data["HSU"].update({"Lower rated speed": nLimLow,
                        "Rated speed": nLim, "Upper rated speed": nLimUp})
    with open("data.json", "w") as write_file:
        json.dump(data, write_file)
    return out


def efficiency(s, n, p, aux):
    """
    Calculates efficiencies and performance characteristics of the HSU for the given pump speed, pressures and size of the consituent same-displacement axial-piston machines.

    Parameters
    ---
    - s: dict{V": float, "sa": float, "d": float, "D": float, "h": float, "eng": float,"rbo": float, "Rbo": float, "Rbi": float, "rbi": float, "rs": float, "Rs": float}
        The dictionary of the main sizes of the pumping group calculated fromv sizing(V, aux).

    - n: float
        The HSU input, or pump, speed in rpm.

    - p: list
        The list containing the charge p[0] and discharge p[1] pressures in bar.

    - aux: dict
        The dictionary of auxiliary parameters imported with .imp().

    Returns
    ---
    - dict{"p": float, "nIn": float,"nOut": float, "tIn": float, "tOut": float, "powIn": float, "powOut": float, "effHSU": float, "effP": float, "effM": float, "mechP": float}

        The dictionary of the key performance indicators of the HSU for the specific steady state condition defined by the Parameters.
    """
    n = float(n)
    ql1 = float((np.pi * aux["h"][0] ** 3 * 0.5 * (p[1] * 1e5 + p[0] * 1e5) * (
        1 / np.log(s["Rbo"] / s["rbo"]) + 1 / np.log(s["Rbi"] / s["rbi"])) / (6 * aux["mu"])))
    ql2 = float((aux["z"] * np.pi * aux["h"][1] ** 3 * 0.5 * (p[1] *
                                                              1e5 + p[0] * 1e5) / (
        6 * aux["mu"] * np.log(s["Rs"] / s["rs"]))))
    ql3i = np.array([aux["z"] * np.pi * s["d"] * aux["h"][2] ** 3 * 0.5 * (p[1] * 1e5 + p[0] * 1e5) * (
        1 + 1.5 * aux["e"] ** 3) * (1 / (s["eng"] + s["h"] * np.sin(np.pi * (ii) / aux["z"]))) / (12 * aux["mu"])
        for ii in np.arange(aux["z"])])
    ql3 = float(sum(ql3i))
    ql = sum((ql1, ql2, ql3))
    qth = float(n * s["V"] / 6e7)
    volP = float((1 - (p[1] - p[0]) / aux["b"] - ql / qth) * 100)
    volM = float((1 - ql / qth) * 100)
    volHSU = float(volP * volM * 1e-2)
    mechP = float((1 - aux["A"] * np.exp(
        -aux["Bp"] * aux["mu"] * 1e3 * n / (s["sa"] * (p[1] * 1e5 - p[0] * 1e5) * 1e-5)) - aux["Cp"] * np.sqrt(
        aux["mu"] * 1e3 * n / (s["sa"] * (p[1] * 1e5 - p[0] * 1e5) * 1e-5)) - aux["D"] / (
        s["sa"] * (p[1] * 1e5 - p[0] * 1e5) * 1e-5)) * 100)
    mechM = float((1 - aux["A"] * np.exp(
        -aux["Bm"] * aux["mu"] * 1e3 * n * volHSU * 1e-2 / (s["sa"] * (p[1] * 1e5 - p[0] * 1e5) * 1e-5)) - aux[
        "Cm"] * np.sqrt(
        aux["mu"] * 1e3 * n * volHSU * 1e-2 / (s["sa"] * (p[1] * 1e5 - p[0] * 1e5) * 1e-5)) - aux["D"] / (
        s["sa"] * (p[1] * 1e5 - p[0] * 1e5) * 1e-5)) * 100)
    mechHSU = mechP * mechM * 1e-2
    totP = volP * mechP * 1e-2
    totM = volM * mechM * 1e-2
    totHSU = totP * totM * 1e-2
    tIn = float((p[1] * 1e5 - p[0] * 1e5) * s["V"]
                * 1e-6 / (2 * np.pi * mechP * 1e-2))
    tOut = float((p[1] * 1e5 - p[0] * 1e5) * s["V"] * 1e-6 /
                 (2 * np.pi * mechP * 1e-2) * (mechHSU * 1e-2))
    powIn = tIn * n * np.pi / 30 * 1e-3
    powOut = powIn * totHSU * 1e-2
    nOut = n * volHSU * 1e-2
    return {"p": float(p[1]), "nIn": n, "nOut": nOut, "tIn": tIn, "tOut": tOut, "powIn": powIn, "powOut": powOut,
            "effHSU": totHSU, "effP": totP, "effM": totM, "mechP": mechP}


def plotMap(nmax, pmax, s, aux, nmin=1000, pmin=100, charge=25, res=100, gr=0.75, maxPower=682):
    """
    Plots and saves the HSU efficiency map.

    Parameters
    ---
    - nmax: float
        The upper limit of the input (pump) speed range on the map in rpm.

    - pmax: float
        The upper limit of the discharge pressure range on the map in bar.

    - s: dict
        The dictionary containing the results of the sizing().

    - aux: dict
        The dictionary of auxiliary parameters imported with imp()

    - nmin: float
        The lower limit of the input speed range on the map in rpm, default nmin = 1000 rpm.

    - pmin: float
        The lower limit of the discharge pressure range on the map in bar, default pmin = 100 bar.

    - charge: float
        The charge pressure level in bar, default charge = 25 bar.

    - res: float
        The resolution of the map. The number of efficiency samples calculated per axis, default = 100.

    - gr: float
        The reduction ratio of a gear train connecting the engine with the HSU input, or pump, shaft

    - maxPower: float
        The max input power available to the HSU in kW, default = 682 kW.

    """
    n = np.linspace(nmin, nmax, res)
    p = np.linspace(pmin, pmax, res)
    eff = [[efficiency(s, n[i], [charge, p[j]], aux)["effHSU"]
            for i in np.arange(len(n))] for j in np.arange(len(p))]
    effPM = [[efficiency(s, n[i], [charge, p[j]], aux)["mechP"]
              for i in np.arange(len(n))] for j in np.arange(len(p))]
    tin = s["V"] * 1e-6 * (p - charge) * 1e5 / \
        (2 * np.pi * np.amax(effPM, axis=1) * 1e-2)
    nHSU = aux["nE"] * gr
    nPivot = nHSU[aux["nE"] == 2700]
    nMaxSpeed = nHSU[aux["nE"] == 2900]
    nMaxTorque = nHSU[aux["nE"] == 2100]
    pPivot = maxPower * 1e3 * 30 / np.pi / nPivot * 2 * np.pi / \
        s["V"] / 1e-6 / 1e5 * np.amax(effPM) * 1e-2 + charge
    pMaxSpeed = aux["tE"][aux["nE"] == 2900] / gr * 2 * np.pi / \
        s["V"] / 1e-6 / 1e5 * np.amax(effPM) * 1e-2 + charge
    pMaxTorque = aux["tE"][aux["nE"] == 2100] / gr * 2 * np.pi / s["V"] / 1e-6 / 1e5 * effPM[int([i for i, x in enumerate(
        nHSU == nMaxTorque) if x][0])][int([np.argmin(abs(tin / np.amax(aux["tE"] / gr) - 1))][0])] * 1e-2 + charge
    effPivot = efficiency(s, nPivot, [charge, pPivot], aux)
    effMaxSpeed = efficiency(s, nMaxSpeed, [charge, pMaxSpeed], aux)
    effMaxTorque = efficiency(s, nMaxTorque, [charge, pMaxTorque], aux)
    with open("data.json", "r") as read_file:
        data = json.load(read_file)
    data["HSU"]["Pivot"], data["HSU"]["Max Speed"], data["HSU"]["Max Torque"] = effPivot, effMaxSpeed, effMaxTorque
    nLimLow, nLim, nLimUp = data["HSU"]["Lower rated speed"], data["HSU"]["Rated speed"], data["HSU"]["Upper rated speed"]
    data["Engine"].update({"Input gear ratio": gr})
    with open("data.json", "w") as write_file:
        json.dump(data, write_file)
    levels = np.linspace(50, 100, num=100 - 50 + 1, dtype=int)
    colours = cm.viridis(np.linspace(0, 1, len(levels)))
    colours = colours[colours != 1].reshape((len(levels), 3))
    plt.figure(num=2, dpi=200)
    cs = plt.contour(n, tin, eff, levels, linewidths=0.7,
                     linestyles="dashed", colors=colours)
    plt.plot(nHSU, aux["tE"] / gr, linewidth=0.9,
             label='Available engine torque')
    plt.plot([nLimLow, nLimLow], [np.amin(tin), np.amax(tin)],
             linestyle='dashed', label="Lower rated speed", linewidth=0.9)
    plt.plot([nLim, nLim], [np.amin(tin), np.amax(tin)],
             linestyle='dashed', label="Rated speed", linewidth=0.9)
    plt.plot([nLimUp, nLimUp], [np.amin(tin), np.amax(tin)],
             linestyle='dashed', label="Upper rated speed", linewidth=0.9)
    plt.plot(n, maxPower * 1e3 * 30 / (np.pi * n), linewidth=0.9,
             label='Torque at ' + str(maxPower) + ' kW')
    plt.scatter([effPivot["nIn"], effMaxSpeed["nIn"], effMaxTorque["nIn"]], [
        effPivot["tIn"], effMaxSpeed["tIn"], effMaxTorque["tIn"]], s=10)
    plt.clabel(cs, levels, fmt='%1.0f', fontsize=5,
               inline=1, inline_spacing=1, colors='k')
    plt.grid(True, 'both', 'both', linewidth=0.1, linestyle='dashed')
    plt.xlabel('Speed, rpm')
    plt.ylabel('Torque, Nm')
    plt.title('Efficiency map of HSU ' +
              str(s["V"]) + '.' + str(s["sa"]) + '.' + str(gr))
    plt.axis([nmin, nmax, np.amin(tin), np.amax(tin)])
    plt.legend()
    plt.tight_layout()
    plt.savefig('eff map {}.{}.{}.png'.format(s["V"], s["sa"], gr))
    plt.cla()


if __name__ == "__main__":
    aux = imp()
    s = sizing(440, aux)
    plotMap(2300, 620, s, aux)

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.figure_factory as ff
from joblib import dump, load
from effmap.hst import HST
from effmap.regressor import Regressor
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold, train_test_split, cross_validate


def plot_validation():
    """Performs validation of the efficiency model by computing the volumetric efficiency of the benchmark HST, for which test data is available containing results of measurements of the input/output speeds, and comparing the computed efficiency with the test data.

    Returns:
        fig: plotly figure object
    """
    data = pd.read_csv('.\\data\\test_data.csv')
    data.dropna(subset=['Forward Speed', 'Reverse Speed',
                        'Volumetric at 1780RPM'], inplace=True)
    speeds = data[['Forward Speed', 'Reverse Speed']].astype(float)
    speeds = speeds.stack()
    vol_eff = speeds / 1780 * 1e2
    piston_max = 1.1653 * 25.4 * 1e-3
    piston_min = 1.1650 * 25.4 * 1e-3
    bore_max = 1.1677 * 25.4 * 1e-3
    bore_min = 1.1671 * 25.4 * 1e-3
    rad_clearance_max = (bore_max - piston_min) / 2
    rad_clearance_min = (bore_min - piston_max) / 2
    benchmark = HST(disp=196, swash=15, oil='SAE 30', oil_temp=60)
    benchmark.compute_sizes(k1=.7155, k2=.9017, k3=.47, k4=.9348, k5=.9068)
    eff_min = benchmark.compute_eff(
        speed_pump=1780, pressure_discharge=207, pressure_charge=14, h3=rad_clearance_max)
    eff_max = benchmark.compute_eff(
        speed_pump=1780, pressure_discharge=207, pressure_charge=14, h3=rad_clearance_min)
    fig = ff.create_distplot([vol_eff], ['Test data'],
                             show_hist=True, bin_size=.3, show_rug=False)
    fig.add_scatter(
        x=[eff_max['hst']['volumetric'], eff_max['hst']['volumetric']],
        y=[0, .6],
        mode='lines',
        name='Prediciton. Min clearance',
        line=dict(
            width=1.5,
        ),
    )
    fig.add_scatter(
        x=[eff_min['hst']['volumetric'], eff_min['hst']['volumetric']],
        y=[0, .6],
        mode='lines',
        name='Prediciton. Max clearance',
        line=dict(
            width=1.5,
        ),
    )
    fig.add_scatter(
        x=[vol_eff.mean(), vol_eff.mean()],
        y=[0, .6],
        mode='lines',
        name='Test mean',
        line=dict(
            width=1.5,
            dash='dash'),
    )
    fig.add_scatter(
        x=[vol_eff.mean()+vol_eff.std(), vol_eff.mean()+vol_eff.std()],
        y=[0, .6],
        mode='lines',
        name='Test mean + STD',
        line=dict(
            width=1.5,
            dash='dash'),
    )
    fig.add_scatter(
        x=[vol_eff.mean()-vol_eff.std(), vol_eff.mean()-vol_eff.std()],
        y=[0, .6],
        mode='lines',
        name='Test mean - STD',
        line=dict(
            width=1.5,
            dash='dash'),
    )
    fig.update_layout(
        title=f'Sample of {len(vol_eff)} measurements of the {benchmark.displ} cc/rev HST with {benchmark.oil} at {benchmark.oil_temp}C',
        width=800,
        height=500,
        xaxis=dict(
            title='HST volumetric efficiency, %',
            showline=True,
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.5,
            range=[84, 94],
            dtick=2,
        ),
        yaxis=dict(
            title='Probability density',
            showline=True,
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.5,
            range=[0, .6],
        ),
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,0)',
        showlegend=True,
        legend_orientation='h',
        legend=dict(x=0, y=-.15))
    return fig


def fit_catalogues(data_in):
    models = {}
    if not os.path.exists('models'):
        os.mkdir('models')
    for machine_type in ('pump', 'motor'):
        for data_type in ('speed', 'mass'):
            data = data_in[data_in['type'] ==
                           f'{machine_type.capitalize()}']
            model = Regressor(
                machine_type=machine_type,
                data_type=data_type)
            x_full = data['displacement'].to_numpy(dtype='float64')
            y_full = data[data_type].to_numpy(dtype='float64')
            x_train, x_test, y_train, y_test = train_test_split(
                x_full, y_full, test_size=0.2, random_state=0)
            strat_k_fold = KFold(
                n_splits=5, shuffle=True,
                random_state=42)
            cv_results = cross_validate(
                model, x_train, y_train,
                cv=strat_k_fold, scoring=['neg_root_mean_squared_error', 'r2'], return_estimator=True, n_jobs=-1, verbose=0)
            model.r2_ = np.mean([k for k in cv_results['test_r2']])
            model.cv_rmse_ = - np.mean(
                [k for k in cv_results['test_neg_root_mean_squared_error']])
            model.cv_r2std_ = np.std([k for k in cv_results['test_r2']])
            model.cv_rmsestd_ = np.std(
                [k for k in cv_results['test_neg_root_mean_squared_error']])
            model.coefs_ = np.mean(
                [k.coefs_ for k in cv_results['estimator']], axis=0)
            model.test_rmse_, model.test_r2_ = model.eval(x_test, y_test)
            model.fitted_ = True
            dump(model, os.path.join(
                'models', f'{machine_type}_{data_type}.joblib'))
            models['_'.join((machine_type, data_type))] = model
    return models


def plot_catalogues(models, data_in):
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                        shared_yaxes=True, vertical_spacing=0.1, horizontal_spacing=0.07,
                        )
    for i, j in enumerate(models):
        model = models[j]
        data_type = model.data_type
        machine_type = model.machine_type
        data = data_in[data_in['type'] == f'{machine_type.capitalize()}']
        x = data['displacement'].to_numpy(dtype='float64')
        x_cont = np.linspace(.2 * np.amin(x),
                             1.2 * np.amax(x), num=100)
        for l in zip(('Regression model', 'Upper limit', 'Lower limit'), (0, model.test_rmse_, -model.test_rmse_)):
            fig.add_scatter(
                x=x_cont,
                y=model.predict(x_cont)+l[1],
                mode='lines',
                name=l[0],
                line=dict(
                    width=1,
                    dash='dash'),
                row=1 - (-1)**(i)//2,
                col=(i + 2) // 2
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
                row=1 - (-1)**(i)//2,
                col=(i + 2) // 2
            )
        fig.update_xaxes(
            title_text=f'{machine_type.capitalize()} displacement, cc/rev',
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.5,
            range=[0, round(1.1 * max(data['displacement']), -2)],
            row=1 - (-1)**(i)//2,
            col=(i + 2) // 2
        )
        fig.update_yaxes(
            title_text=f'{data_type.capitalize()}, rpm' if data_type == 'speed' else f'{data_type.capitalize()}, kg',
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.5,
            range=[0, round(1.2 * max(data[data_type]), -2)] if data_type == 'mass' else [
                round(.7 * min(data[data_type]), -2), round(1.2 * max(data[data_type]), -2)],
            row=1 - (-1)**(i)//2,
            col=(i + 2) // 2
        )
    fig.update_layout(
        width=800,
        height=800,
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,0)',
        showlegend=False,
    )
    return fig


def set_sidebar():
    """Assigns the default values of oil, its paramteres and initial design paramteres to initialize a HST to plot its efficiency map.

    Returns:
        oil: string, default 'SAE 15W40'
        oil_temp: int, default 100
        max_displ: int, default 440
        max_power: int, default 685
        gear_ratio: float, default .75
        max_speed: int, default 2400
        max_pressure: int, default 650
        pressure_lim: int, default 480
    """
    st.sidebar.markdown('Setting up the efficiency map')
    oil = st.sidebar.selectbox('Select an oil:',
                               ('SAE 15W40', 'SAE 10W40', 'SAE 10W60', 'SAE 5W40', 'SAE 0W30', 'SAE 30'))
    oil_temp = st.sidebar.slider(
        'Select oil temperature', min_value=0, max_value=100, value=100, step=10)
    max_displ = st.sidebar.slider(
        'Select a displacement', min_value=100, max_value=800, value=440, step=5)
    max_power = st.sidebar.slider(
        'Select a max power', min_value=400, max_value=800, value=685, step=5)
    gear_ratio = st.sidebar.slider(
        'Select an input gear ratio', min_value=.5, max_value=2., value=.75, step=.25)
    max_speed = st.sidebar.slider(
        'Select a max speed', min_value=1000, max_value=4000, value=2400, step=100)
    max_pressure = st.sidebar.slider(
        'Select the max pressure', min_value=100, max_value=800, value=650, step=50)
    pressure_lim = st.sidebar.slider(
        'Select the pressure limiter setting', min_value=300, max_value=800, value=480, step=10)
    return oil, oil_temp, max_displ, max_power, gear_ratio, max_speed, max_pressure, pressure_lim


def main():
    """Runs through the funcionality of the Regressor and HST classes."""
    data = pd.read_csv('.\\data\\data.csv', index_col='#')
    if os.path.exists('.\\models') and len(os.listdir('.\\models')):
        models = {}
        for file in os.listdir('.\\models'):
            models[file[:-7]] = load(os.path.join(os.getcwd(), 'models', file))
    else:
        models = fit_catalogues(data)
    st.title('Catalogue data and regressions')
    st.write(plot_catalogues(models, data))
    for i in models:
        units = 'rpm' if 'speed' in i else 'kg'
        st.write(
            f'RMSE {models[i].machine_type} {models[i].data_type} = {np.round(models[i].test_rmse_, decimals=2)}',
            u'\u00B1', f'{np.round(models[i].cv_rmsestd_,decimals=2)}', units
        )
    oil, oil_temp, max_displ, max_power, gear_ratio, max_speed, max_pressure, pressure_lim = set_sidebar()
    hst = HST(max_displ, oil=oil, oil_temp=oil_temp, max_power_input=max_power,
              input_gear_ratio=gear_ratio)
    hst.compute_sizes()
    hst.compute_speed_limit(models['pump_speed'])
    hst.add_no_load((1800, 140), (2025, 180))
    hst.compute_loads(pressure_lim)
    st.title('Efficiency map')
    st.write(hst.plot_eff_maps(max_speed, max_pressure, pressure_lim=pressure_lim,
                               show_figure=False,
                               save_figure=False))
    st.title('Physical properties of oil')
    st.write(hst.plot_oil())
    st.title('Validation of the HST efficiency model')
    st.write(plot_validation())


if __name__ == '__main__':
    main()

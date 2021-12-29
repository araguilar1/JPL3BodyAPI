
"""
Script to query JPL Horizons for the initial conditions of periodic 3-body orbits, their families, and other data.

"""
import requests
import json
import argparse
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"
pio.renderers.default = "browser"

def queryJPL(sys="earth-moon", family="halo", libr="2", branch="N", periodmin="", periodmax="", periodunits="", jacobimin="", jacobimax="", stabmin="", stabmax=""):
    """
    Get information from the JPL Horizons periodic 3-body orbits API.

    Args:
        sys (str): three-body system defined in lower-case as “primary-secondary,” e.g. earth-moon, mars-phobos, sun-earth.
        family (str): name of the orbit family: halo,vertical,axial,lyapunov,longp,short,butterfly,dragonfly,resonant,dro,dpo,lpo
        libr (str): libration point. Required for lyapunov,halo (1,2,3), longp, short (4,5), and axial,vertical (1,2,3,4,5).
        branch (str): branch of orbits within the family: N/S for halo,dragonfly,butterfly, E/W for lpo, and pq integer sequence for resonant (e.g., 12 for 1:2).
        periodmin (str): minimum period (inclusive). Units defined by periodunits.
        periodmax (str): maximum period (inclusive). Units defined by periodunits.
        periodunits (str): units of pmin and pmax: s for seconds, h for hours, d for days, TU for nondimensional.
        jacobimin (str): minimum Jacobi constant (inclusive). Nondimensional units.
        jacobimax (str): maximum Jacobi constant (inclusive). Nondimensional units.
        stabmin (str): minimum stability index (inclusive).
        stabmax (str): maximum stability index (inclusive).

    Returns:
        query: JSON object containing the requested data.
    """
    # Inputs
    args = locals()

    # Define the base URL for the JPL Horizons API.
    baseUrl = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api?"
    for key in args:
        if args[key]:
            baseUrl += "{}={}&".format(key, args[key])

    baseUrl = baseUrl[:-1]

    # Get the data from the JPL Horizons API.
    r = requests.get(baseUrl)
    data = r.text
    # Check response status
    if r.status_code != 200:
        print("Error: {}".format(r.status_code))
        return None
    else:
        return data

def parseData(data):
    """
    Parses the JSON data returned by queryJPL functions

    Args:
        json: json struct returned by queryJPLY


    Returns:
        sysDict (dict): dictionary of system data 
            primary (str): primary body
            secondary (str): secondary body
            family (str): family of orbits
            mu (float): mass parameter of the system
            lpoints (list, float): list of libration point locations
        ics (list, float): list of initial conditions
    """
    data = json.loads(data)

    # Get the System Information
    system = data['system']['name']
    primary, secondary = system.split('-')
    fam = data['family']
    libpoint = data['libration_point']
    mu = float(data['system']['mass_ratio'])

    # Get the Lagrange Points
    lpoints = [data['system']['L{}'.format(i)] for i in range(1,6)]
    lpoints = [list(map(float, pos)) for pos in lpoints]

    # Build Dict of system information
    sysDict = {'prim': primary, 'sec': secondary,'fam': fam, 'libpoint': libpoint,'mu': mu, 'lpoints': lpoints}

    # Get the initial conditions of the orbits.
    initial_conditions = data['data']
    ics = [list(map(float, ic)) for ic in initial_conditions]

    return sysDict, ics

def cr3bp_ode(y_, t, mu):
    """
    CR3BP EOMs
    """
    yd_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Position
    x = y_[0]
    y = y_[1]
    z = y_[2]
    # Velocity
    vx = y_[3]
    vy = y_[4]
    vz = y_[5]

    d = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r = np.sqrt((x + mu - 1) ** 2 + y ** 2 + z ** 2)

    # Velocity
    yd_[0] = y_[3]
    yd_[1] = y_[4]
    yd_[2] = y_[5]

    # Eq. of Motion
    yd_[3] = -(1 - mu) * (x + mu) / (d ** 3) - mu * (x - 1 + mu) / (r ** 3) + 2 * vy + x
    yd_[4] = -(1 - mu) * y / (d ** 3) - mu * y / (r ** 3) - 2 * vx + y
    yd_[5] = -(1 - mu) * z / (d ** 3) - mu * z / (r ** 3)

    return yd_

def propagate(mu, ics, n=5):
    """
    Propagates the periodic orbits from the JPL query

    Args:
        mu (float): mass parameter of the system
        ics (list, float): list of initial conditions
        n (int): Propagates every nth initial condition to reduce computational time

    Returns:
        trajList (list): list of propagated trajectories
    """
    if n > 0:
        ics = ics[::n]

    trajList = []
    ODE = lambda t, y_: cr3bp_ode(y_, t, mu)
    for ic in ics:
        # Initial Conditions
        y0 = np.array(ic[:6])
        # Time
        tspan = [0, ic[7]]
        t = np.linspace(tspan[0], tspan[1], 500)
        # Solve ODE
        sol = solve_ivp(ODE, tspan, y0, method='DOP853', atol=1e-12, rtol=1e-12, t_eval=t)
        # Append to trajectory list
        trajList.append(sol.y)

    return trajList

def plotTrajs(sysDict, trajList, savefig=False):
    """
    Plots a family of periodic orbits.
    """
    data = []
    layout = go.Layout(showlegend=True, scene_aspectmode='data')
    for traj in trajList:
        data.append(
            go.Scatter3d(
                x=traj[0, :],
                y=traj[1, :],
                z=traj[2, :],
                mode='lines',
                showlegend=False,
            )
        )
    # for lpoint in sysDict['lpoints']:
    #     data.append(
    #         go.Scatter3d(
    #             x=[lpoint[0]],
    #             y=[lpoint[1]],
    #             z=[lpoint[2]],
    #             mode='markers',
    #             marker=dict(size=2, color='lightblue'),
    #             showlegend=False,
    #         )
    #     )
    fig = go.Figure(data=data, layout=layout)
    if savefig:
        fig.write_html('./{}-{}_{}{}.html'.format(sysDict['prim'], sysDict['sec'], sysDict['libpoint'], sysDict['fam']))
    fig.show()
    
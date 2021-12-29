
"""
Script to query JPL Horizons for the initial conditions of periodic 3-body orbits, their families, and other data.

"""
import requests
import json
import argparse
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio

def queryJPL(sys="earth-moon", family="halo", libr="2", branch="N", periodmin="", periodmax="", periodunit="", jacobimin="", jacobimax="", stabmin="", stabmax=""):
    """
    Get information from the JPL Horizons periodic 3-body orbits API.

    Args:
        sys (str): three-body system defined in lower-case as “primary-secondary,” e.g. earth-moon, mars-phobos, sun-earth.
        family (str): name of the orbit family: halo,vertical,axial,lyapunov,longp,short,butterfly,dragonfly,resonant,dro,dpo,lpo
        libr (str): libration point. Required for lyapunov,halo (1,2,3), longp, short (4,5), and axial,vertical (1,2,3,4,5).
        branch (str): branch of orbits within the family: N/S for halo,dragonfly,butterfly, E/W for lpo, and pq integer sequence for resonant (e.g., 12 for 1:2).
        periodmin (str): minimum period (inclusive). Units defined by periodunits.
        periodmax (str): maximum period (inclusive). Units defined by periodunits.
        periodunit (str): units of pmin and pmax: s for seconds, h for hours, d for days, TU for nondimensional.
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
        primary (str): name of the primary body
        secondary (str): name of the secondary body
        mu (float): mass parameter of the system
        lpoints (list. float): list of lagrange point positions
        ics (list, float): list of initial conditions
    """
    data = json.loads(data)

    system = data['system']['name']
    primary, secondary = system.split('-')
    mu = float(data['system']['mass_ratio'])
    
    # Get the Lagrange Points
    lpoints = [data['system']['L{}'.format(i)] for i in range(1,6)]

    # Get the initial conditions of the orbits.
    initial_conditions = data['data']
    ics = [list(map(float, ic)) for ic in initial_conditions]
    sysDict = {'prim': primary, 'sec': secondary, 'mu': mu, 'lpoints': lpoints}
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

    # Eq. of Motion
    d = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r = np.sqrt((x + mu - 1) ** 2 + y ** 2 + z ** 2)

    # Velocity
    yd_[0] = y_[3]
    yd_[1] = y_[4]
    yd_[2] = y_[5]

    # Acceleration
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
        n (int): Takes every nth initial condition to reduce computational time

    Returns:
        trajList (list): list of propagated trajectories
    """
    if n > 0:
        ics = ics[::n]

    trajList = []
    ODE = lambda t, y_: cr3bp_ode(y_, t, mu)
    for ic in ics:
        # Initial Conditions
        y0 = np.array([ic[0], ic[1], ic[2], ic[3], ic[4], ic[5]])
        # Time
        tspan = [0, ic[7]]
        # Solve ODE
        sol = solve_ivp(ODE, tspan, y0, method='DOP853', atol=1e-12, rtol=1e-12)
        # Append to trajectory list
        trajList.append(sol.y)

    return trajList

def plotTrajs(sysDict, trajList, name='plot'):
    """
    Plots a family of periodic orbits.
    """
    data = []
    layout = go.Layout(showlegend=True)
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
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(scene_aspectmode='data')
    fig.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sys", type=str, default="earth-moon", help="three-body system defined in lower-case as “primary-secondary,” e.g. earth-moon, mars-phobos, sun-earth.")
    parser.add_argument("--family", type=str, default="halo", help="name of the orbit family: halo,vertical,axial,lyapunov,longp,short,butterfly,dragonfly,resonant,dro,dpo,lpo")
    parser.add_argument("--libr", type=str, default="2", help="libration point. Required for lyapunov,halo (1,2,3), longp, short (4,5), and axial,vertical (1,2,3,4,5).")
    parser.add_argument("--branch", type=str, default="N", help="branch of orbits within the family: N/S for halo,dragonfly,butterfly, E/W for lpo, and pq integer sequence for resonant (e.g., 12 for 1:2).")
    parser.add_argument("--periodmin", type=str, default="", help="minimum period (inclusive). Units defined by periodunits.")
    parser.add_argument("--periodmax", type=str, default="", help="maximum period (inclusive). Units defined by periodunits.")
    parser.add_argument("--periodunit", type=str, default="", help="units of pmin and pmax: s for seconds, h for hours, d for days, TU for nondimensional.")
    parser.add_argument("--jacobimin", type=str, default="", help="minimum Jacobi constant (inclusive). Nondimensional units.")
    parser.add_argument("--jacobimax", type=str, default="", help="maximum Jacobi constant (inclusive). Nondimensional units.")
    parser.add_argument("--stabmin", type=str, default="", help="minimum stability index (inclusive).")
    parser.add_argument("--stabmax", type=str, default="", help="maximum stability index (inclusive).")


    args = parser.parse_args()

    data = queryJPL(**vars(args))
    sysDict, ics = parseData(data)
    mu = sysDict['mu']

    trajList = propagate(mu, ics, n=10)
    plotTrajs(sysDict, trajList)

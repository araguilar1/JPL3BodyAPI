
"""
Script to query JPL Horizons for the initial conditions of periodic 3-body orbits, their families, and other data.

"""
import requests
import json
import argparse

def queryJPL(sys="earth-moon", family="halo", libr="2", branch="N", periodmin="", periodmax="", periodunit="", jacobimin="", jacobimax="", stabmin="", stabmax=""):
    """
    Get information from the JPL Horizons periodic 3-body orbits API.

    Args:
        sys (str): three-body system defined in lower-case as “primary-secondary,” e.g. earth-moon, mars-phobos, sun-earth.
        family (str): name of the orbit family: halo,vertical,axial,lyapunov,longp,short,butterfly,dragonfly,resonant,dro,dpo,lpo
        libr (int): libration point. Required for lyapunov,halo (1,2,3), longp, short (4,5), and axial,vertical (1,2,3,4,5).
        branch (str): branch of orbits within the family: N/S for halo,dragonfly,butterfly, E/W for lpo, and pq integer sequence for resonant (e.g., 12 for 1:2).
        periodmin (float): minimum period (inclusive). Units defined by periodunits.
        periodmax (float): maximum period (inclusive). Units defined by periodunits.
        periodunit (str): units of pmin and pmax: s for seconds, h for hours, d for days, TU for nondimensional.
        jacobimin (float): minimum Jacobi constant (inclusive). Nondimensional units.
        jacobimax (float): maximum Jacobi constant (inclusive). Nondimensional units.
        stabmin (float): minimum stability index (inclusive).
        stabmax (float): maximum stability index (inclusive).

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
        System (str): information about the 3-body system
        tbd:
    """
    data = json.loads(data)

    system = data['system']['name']
    primary, secondary = system.split('-')
    mu = data['system']['mass_ratio']
    
    initial_conditions = data['data']

    return primary, secondary, mu, initial_conditions

def plotFamily():
    """
    Plots a family of periodic orbits.
    """

if __name__ == "__main__":
    data = queryJPL()
    prim, sec, mu, ics = parseData(data)

    print(ics)
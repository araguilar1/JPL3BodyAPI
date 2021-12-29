from jpl3bodyAPI import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys", type=str, default="earth-moon", help="three-body system defined in lower-case as “primary-secondary,” e.g. earth-moon, mars-phobos, sun-earth.")
    parser.add_argument("--family", type=str, default="halo", help="name of the orbit family: halo,vertical,axial,lyapunov,longp,short,butterfly,dragonfly,resonant,dro,dpo,lpo")
    parser.add_argument("--libr", type=str, default="2", help="libration point. Required for lyapunov,halo (1,2,3), longp, short (4,5), and axial,vertical (1,2,3,4,5).")
    parser.add_argument("--branch", type=str, default="N", help="branch of orbits within the family: N/S for halo,dragonfly,butterfly, E/W for lpo, and pq integer sequence for resonant (e.g., 12 for 1:2).")
    parser.add_argument("--periodmin", type=str, default="", help="minimum period (inclusive). Units defined by periodunits.")
    parser.add_argument("--periodmax", type=str, default="", help="maximum period (inclusive). Units defined by periodunits.")
    parser.add_argument("--periodunits", type=str, default="", help="units of pmin and pmax: s for seconds, h for hours, d for days, TU for nondimensional.")
    parser.add_argument("--jacobimin", type=str, default="", help="minimum Jacobi constant (inclusive). Nondimensional units.")
    parser.add_argument("--jacobimax", type=str, default="", help="maximum Jacobi constant (inclusive). Nondimensional units.")
    parser.add_argument("--stabmin", type=str, default="", help="minimum stability index (inclusive).")
    parser.add_argument("--stabmax", type=str, default="", help="maximum stability index (inclusive).")


    args = parser.parse_args()

    data = queryJPL(**vars(args))
    sysDict, ics = parseData(data)
    mu = sysDict['mu']

    trajList = propagate(mu, ics, n=10)
    plotTrajs(sysDict, trajList, savefig=False)

if __name__=='__main__':
    main()
import pandapower.networks as nw
import pandapower as pp
import sys
import numpy as np 

def main():
    # Load the n bus system
    n = sys.argv[1]
    if n == "14":
        net = nw.case14()
    elif n == "30":
        net = nw.case30()
    elif n == "57":     
        net = nw.case57()
    else:
        net = nw.case118()
    
    # See all lines (from_bus, to_bus, 0-indexed)
    print(net.line[['from_bus', 'to_bus', 'name']])

    # sample from a gaussian distribution centred at 1 so we bias one line
    # outages to be more likely than multiple line outages
    number_of_line_outages = int(np.clip(np.random.normal(loc=1, scale=1), 1, len(net.line)))
    print(f"Number of line outages: {number_of_line_outages}")
    line_outage_indices = np.random.randint(0, len(net.line), size=number_of_line_outages)
    print(f"Line outage indices: {line_outage_indices}")

    for idx in line_outage_indices:
        net.line.at[idx, 'in_service'] = False

    try:
        pp.runpp(net)
        print("Power flow calculation successful.")
    except pp.LoadflowNotConverged:
        print("Power flow calculation did not converge.")

if __name__ == "__main__":
    main()

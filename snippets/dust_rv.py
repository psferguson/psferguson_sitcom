from rubin_sim.phot_utils import DustValues, Bandpass,rubin_bandpasses
import astropy
filters = ["u", "g", "r", "i", "z", "y"]
throughputs = {}
for filter_ in filters:
    tput = astropy.table.Table.read(f"https://raw.githubusercontent.com/lsst/throughputs/main/baseline/total_{filter_}.dat", format="ascii")
    tput.rename_column("col1", "wavelength")
    tput.rename_column("col2", "throughput")
    throughputs[filter_] =Bandpass( wavelen=tput["wavelength"], sb=tput["throughput"])

Rx_dict=DustValues(bandpass_dict=throughputs).ax1
_=[print(f"{band}: {Rx_dict[band]:0.3f}") for band in filters]

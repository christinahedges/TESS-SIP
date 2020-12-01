# TESS SIP
<a href="https://doi.org/10.5281/zenodo.4300754"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4300754.svg" alt="DOI"></a>

Demo tool for creating a Systematics-insensitive Periodogram (SIP) to detect long period rotation in NASA's TESS mission data.

## What is SIP

SIP is a method of detrending telescope systematics simultaneously with calculating a Lomb-Scargle periodogram. You can read a more in-depth work of how SIP is used in NASA's Kepler/K2 data [here](https://ui.adsabs.harvard.edu/abs/2016ApJ...818..109A/abstract).


## Usage

This repository contains a Python tool to create a SIP. An example of a SIP output is below. You can run a simple notebook in the `docs` folder to show how to use SIP.

```python
from tess_sip import SIP
import lightkurve as lk
# Download target pixel files
tpfs = lk.search_targetpixelfile('TIC 288735205', mission='tess').download_all()
# Run SIP
r = SIP(tpfs)
```

`r` is a dictionary containing all the information required to build a plot like the one below.

![Example SIP output](https://github.com/christinahedges/TESS-SIP/blob/master/demo.png?raw=true)

### Installation

You can pip install this tool:

```
pip install tess_sip
```


## Requirements

To run this demo you will need to have [lightkurve](https://github.com/keplerGO/lightkurve) installed, with a minimum version number of v2.0.

## Acknowledgements

This tool uses the [lightkurve](https://github.com/keplerGO/lightkurve) tool to build a SIP, and relies on the `RegressionCorrector` and `SparseDesignMatrix` lightkurve tools. The SIP project was developed in part at the `online.tess.science` meeting, which took place globally in 2020 September. This research made use of [Astropy](http://www.astropy.org.) a community-developed core Python package for Astronomy.

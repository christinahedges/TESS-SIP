# TESS SIP

Demo tool for creating a Systematics-insensitive Periodogram (SIP) to detect long period rotation in NASA's TESS mission data. 

## What is SIP

SIP is a method of detrending telescope systematics simultaneously with calculating a Lomb-Scargle periodogram. You can read a more in-depth work of how SIP is used in NASA's Kepler/K2 data [here](https://ui.adsabs.harvard.edu/abs/2016ApJ...818..109A/abstract). 


## Usage

This repository contains a demo notebook for how to calculate a SIP for NASA CVZ targets. To use this demo, you can either download and run the notebook, run the notebook online google's [Colab](https://colab.research.google.com/github/christinahedges/TESS-SIP/blob/master/TESS-SIP.ipynb) (for free!), or re-write/copy the scripts in the notebook in your own tools. 

![Example SIP output](https://github.com/christinahedges/TESS-SIP/blob/master/demo.png?raw=true)


## Acknowledgements

This tool uses the [lightkurve](https://github.com/keplerGO/lightkurve) tool to build a SIP, and relies on the `RegressionCorrector` and `SparseDesignMatrix` lightkurve tools. The SIP project was developed in part at the `online.tess.science` meeting, which took place globally in 2020 September.
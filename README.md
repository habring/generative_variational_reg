# A Generative Variational Model for Inverse Problemss in Imaging

In this repository we provide the source code to reproduce the results from the paper [A Generative Variational Model for Inverse Problemss in Imaging](https://arxiv.org/abs/2104.12630).

## Highlights
* High quality tomography reconstructions.
* Easy to use graphical user interface.
* Fast reconstruction due to custom OpenCL/GPU-implementation.
* Preprocessing options to ensure data fits the framework.
 

## Requirements
The code is written for Python 3.9. Dependency management is handled with [poetry](https://python-poetry.org/docs/). For details on necessary versions see the file `pyproject.toml`. To use the code, clone the repository and in the repository run the following command from your shell:
```
poetry install
```
This should install all necessary dependencies. Afterwards you can run for instance the python script `demo.py` via
```
poetry run python demo.py
```


## Requirements
The code is written for Python 2.7 though it also works in Python 3. No dedicated installation is needed for the program, simply download the code and get started. Be sure to have the following Python modules installed, most of which should be standard.

* tkinter
* [pyopencl](https://pypi.org/project/pyopencl/) (>=pyopencl-2014.1)
* [argparse](https://pypi.org/project/argparse/)
* [numpy](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)
* [matplotlib](https://pypi.org/project/matplotlib/) (with tkagg backend)
* [mrcfile](https://pypi.org/project/mrcfile/)
* subprocess
* queue
* threading
* shlex
* shutil
* [h5py](https://pypi.org/project/h5py/)

Particularly, correctly installing and configuring PyOpenCL might take some time, as dependent on the used platform/GPU, suitable drivers must be installed.

## Getting started
To start the Graphical User interface, run `GUI.py` inside the Graptor folder (e.g. in a terminal via `python GUI.py` or similarly from an Python development environment). 
We refer to the [manual](manual/manual.pdf) for precise instructions on how to use the GUI. It is adviced to run the examples in the manual with the phantom test data in order to get a grasp of the relevant functions and options.

Additionally, the script `Reconstruction_coupled.py` is provided for using the reconstruction algorithm inside a terminal. You can find help via `python Reconstruction_coupled.py --help` concerning possible parameters as well as an example for the call.

## Known issues

* There appears to be an issue with the automatic splitting in case of insufficient GPU memory under Windows. If problems occur, try to use the `Maximal chunk size` option in the GUI or the `--Chunksize` parameter of `Reconstruction_coupled.py` to reduce memory requirements.

## Authors

* **Andreas Habring** andreas.habring@uni-graz.at
* **Martin Holler** martin.holler@uni-graz.at 

All authors are affiliated with the [Institute of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en) at the [University of Graz](https://www.uni-graz.at/en).


## Reproduction of numerical results

The results of the article *Total Generalized Variation regularization for multi-modal electron tomography* can be reproduced with the software by running the following terminal commands inside the Graptor folder.

```
# Reproduce results with phantom data and save to `phantom_data_results`
python Reconstruction_coupled.py "example/HAADF_lrsino.mrc" "example/Yb_EDXsino.mrc" "example/Al_EDXsino.mrc" "example/Si_EDXsino.mrc" --Outfile "phantom_data_results/reconstruction" --alpha 4.0 1.0 --mu 0.2 0.0005 0.004 0.0005 --Maxiter 2500 --Regularisation TGV --Discrepancy KL --Coupling FROB3D --SliceLevels 0 59 1 --Channelnames "HAADF" "ytterbium" "aluminum" "silicon" --Datahandling bright thresholding 0 "" 0.05 --Find_Bad_Projections 0 0 --Overlapping 1 
# Download experimental data
wget -r -l 1 -nd -P experimental_data -A .mrc,.rawtlt https://zenodo.org/record/2578866
# Reproduce results with phantom data and save to `experimental_data_results`
python Reconstruction_coupled.py "experimental_data/FEI HAADF_aligned_norm_ad.mrc" "experimental_data/EDS Al K Map_aligned_norm.mrc"  "experimental_data/EDS Si K Map_aligned_norm.mrc"  "experimental_data/EDS Yb L Map_aligned_norm.mrc" --Outfile "experimental_data_results/reconstruction" --Datahandling bright thresholding 0.0 "" 0.05 --Discrepancy=KL  --Regularisation=TGV --SliceLevels 10 270 1 --alpha 4.0 1.0 --mu 0.1 0.0024 0.0014 0.001 --Coupling=Frob3d --Channelnames HAADF Aluminum Silicon Ytterbium --Maxiter=5000 --Scalepars 0 100
```

## Test Images

In this repository you will find eexperiments with 5 different images. Three of them are copyright free images. The fish image ["Pomocanthus imperator facing right"](https://commons.wikimedia.org/wiki/File:Pomocanthus_imperator_facing_right.jpg), by [Albert kok](https://commons.wikimedia.org/wiki/User:Albert_kok), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).


## Acknowledgements

The authors acknowledge funding by the Austrian Research Promotion Agency (FFG) (Project number 881561). Martin Holler further is a member NAWI Graz (https://www.nawigraz.at) and BioTechMed Graz (https://biotechmedgraz. at).

## Citation

```
@misc{habring2021generative,
      title={A Generative Variational Model for Inverse Problems in Imaging}, 
      author={Andreas Habring and Martin Holler},
      year={2021},
      eprint={2104.12630},
      archivePrefix={arXiv},
      primaryClass={math.OC}
      journal={SIAM Journal on Mathematics of Data Science}
}
```

## License

???????????????????????

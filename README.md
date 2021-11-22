# A Generative Variational Model for Inverse Problemss in Imaging

In this repository we provide the source code to reproduce the results from the paper [A Generative Variational Model for Inverse Problems in Imaging](https://arxiv.org/abs/2104.12630). The repository contains scripts to reproduce the paper results with proposed method and with TGV regularization. We did not include the scripts to reproduce the results on the Imagenet data set due to copyright issues. The offered source code, however, should allow you to easily use the method on any test image you like. To reproduce the results with the other comparison methods in the paper we refer to the respective git repositories ([A convex variational model for learning convolutional image atoms from incomplete data](https://github.com/hollerm/convex_learning), [Deep Image Prior](https://github.com/DmitryUlyanov/deep-image-prior)).

## Requirements
The code is written for Python 3.9 (versions >=3.9 and <3.11). Dependency management is handled with [poetry](https://python-poetry.org/docs/). For details on necessary package versions see the file `pyproject.toml`. Before using the code make sure you have installed poetry on your system. Then, clone the repository and in the repository run the following command from your shell:
```
poetry install
```
This should install all necessary dependencies. Afterwards you can run for instance the python script `demo.py` via
```
poetry run python demo.py
```

## Repository Structure
The repository is structured as follows: In the folder `scripts`, there are all scripts to reproduce the results from the paper. Details are explained in the next section. In the folder `source`, the used methods and other auxiliary functions are implemented:
* `source/gen_reg.py` contains the implementation of the proposed generative variational regularization method.
* `source/libjpeg.py` contains functions for the handling of JPEG data.
* `source/matpy.py` contains auxiliary functions that are used multiple times in the code as well as the implementation of the TGV regularization.

## Reproduction of the Results
* Run the script `demo.py` via typing
```
poetry run python demo.py
```
in your shell in the directory `scripts`. If everything is installed correctly it should perform inpainting with TGV and the proposed method each on the Barbara test image and store the results in the directory scripts/experiments/demo.

* The file `reproduce_generative_regularization_results.py` contains a script to reproduce the results with the proposed method. You can choose which experiments to perform by modifying the list `cases` at the beginning of the file. If you do not modify anything, all experiments will be performed with all possible test images. The results will be stored in scripts/experiments/gen_reg. Run the script via typing
```
poetry run python reproduce_generative_regularization_results.py
```
in your shell.

* The file `reproduce_tgv_results.py` contains a script to reproduce the results for TGV regularization. The results will be stored in scripts/experiments/tgv. To execute it type
```
poetry run python reproduce_tgv_results.py
```
in your shell.

* Run the script `conv_incr_reg.py` to reproduce Figure 2.1 in the paper.

## Test Images

For the experiments we use five different test images which are located in scripts/imsource. The fish image ["Pomocanthus imperator facing right"](https://commons.wikimedia.org/wiki/File:Pomocanthus_imperator_facing_right.jpg), by [Albert kok](https://commons.wikimedia.org/wiki/User:Albert_kok) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The other images are license free.

## Authors

* **Andreas Habring** andreas.habring@uni-graz.at
* **Martin Holler** martin.holler@uni-graz.at 
* **Kristian Bredies** kristian.bredies@uni-graz.at provided the code in source/libjpeg.py.

All authors are affiliated with the [Institute of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en) at the [University of Graz](https://www.uni-graz.at/en).

## Acknowledgements

The authors acknowledge funding by the Austrian Research Promotion Agency (FFG) (Project number 881561). Martin Holler further is a member of NAWI Graz (https://www.nawigraz.at) and BioTechMed Graz (https://biotechmedgraz.at).

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

This project is licensed under the GPLv3 license - see the [LICENSE](LICENSE) file for details.

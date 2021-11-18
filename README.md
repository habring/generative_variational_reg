# A Generative Variational Model for Inverse Problemss in Imaging

In this repository we provide the source code to reproduce some of the results from the paper [A Generative Variational Model for Inverse Problems in Imaging](https://arxiv.org/abs/2104.12630). To be precise, this repository contains scripts to reproduce the results shown in the paper that were obtained with the proposed method (`gen_reg.py`) and with TGV regularization. We did not include the scripts to reproduce the results on the Imagenet data set due to copy-right issues. The offered source code, however, should also allow you to easily use the method on any test image you like.

## Requirements
The code is written for Python 3.9 (it works for versions >=3.9 and <3.11) Dependency management is handled with [poetry](https://python-poetry.org/docs/). For details on necessary versions see the file `pyproject.toml`. To use the code, clone the repository and in the repository run the following command from your shell:
```
poetry install
```
This should install all necessary dependencies. Afterwards you can run for instance the python script `demo.py` via
```
poetry run python demo.py
```

## Reproduction of numerical results
* The file `demo.py` should serve as a test, if everything is installed/set up correctly. Run it via
```
poetry run python demo.py
```
It should perform inpainting with TGV and the proposed method and store the results in scripts/experiments/demo.

* The file `reproduce_generative_regularization_results.py` contains a scripts to reproduce the results for all applications shown in the paper on all test images with the proposed method. You can, however, chose to only perform specific applications by modifying the script. There is a line at the beginning where you can choose which cases to consider. To run the file, type 
```
poetry run python reproduce_generative_regularization_results.py
```
in your shell.

* The file `reproduce_tgv_results.py` contains a scripts to reproduce the results for TGV regularization. To execute it type
```
poetry run python reproduce_generative_regularization_results.py
```
in your shell.

## Authors

* **Andreas Habring** andreas.habring@uni-graz.at
* **Martin Holler** martin.holler@uni-graz.at 

All authors are affiliated with the [Institute of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en) at the [University of Graz](https://www.uni-graz.at/en).

## Test Images

For the experiments we use five diffeerent test images which are located in scripts/imsource. The fish image ["Pomocanthus imperator facing right"](https://commons.wikimedia.org/wiki/File:Pomocanthus_imperator_facing_right.jpg), by [Albert kok](https://commons.wikimedia.org/wiki/User:Albert_kok), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

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

# A Generative Variational Model for Inverse Problemss in Imaging

In this repository we provide the source code to reproduce some of the results from the paper [A Generative Variational Model for Inverse Problems in Imaging](https://arxiv.org/abs/2104.12630). To be precise, this repository contains scripts to reproduce the results shown in the paper that were obtained with the proposed method (`gen_reg.py`) and with TGV regularization. We did not include the scripts to reproduce the results on the Imagenet data set due to copy-right issues. The offered source code, however, should also allow you to easily use the method on any test image you like.

## Requirements
The code is written for Python 3.9. Dependency management is handled with [poetry](https://python-poetry.org/docs/). For details on necessary versions see the file `pyproject.toml`. To use the code, clone the repository and in the repository run the following command from your shell:
```
poetry install
```
This should install all necessary dependencies. Afterwards you can run for instance the python script `demo.py` via
```
poetry run python demo.py
```

## Getting started
Running the file `demo.py` will

## Known issues

## Authors

* **Andreas Habring** andreas.habring@uni-graz.at
* **Martin Holler** martin.holler@uni-graz.at 

All authors are affiliated with the [Institute of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en) at the [University of Graz](https://www.uni-graz.at/en).


## Reproduction of numerical results

* The results of the article *Total Generalized Variation regularization for multi-modal electron tomography* can be reproduced with the software by running the following terminal commands inside the Graptor folder.

```
poetry run python reproduce_gen_reg_results.py
poetry run python reproduce_tgv_results.py
```

## Test Images

In this repository you will find eexperiments with 5 different images. Three of them are copyright free images. The fish image ["Pomocanthus imperator facing right"](https://commons.wikimedia.org/wiki/File:Pomocanthus_imperator_facing_right.jpg), by [Albert kok](https://commons.wikimedia.org/wiki/User:Albert_kok), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).


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

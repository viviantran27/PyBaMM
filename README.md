# PyBaMM

[![Build](https://github.com/pybamm-team/PyBaMM/workflows/PyBaMM/badge.svg)](https://github.com/pybamm-team/PyBaMM/actions?query=workflow%3APyBaMM+branch%3Adevelop)
[![readthedocs](https://readthedocs.org/projects/pybamm/badge/?version=latest)](https://pybamm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pybamm-team/PyBaMM/branch/master/graph/badge.svg)](https://codecov.io/gh/pybamm-team/PyBaMM)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/master/)
[![black_code_style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-21-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

PyBaMM (Python Battery Mathematical Modelling) solves physics-based electrochemical DAE models by using state-of-the-art automatic differentiation and numerical solvers. The Doyle-Fuller-Newman model can be solved in under 0.1 seconds, while the reduced-order Single Particle Model and Single Particle Model with electrolyte can be solved in just a few milliseconds. Additional physics can easily be included such as thermal effects, fast particle diffusion, 3D effects, and more. All models are implemented in a flexible manner, and a wide range of models and parameter sets (NCA, NMC, LiCoO2, ...) are available. There is also functionality to simulate any set of experimental instructions, such as CCCV or GITT, or specify drive cycles.

## How do I use PyBaMM?

The easiest way to use PyBaMM is to run a 1C constant-current discharge with a model of your choice with all the default settings:
```python3
import pybamm
model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
sim = pybamm.Simulation(model)
sim.solve([0, 3600])  # solve for 1 hour
sim.plot()
```
or simulate an experiment such as CCCV:
```python3
import pybamm
experiment = pybamm.Experiment(
    [
        "Discharge at C/10 for 10 hours or until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        "Rest for 1 hour",
    ]
    * 3,
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()
sim.plot()
```
However, much greater customisation is available. It is possible to change the physics, parameter values, geometry, submesh type,  number of submesh points, methods for spatial discretisation and solver for integration (see DFN [script](examples/scripts/DFN.py) or [notebook](examples/notebooks/models/DFN.ipynb)).

For new users we recommend the [Getting Started](examples/notebooks/Getting%20Started/) guides. These are intended to be very simple step-by-step guides to show the basic functionality of PyBaMM, and can either be downloaded and used locally, or used online through [Google Colab](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/master/).

Further details can be found in a number of [detailed examples](examples/notebooks/README.md), hosted here on
github. In addition, there is a [full API documentation](http://pybamm.readthedocs.io/),
hosted on [Read The Docs](readthedocs.io).
Additional supporting material can be found
[here](https://github.com/pybamm-team/pybamm-supporting-material/).

For further examples, see the list of repositories that use PyBaMM [here](https://github.com/pybamm-team/pybamm-example-results)

## How can I install PyBaMM?
PyBaMM is available on GNU/Linux, MacOS and Windows.
We strongly recommend to install PyBaMM within a python virtual environment, in order not to alter any distribution python files.
For instructions on how to create a virtual environment for PyBaMM, see [the documentation](https://pybamm.readthedocs.io/en/latest/install/GNU-linux.html#user-install).

### Using pip
```bash
pip install pybamm
```

### Using conda
PyBaMM is available as a conda package through the conda-forge channel.
```bash
conda install -c conda-forge pybamm
```

### Optional solvers
On GNU/Linux and MacOS, an optional [scikits.odes](https://scikits-odes.readthedocs.io/en/latest/)-based solver is available, see [the documentation](https://pybamm.readthedocs.io/en/latest/install/GNU-linux.html#scikits-odes-label).

## Citing PyBaMM

If you use PyBaMM in your work, please cite our paper

> Sulzer, V., Marquis, S. G., Timms, R., Robinson, M., & Chapman, S. J. (2020). Python Battery Mathematical Modelling (PyBaMM). _ECSarXiv. February, 7_.

You can use the bibtex

```
@article{sulzer2020python,
  title={Python Battery Mathematical Modelling (PyBaMM)},
  author={Sulzer, Valentin and Marquis, Scott G and Timms, Robert and Robinson, Martin and Chapman, S Jon},
  journal={ECSarXiv. February},
  volume={7},
  year={2020}
}
```

We would be grateful if you could also cite the relevant papers. These will change depending on what models and solvers you use. To find out which papers you should cite, add the line

```python3
pybamm.print_citations()
```

to the end of your script. This will print bibtex information to the terminal; passing a filename to `print_citations` will print the bibtex information to the specified file instead. A list of all citations can also be found in the [citations file](pybamm/CITATIONS.txt). In particular, PyBaMM relies heavily on [CasADi](https://web.casadi.org/publications/).
See [CONTRIBUTING.md](CONTRIBUTING.md#citations) for information on how to add your own citations when you contribute.

## How can I contribute to PyBaMM?

If you'd like to help us develop PyBaMM by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](CONTRIBUTING.md) first.

## Licensing

PyBaMM is fully open source. For more information about its license, see [LICENSE](./LICENSE.txt).

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://sites.google.com/view/valentinsulzer"><img src="https://avatars3.githubusercontent.com/u/20817509?v=4" width="100px;" alt=""/><br /><sub><b>Valentin Sulzer</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Atinosulzer" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tinosulzer" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tinosulzer" title="Documentation">📖</a> <a href="#example-tinosulzer" title="Examples">💡</a> <a href="#ideas-tinosulzer" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-tinosulzer" title="Maintenance">🚧</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Atinosulzer" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tinosulzer" title="Tests">⚠️</a> <a href="#tutorial-tinosulzer" title="Tutorials">✅</a> <a href="#blog-tinosulzer" title="Blogposts">📝</a></td>
    <td align="center"><a href="http://www.robertwtimms.com"><img src="https://avatars1.githubusercontent.com/u/43040151?v=4" width="100px;" alt=""/><br /><sub><b>Robert Timms</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Artimms" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=rtimms" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=rtimms" title="Documentation">📖</a> <a href="#example-rtimms" title="Examples">💡</a> <a href="#ideas-rtimms" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-rtimms" title="Maintenance">🚧</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Artimms" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=rtimms" title="Tests">⚠️</a> <a href="#tutorial-rtimms" title="Tutorials">✅</a></td>
    <td align="center"><a href="https://github.com/Scottmar93"><img src="https://avatars1.githubusercontent.com/u/22661308?v=4" width="100px;" alt=""/><br /><sub><b>Scott Marquis</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3AScottmar93" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=Scottmar93" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=Scottmar93" title="Documentation">📖</a> <a href="#example-Scottmar93" title="Examples">💡</a> <a href="#ideas-Scottmar93" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-Scottmar93" title="Maintenance">🚧</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3AScottmar93" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=Scottmar93" title="Tests">⚠️</a> <a href="#tutorial-Scottmar93" title="Tutorials">✅</a></td>
    <td align="center"><a href="https://github.com/martinjrobins"><img src="https://avatars3.githubusercontent.com/u/1148404?v=4" width="100px;" alt=""/><br /><sub><b>Martin Robinson</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Amartinjrobins" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=martinjrobins" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=martinjrobins" title="Documentation">📖</a> <a href="#example-martinjrobins" title="Examples">💡</a> <a href="#ideas-martinjrobins" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Amartinjrobins" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=martinjrobins" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://www.brosaplanella.com"><img src="https://avatars3.githubusercontent.com/u/28443643?v=4" width="100px;" alt=""/><br /><sub><b>Ferran Brosa Planella</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Aferranbrosa" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Aferranbrosa" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=ferranbrosa" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=ferranbrosa" title="Documentation">📖</a> <a href="#example-ferranbrosa" title="Examples">💡</a> <a href="#ideas-ferranbrosa" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-ferranbrosa" title="Maintenance">🚧</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=ferranbrosa" title="Tests">⚠️</a> <a href="#tutorial-ferranbrosa" title="Tutorials">✅</a> <a href="#blog-ferranbrosa" title="Blogposts">📝</a></td>
    <td align="center"><a href="https://github.com/TomTranter"><img src="https://avatars3.githubusercontent.com/u/7068741?v=4" width="100px;" alt=""/><br /><sub><b>Tom Tranter</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3ATomTranter" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=TomTranter" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=TomTranter" title="Documentation">📖</a> <a href="#example-TomTranter" title="Examples">💡</a> <a href="#ideas-TomTranter" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3ATomTranter" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=TomTranter" title="Tests">⚠️</a></td>
    <td align="center"><a href="http://tlestang.github.io"><img src="https://avatars3.githubusercontent.com/u/13448239?v=4" width="100px;" alt=""/><br /><sub><b>Thibault Lestang</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Atlestang" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tlestang" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tlestang" title="Documentation">📖</a> <a href="#example-tlestang" title="Examples">💡</a> <a href="#ideas-tlestang" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Atlestang" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tlestang" title="Tests">⚠️</a> <a href="#infra-tlestang" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/research-software-engineering/"><img src="https://avatars1.githubusercontent.com/u/6095790?v=4" width="100px;" alt=""/><br /><sub><b>Diego</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Adalonsoa" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Adalonsoa" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=dalonsoa" title="Code">💻</a> <a href="#infra-dalonsoa" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://github.com/felipe-salinas"><img src="https://avatars2.githubusercontent.com/u/64426781?v=4" width="100px;" alt=""/><br /><sub><b>felipe-salinas</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=felipe-salinas" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=felipe-salinas" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/suhaklee"><img src="https://avatars3.githubusercontent.com/u/57151989?v=4" width="100px;" alt=""/><br /><sub><b>suhaklee</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=suhaklee" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=suhaklee" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/viviantran27"><img src="https://avatars0.githubusercontent.com/u/6379429?v=4" width="100px;" alt=""/><br /><sub><b>viviantran27</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=viviantran27" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=viviantran27" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/gyouhoc"><img src="https://avatars0.githubusercontent.com/u/60714526?v=4" width="100px;" alt=""/><br /><sub><b>gyouhoc</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Agyouhoc" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=gyouhoc" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=gyouhoc" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/YannickNoelStephanKuhn"><img src="https://avatars0.githubusercontent.com/u/62429912?v=4" width="100px;" alt=""/><br /><sub><b>Yannick Kuhn</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=YannickNoelStephanKuhn" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=YannickNoelStephanKuhn" title="Tests">⚠️</a></td>
    <td align="center"><a href="http://batterymodel.co.uk"><img src="https://avatars2.githubusercontent.com/u/39409226?v=4" width="100px;" alt=""/><br /><sub><b>Jacqueline Edge</b></sub></a><br /><a href="#ideas-jedgedrudd" title="Ideas, Planning, & Feedback">🤔</a> <a href="#eventOrganizing-jedgedrudd" title="Event Organizing">📋</a> <a href="#fundingFinding-jedgedrudd" title="Funding Finding">🔍</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://www.rse.ox.ac.uk/"><img src="https://avatars3.githubusercontent.com/u/3770306?v=4" width="100px;" alt=""/><br /><sub><b>Fergus Cooper</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=fcooper8472" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=fcooper8472" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/jonchapman1"><img src="https://avatars1.githubusercontent.com/u/28925818?v=4" width="100px;" alt=""/><br /><sub><b>jonchapman1</b></sub></a><br /><a href="#ideas-jonchapman1" title="Ideas, Planning, & Feedback">🤔</a> <a href="#fundingFinding-jonchapman1" title="Funding Finding">🔍</a></td>
    <td align="center"><a href="https://github.com/colinplease"><img src="https://avatars3.githubusercontent.com/u/44977104?v=4" width="100px;" alt=""/><br /><sub><b>Colin Please</b></sub></a><br /><a href="#ideas-colinplease" title="Ideas, Planning, & Feedback">🤔</a> <a href="#fundingFinding-colinplease" title="Funding Finding">🔍</a></td>
    <td align="center"><a href="https://faraday.ac.uk"><img src="https://avatars2.githubusercontent.com/u/42166506?v=4" width="100px;" alt=""/><br /><sub><b>Faraday Institution</b></sub></a><br /><a href="#financial-FaradayInstitution" title="Financial">💵</a></td>
    <td align="center"><a href="https://github.com/bessman"><img src="https://avatars3.githubusercontent.com/u/1999462?v=4" width="100px;" alt=""/><br /><sub><b>Alexander Bessman</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Abessman" title="Bug reports">🐛</a> <a href="#example-bessman" title="Examples">💡</a></td>
    <td align="center"><a href="https://github.com/dalbamont"><img src="https://avatars1.githubusercontent.com/u/19659095?v=4" width="100px;" alt=""/><br /><sub><b>dalbamont</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=dalbamont" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/anandmy"><img src="https://avatars1.githubusercontent.com/u/34894671?v=4" width="100px;" alt=""/><br /><sub><b>Anand Mohan Yadav</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=anandmy" title="Documentation">📖</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
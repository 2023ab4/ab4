# Inference and design of antibody specificity

This repository hosts the Julia code for the paper:

> Inference and design of antibody specificity: from experiments to models and back
> 
> Jorge Fernandez-de-Cossio-Diaz, Guido Uguzzoni, Kévin Ricard, Francesca Anselmi, Clément Nizak, Andrea Pagnani, Olivier Rivoire
> 
> bioRxiv 2023.10.23.563570; doi: https://doi.org/10.1101/2023.10.23.563570

See the full paper at this [link](https://www.biorxiv.org/content/10.1101/2023.10.23.563570). If you use this code in your work, please cite using the included [CITATION.bib](https://github.com/2023ab4/ab4/blob/main/CITATION.bib) file.

## Installation

The code is written in **Julia**. To install Julia, follow the instructions at [JuliaLang Downloads](https://julialang.org/downloads/).

### Steps to Install the Package

1. Clone this repository using `git`:

```bash
git clone https://github.com/2023ab4/ab4.git
```

2. In the Julia REPL, add the package:

```julia
import Pkg
Pkg.add("<path-to-cloned-repository>")
```

## Pluto and Jupyter Notebooks

### Pluto Notebooks

The `notebooks` directory contains Pluto notebooks that provide examples for:

- Training models.
- Performing various analyses.
- Reproducing the figures and plots from the paper.

We recommend using **Julia 1.10** for the best compatibility. To launch these, open a Julia REPL, and run:

```julia
using Pluto
Pluto.run()
```

Then, navigate to the notebooks folder and select the desired notebook.

### Jupyter Notebooks
Jupyter notebooks are also available in the notebooks directory to help reproduce the results from the paper. To use them:

1. Install Python and Jupyter. For example, you can use https://www.anaconda.com/download.
2. Install the IJulia kernel. Follow instructions here: https://github.com/JuliaLang/IJulia.jl.
3. Start a Jupyter session:

```bash
jupyter notebook
```

4. Navigate to the notebooks directory, and ensure that the code is executed within the Ab4Paper2023 environment, and with the IJulia kernel selected.

As with Pluto, we suggest using Julia 1.10 for compatibility.

## Project Structure
- src/: Contains the source code for the project.
- notebooks/: Contains Pluto and Jupyter notebooks for reproducing results.
- data/: Input data used in the paper's analysis. 
- test/: Contains test module for simple testing functionality


## License
[MIT license](LICENSE)

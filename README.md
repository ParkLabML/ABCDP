# abcdp

Approximate Bayesian Computation with Differential Privacy

## How to install?
It is best to use [Anaconda](https://www.anaconda.com/). Make sure you [create
a separate Anaconda
environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for this project. Use Python 3.

To install, first switch to the right environment. If you use Anaconda, then
you can do this with `conda activate name_of_environment`. This repository
contains a Pyton-importable package that can be installed with `pip` command:

    pip install -e /path/to/the/code/folder/of/this/repo/on/your/local/drive

The "code" folder is the one that contains `setup.py`.
Also do:
  
    pip install autodp

Once installed, you should be able to do `import abcdp` without any error.

## How to run the experiments?

### Flip probability plot

- Run `./ABCDP/abcdp/prob_flip.py` in order to get Figure 1.

### Toy experiment
 - Run `./ABCDP/abcdp/auxiliary_files/run_toy_example.py` to compute the ABCDP algortihm for different parameter settings. 
 - To obtain Figure 2, run  `./ABCDP/abcdp/figure_toy_example.py`. 

### Coronavirus outbreak data

- To obtain Figure 4 run `./ABCDP/code/abcdp/figure_covid.py`

### Modelling Tuberculosis (TB) Outbreak

- Run `./ABCDP/abcdp/auxiliary_files/test_TB.py` file to get the results.
- For plotting Figure 5 run `./ABCDP/abcdp/figure_covid.py`file.




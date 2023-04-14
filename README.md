# \{PF\} $^2$ ES: Parallel Feasible Pareto Frontier Entropy Search Acquisition Function for Multi-Objective Bayesian Optimization 
![](PF2ES_VLMOP2.gif)


### Installation
:exclamation::exclamation::exclamation:Caution: You are away from [the main branch of Trieste](https://github.com/secondmind-labs/trieste), this branch contains certain other functionalities  

install from sources, run
```bash
$ pip install -e.
```
in the repository root (tested with Python version 3.7.11).

### Tutorial/Demonstration Notebook
There is a standalone tutorial notebook `multi_objective_entropy_search_via_PF2ES.py` at ([`\docs\notebooks`](https://github.com/TsingQAQ/trieste/blob/PF2ES_preview_notebook/docs/notebooks/multi_objective_entropy_search_pf2es.pct.py)) demonstrating 
how to make use of the acquisition function for different multi-objective optimization tasks.
 
  
  In order to run the notebook, install the following dependency:     
  ``` 
  $ pip install -r notebooks/requirements.txt
  ```  

  Then, run the notebooks with  
   ```
  $ jupyter-notebook notebooks
  ```
  
  #### How to run on your favorate problems 
  - Define your own problem to optimize in `trieste/objectives/multi_objectives.py`
  - Optimization as done in the tutorial notebook, then have fun!
  
  --------------

### Other info
- For **comparison** and **reproduciability**:  the repo capable of running all the experiments can be found [here](https://github.com/TsingQAQ/pf2es), note that all the experiments are conducted via `docs\exp\`
- For other information-theoretic acquisitions utilized in experimental comparison, please refer their branch:
  - PESMO: https://github.com/HIPS/Spearmint/tree/PESM
  - MESMOC+: https://github.com/fernandezdaniel/Spearmint
  - PPESMOC: https://github.com/EduardoGarrido90/spearmint_ppesmoc
- There is also a backup repo powered by [UGENT SUMO lab](https://sumo.intec.ugent.be/) at [here](https://gitlab.ilabt.imec.be/sumolab/trieste/-/tree/PF2ES_preview_notebook).  

### Citation
If you find this work or repository helpful, please kindly consider citing our work:
```
@InProceedings{pmlr-v206-qing23a,
  title = 	 {\{PF\}$^2$ES: Parallel Feasible Pareto Frontier Entropy Search for Multi-Objective Bayesian Optimization},
  author =       {Qing, Jixiang and Moss, Henry B. and Dhaene, Tom and Couckuyt, Ivo},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {2565--2588},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v206/qing23a/qing23a.pdf},
  url = 	 {https://proceedings.mlr.press/v206/qing23a.html},
  abstract = 	 {We present Parallel Feasible Pareto Frontier Entropy Search ($\{\mathrm{PF}\}^2$ES) — a novel information-theoretic acquisition function for multi-objective Bayesian optimization supporting unknown constraints and batch queries. Due to the complexity of characterizing the mutual information between candidate evaluations and (feasible) Pareto frontiers, existing approaches must either employ crude approximations that significantly hamper their performance or rely on expensive inference schemes that substantially increase the optimization’s computational overhead. By instead using a variational lower bound, $\{\mathrm{PF}\}^2$ES provides a low-cost and accurate estimate of the mutual information. We benchmark $\{\mathrm{PF}\}^2$ES against other information-theoretic acquisition functions, demonstrating its competitive performance for optimization across synthetic and real-world design problems.}
}
```

  

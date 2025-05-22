# Deep Counterfactual Regret Minimization
An implementation of <cite>[Brown et al. 2019][1]</cite> with some adaptations.

DeepCFR is an approach to solve Counterfactual Regret Minimization by using a neural network to learn the relation between information sets and action probabilities, rather than using a tabular approach.

This implementation applies DeepCRF to Texas Holdem No Limit poker.

## Getting started
Clone the repo and create install the requirements with 
```pip install -r requirements.txt```

## Training
Start a training with `python train.py`

In the `config.yaml` file, you can change experiment and training hyperparameters.

## WIP Evaluation



[1]: https://arxiv.org/abs/1811.00164

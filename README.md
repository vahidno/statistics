# Statistics

Different topics and problems in statistics and probability:

## a_combination_problem.ipynb

### Problem Statement

Anita randomly picks 4 cards from a deck of 52-cards and places them back into the deck (Any set of 4 cards is equally likely). 
Then, Babita randomly chooses 8 cards out of the same deck (Any set of 8 cards is equally likely). 
Assume that the choice of 4 cards by Anita and the choice of 8 cards by Babita are independent. 
What is the probability that all 4 cards chosen by Anita are in the set of 8 cards chosen by Babita?

### Solutions

Two solution methods are suggested: 

	(i) Monte Carlo simulation, and 
	(ii) Exact method

## t_distribution.ipynb

### Topic 1

Show how t-distribution converges to normal distribution

### Topic 2

Use Monte Carlo Integration to estimate the CDF of a t-distribution at point `a`.

Confirm the results using the inverse of `CDF` function: `PPF`

## forward_selection_with_statsmodel.py

### Topic

Perform forward stepwise feature selection to narrow down predictors (a.k.a. features, or independent variables) in linear regression

Use k-fold cross-validation method and Root Mean Squared Error (RMSE) for evaluating and choosing regression models.
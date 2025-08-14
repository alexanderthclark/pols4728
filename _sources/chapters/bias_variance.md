# Predictive Modeling

## Predicting $\hat{y}$

Most of the machine learning models we will cover focus on prediction. In this world, we are not interested in marginal effects like the increase in wages attributable to schooling, standard errors, or even interpretability. Instead, we focus on some measure of predictive accuracy. A black box model is fine and a simpler model might only be preferred with parsimony as a tiebreaker.

Let's focus on regression problems where $y$ is a continuous scalar value. For concreteness, let's say we are predicting midterm election vote share like in {cite}`tufte1975determinants`. The $y$ variable is $\frac{\text{Votes for Incumbent's Party in House Races}}{\text{Total Votes in House Races}}$ and suppose we have a measure of economic growth as the single predictor variable $x$. We can use simple linear regression for this and 



## 

Many machine learning methods essentially seek to learn a function $f$. We observe outcomes $y_i$ and a predictor variable $x_i$ where there is some underlying process $y_i = f(x_i) + \varepsilon$. $\varepsilon$ is unavoidable noise in the process. If we learn $f$, then we can predict $y$ for new $x$. If we learn $f$ perfectly, then our predictive accuracy is only limited by the noise component $\varepsilon$. 


# Summary 

Over-fitting is a major concern for anyone doing predictive modeling. Researchers have 



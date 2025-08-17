(linreg)=
# Linear Regression

Regression should be a familiar topic. We will move quickly through what might be familiar and then introduce machine learning vocabulary and other points of emphasis. Some of this, like discussion of the loss surface, will feel unnecessary. However, this is a chance to ease into new concepts that will arise again for other prediction models. 

```{admonition} Reading
:class: seealso
- {cite}`kuhn2013applied`, Chapter 6
- {cite}`hastie2009elements`, Chapter 3
```

## Linear Regression

Can you think of some famous lines? Maybe you are thinking of the Equator, the Mason Dixon, or the line in the sand at the Alamo. We can do better. $\hat{y} = 68 + \frac{2}{3}(68-x)$ is the line from which we get the term **regression**.

> "The height-deviate of the offspring is, on the average, two-thirds of the height-deviate of its mid-parentage." {cite}`galton1886regression`

Francis Galton found this line by observing that tall parents tended to have shorter (closer to average) children, while short parents tended to have taller kids. He described this phenomenon as "regression to mediocrity," reflecting the tendency of extreme characteristics to move back toward the population average in subsequent generations. Galton actually used "ocular regression" (eyeballing it) and the term *regression* has stuck for the general line-of-best-fit technique, even when applied to data that don't follow this pattern. Regression is also sometimes used to describe any kind of model that predicts a numeric value (for example, a decision tree might be called regression tree).

In 2025, ocular regression doesn't cut it. Ordinary least squares (OLS) is the most common method for estimating the parameters in a linear regression model. Linear models are flexible because they can still accommodate interactions, categorical predictors, and nonlinearities. You, the analyst, just have to include them in your specification. 

Our predictors give us the design matrix, $X$. With $n$ observations and $k$ features (including an intercept), this is $n\times k$. The target variable is stored in the $n\times 1$ matrix, $y$.

Then, 

$$\hat{\beta} = (X^TX)^{-1} X^T y.$$


Because linear algebra is king in machine learning, we'll give the geometric interpretation. OLS solves a projection problem:

* The column space of $X$ is the set of all possible predictions we can make using linear combinations of our features. This forms a $k$-dimensional subspace in $\mathbb{R}^n$.
* Our observed $y$ vector typically doesn't lie exactly in this column space (for example, for three points that you can't draw a single line through in simple linear regression).
* OLS finds $\hat{\beta}$ such that $X\hat{\beta}$ is the vector in the column space closest to $y$.

Mathematically, $X\hat{\beta}$ is the orthogonal projection of $y$ onto the column space of $X$. Orthogonality is what makes $X\hat{\beta}$ closer to $y$ than any other candidate:

$$\Vert y - X\hat{\beta} \Vert \leq \Vert y - Xv \Vert.$$

for any other $k \times 1$ vector $v$. In other words, no other choice of coefficients can get us closer to $y$.

The quality of the fit is not generally measured by $\Vert y - X\hat{\beta} \Vert$.[^1] Instead we usually report the mean squared error (MSE),

[^1]: $\Vert \cdot \Vert$ is the Euclidean or L2 norm. 

$$ \mathrm{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n}\Vert y - X\hat{\beta} \Vert^2.$$

This scaling makes MSE comparable across datasets of different sizes.

## Residuals 

## Feature Engineering


## Multiple Linear Regression



## Application: Determinants of the Outcomes of Midterm Congressional Elections

{cite}`tufte1975determinants` proposes a regression model to understand the outcomes of midterm congressional elections in the US,

$$\hat{Y}_i = \beta_0 + \beta_1 P_i + \beta_2 (\Delta E_i),$$

where 

* $Y_i$ is the standardized midterm vote for the party of the sitting president in midterm election $i$,
* $P_i$ is the percentage approving of the President in the Gallup poll in the September prior to the election,
* $\Delta E_i$ is the year-over-year change in real disposable personal income per capita.



# Python

## Simple Linear Regression

Let's take a look at the father-and-son height data analyzed by Karl Pearson. Below, we load the data using the `pandas` library. For large data sets, you might prefer to use `polars` instead.

```
import pandas as pd
url = 'https://raw.githubusercontent.com/alexanderthclark/Stats1101/main/Data/FatherSonHeights/pearson.csv'
df = pd.read_csv(url)  # DataFrame
```

`statsmodels` and `scikit-learn` are the two obvious choices for linear regression in Python. If you are coming to Python from R, you might prefer `statsmodels`. With the formula API, you can use the familiar Wilkinson-Rogers notation.

```
import statsmodels.formula.api as smf

model_smf = smf.ols(formula='Father ~ Son', data=df).fit()
print(model_smf.summary())
```

However, we will prefer `scikit-learn` because it is built for machine learning workflows, while `statsmodels` is more tailored for statistical inference (inspecting standard errors, etc). 

```
from sklearn.linear_model import LinearRegression

# Train
X = df['Father'].values.reshape(-1,1)
y = df['Son']
model = LinearRegression().fit(X, y)

# Inspect the fitted model
coefficients = model.coef_
intercept = model.intercept_
```

And direct matrix multiplication is an option, using `numpy`. 

```
import numpy as np

df['intercept'] = 1
X = df[['intercept', 'Father']].to_numpy()
y = df['Son'].to_numpy()
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
```



## Multiple Regression

Now, let's add some more predictors through "feature engineering." We'll include father's height raised to each power from 2 through 9. 

```
for p in range(2,10):
    df[f'Father_{p}'] = df['Father'] ** p

all_predictors = [c for c in df.columns if "Father" in c]

X = df[all_predictors]
y = df['Son']
model = LinearRegression().fit(X, y)

# Inspect the fitted model
coefficients = model.coef_
intercept = model.intercept_
```

TKTK ATUS survey data for regularization


# impute-dependent_censoring
Thie repo implements the missing data imputation method by James Robins for dependent censoring cases where patients dropped out during follow up study. 

For example, we are trying to estimate the relationship between some independent variables (such as radiation dose, gender etc.) and patients' weight at 100 days after radiation treatment using some regression model. We have longitudianl historical data of weight measurements for patients obtained during their follow up visits, but some of them, say 20% dropped out before 100 days after radiation treatment.

If we simply use the complete case patients to estimate the regression between independent variables and weights at 100 days after radiation treatment, there maybe selection bias. This method imputes the missing weights assuming Missing At Random (MAR). In other words, it assumes that the missing weight depend only on observed historical variables including the observed longitudinal weight measuremnts and the missing weight doesn't depend on its own value.

Then we can use the imputed weights data to obtain a asymptotically unbias estimate of the regression between independent variables and weights at 100 days after radiation treatment.


 ### References:
 1. Rotnitzky, Andrea, and James M. Robins. "Semiparametric regression estimation 
 in the presence of dependent censoring." Biometrika (1995): 805-820.

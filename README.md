# ML_algorithms
Basic and advanced ML algorithms with customised functions

#### DBScan ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/DBScan%20clustering%20algorithm.ipynb))
DBscan is clustering algorithm but it, unlike K-means, does not have centroids, so it is more sensitive to the nonlinear patterns of connections between features we want to group and identify hidden paterns. DBScan thus uses radius and group values of the data if they belong in to the area of some hypotesised radius.
In this script we loop for different noise hyperparameter on toy data set and find different precision of solution.

#### Isolation forest ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/Isolation_forest.ipynb))

Isolation Forest, like any tree ensemble method, is built on the basis of decision trees. In these trees, partitions are created by first randomly selecting a feature and then selecting a random split value between the minimum and maximum value of the selected feature. In principle, outliers are less frequent than regular observations and are different from them in terms of values (they lie further away from the regular observations in the feature space). That is why by using such random partitioning they should be identified closer to the root of the tree (shorter average path length, i.e., the number of edges an observation must pass in the tree going from the root to the terminal node), with fewer splits necessary.

In the following example we will use the ISO forest algorithm on a famous boston data set to detect the cities with the highest crime rate.

#### K-means clusters ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/K-means%20clustering.ipynb))

K-means clusters work by initialising the centroids of the data and then categorise the data by some measure of distance (e.g. Eucledian). The backside of this clustering algorithm is that we must assume number of clusters a priori, but then we can check the measure of goodness of various solutions (i.e. k-number of clusters), comapring variance between clasters and within clasters. The bigger the ratio in favor of huge variance between clusters, solution is better. This relatively simple analysis could give very important business value insights in terms of better understaning the charachteristics and typology of the some pheonomena (e.g. customers, products, etc.).

In this script we make 3-D plots on Iris data set as ground truth and varoious solutions, as well as implement elbow rule.

#### Lasso, Ridge and Elastic Net ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/Lasso%2C%20Ridge%20and%20Elastic%20Net.ipynb))

In this script we apply three different types of regularisation on polinomial and regular regression model. Then we plot the parameter shrinkage. 

#### Principal component analysis (PCA) ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/PCA.ipynb))

Principal component analysis is ML technique for feature reduction. It searches in the space of the features, latent vectors that explains the highest amount of variance among original features (original matrix). The latent features are eigenvectors in matrix decomposition with their own eigenvalues of matrix of covariance of original features. In that way, we can search the more fundamental structure of some matrix, and we can explain the high dimensional space of feature with only a few principal components which are dimension that reflect the "inner structure" of our original matrix (e.g. imagine that we applied 20 IQ tests to our subjects, and it will create a matrix with a lot of variance - some subject will underperform on some tests, due to the tiredness, attention, etc. but we should be able to extract one principal component that should reflect the global IQ of the subject). Principal components are usually not correlated (they are ortogonal in our feature space) but some rotation (oblique Promax) allows correlation between components (in our example it could be an verbal and non-verba IQ component).

We do real example with IQ data set. 

#### Polynomial regression ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/Polynomial%20Regression.ipynb))

We did this small practise to illustrate the overfitting problem when we raise the polinomial parameter of regression model. 
Interesting plotting helps in understanding how polinomial regression could help in reducing bias but also how it can lead to overfitting.

Polinomial regression describes polynomial functions in contrast to linear one, which is more complex and describes nonlinear relationships between predictor and target feature. We will do a little play with some fake data as illustration. PolynomialFeatures with degree three for two features a and b adds not only <img src="https://render.githubusercontent.com/render/math?math=a^2">, <img src="https://render.githubusercontent.com/render/math?math=a^3">, <img src="https://render.githubusercontent.com/render/math?math=b^2">, <img src="https://render.githubusercontent.com/render/math?math=b^3"> but also <img src="https://render.githubusercontent.com/render/math?math=a*b">, <img src="https://render.githubusercontent.com/render/math?math=a*b^2">, <img src="https://render.githubusercontent.com/render/math?math=a^2*b">. Some optimisation, like Akaike information criteria is needed to determine the smallest mean square error but in relation to the number of parameters, due to computational complexity.


#### Random Forest ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/Random%20Forest.ipynb))

Random forests are a way of averaging multiple deep decision trees, trained on different parts of the same training set, with the goal of reducing the variance. This comes at the expense of a small increase in the bias and some loss of interpretability, but generally greatly boosts the performance in the final model. Trees that are grown very deep tend to learn highly irregular patterns: they overfit their training sets, i.e. have low bias, but very high variance. Random forests are a way of averaging multiple deep decision trees, trained on different parts of the same training set, with the goal of reducing the variance.

The training algorithm for random forests applies the general technique of bootstrap aggregating, or bagging, to tree learners. Given a training set X with responses Y bagging repeatedly (B times) selects a random sample with replacement of the training set and fits trees to these samples:

Sample, with replacement, n training examples from X, Y; call these Xb, Yb. Random Forest trains a classification or regression tree fb on Xb, Yb. After training, predictions for unseen samples x' can be made by averaging the predictions from all the individual regression trees on x' or by taking the majority vote in the case of classification trees.

We make in this script full RF model on a wine data set with all relevant graphic illustrations.


#### Simple Neural Network from the scratch ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/Simple%20Neural%20Network%20from%20the%20Scratch.ipynb))

With the help of material of Andrew Ng, we make a simple NN from the scratch for a toy data set. Very good illustration gives the intuition of what NN are capable of. 

#### XGBoost ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/XGBoost-IRIS.ipynb))

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. Boosting is an ensemble technique where new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made. A popular example is the AdaBoost algorithm that weights data points that are hard to predict.

Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models. 

We will do a quick XGboost model on an Iris data set with all relevant illustrations.

#### Linear Programming ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/Linear%20Programming/Linear_programming_with_gurobipy_teachers_example.ipynb))

Linear programming is an optimization technique for a system of linear constraints and a linear objective function. An objective function defines the quantity to be optimized, and the goal of linear programming is to find the values of the variables that maximize or minimize the objective function.
Here we created one interesting task. Imagine we have 100 teachers and 10 schools. We have the data what is the distance from each school from each teacher.
We want that our teachers be satisfied, that they does not travell a lot, in our intent, as municipality officer to assign one new teacher to the each school.
We also have some budget constraing and salary expectation for each teacher. How would you do the optimal assignment of teachers to school? By LP!
The formalisation of all equtations could be seen here: ([Teachers.lp](https://github.com/Vitomir84/ML_algorithms/blob/main/Linear%20Programming/TEACHERS.lp))


#### TimeSeries forecasting with SARIMAX ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/Timeseries%20comprehensive.ipynb))

This scripts explains the basic concepts for understanding timeseries: trend, seasonality, white noise, stationarity and makes forecasting example with autoregresive, differencing and moving averages parameters. This is mandatory step for understanding any timeseries data.

#### Automatised search for hyperparameters for SARIMA forecasting model ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/Auto%20ARIMA%20hyperparameter%20search.ipynb))

Here you cand find a tool that will do auto hyperparameter search for autoregressive, differencing, moving average and seasonality parameters for SARIMA forecasting model.

#### SMOTE and Cost Learning for imbalanced dataset in Fraud Detection ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/Online_Payments_Fraud_Detection.ipynb))

This part is not finished and some parts of explanations are in Serbian, but the majority of the code is there, with interesting visualisation of SMOTE syntheting data oversampling. 

#### XGBoost model on breast_cancer_dataset with precision-recall curve, ROC curve and shap values ([Link](https://github.com/Vitomir84/ML_algorithms/blob/main/Breath_cancer_with_shap_values.ipynb))

This part contains interesting dataset with interesting visualisation of importance of particular features in cancer prediction. Shap package offers very interesting visualisation of feature importance for interpretable ML models




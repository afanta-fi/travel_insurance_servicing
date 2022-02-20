# Travel Insurance
Abeselom Fanta Flex 20 Weeks Instructor: Abhineet Kulkarni
# Business Understanding
This project aims to build a predictive model that can predict whether a travel insurance claim is is approved or denied base on the data provided. The data was collected by a third-party travel insurance servicing company that is based in Singapore. The type of data include claim status, agency name, type of travel insurance agency, distribution channel of travel insurance agencies, name of the travel insurance products, duration of travel, destination of travel, amount of sales of travel insurance policies, commission received for travel insurance agency, gender of insured and age of insured. Insurance claims often involve complex decision, which can be costly to the insurers as well as the insured. Therefor, an automated method to screen claims saves time and valuable resources. 
# Data Understanding
The data used in this project was sourced from [Kaggle](https://www.kaggle.com/mhdzahier/travel-insurance) based on a third-party travel insurance servicing company that is based in Singapore. Although there is no mention of date on which the data was collected, based on the nature of destination, it is assumed to be collected in the mid with in the last 12 years. The data covers above 63 thousand travel insurance claims, which is very comprehensive. 

The data has an inordinate amount of class imbalance with only 1.5% of the claims approved. Therefore, class weight and oversampling techniques were used in the project to address the class imbalance. This can be addressed using class weighting in the model or oversampling. For oversampling, the models were cross-validated with SMOTE, ADASYN and RandomOverSample. The models used in this project are logistic regression, k-nearest neighbors, support vector machine, decision tree, random forest, and gradient boosting methods: XGBoost and CatBoost.  

There are 10 features in the data set, with only 9 selected for use. Additional simplification steps, such as replacing destination countries by continents were used to reduce the amount of columns used for processing.   Categorical variables such as agency name, type and product names were one hot encoded. In addition, for this project, a reusable custom grid search class was written which works well with almost all of libraries used for machine learning.  
# Modeling



## Mean Shift Clustering – Titanic Dataset

#### Dataset contains 1309 columns and most of it is NAN, first we deleted the unnecessary columns which do not have effect on our final Classification. Filled NAN places with string “0.0”

![1](/1.PNG)

#### As the columns have Char/String data, its obvious that we need to Encode them, i am using LabelEncoder to encode them and store in new column and delete the origional one. Same for the Sex, EMBARKED
![1](/2.PNG)
![1](/3.PNG)
![1](/4.PNG)
#### Now define X and y, in our case X is everything except “SURVIVED”, and Y is “SURVIVED”

#### new column = cluster_group for storing the LABELS derived from MeanShift Classifier.

#### survival_rates = {} -> empty dict
![1](/5.png)
#### Calculating Survival Rate by dividing the total number of people that survived in that cluster to the total number of people in that cluster.
![1](/6.PNG)
![1](/7.PNG)

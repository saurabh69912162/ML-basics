## Mean Shift Clustering – Titanic Dataset

#### Dataset contains 1309 columns and most of it is NAN, first we deleted the unnecessary columns which do not have effect on our final Classification. Filled NAN places with string “0.0”

![1](/1.png)

#### As the columns have Char/String data, its obvious that we need to Encode them, i am using LabelEncoder to encode them and store in new column and delete the origional one. Same for the Sex, EMBARKED
![1](/2.png)
#### Now define X and y, in our case X is everything except “SURVIVED”, and Y is “SURVIVED”
![1](/3.png)

#### new column = cluster_group for storing the LABELS derived from MeanShift Classifier.
![1](/4.png)
#### survival_rates = {} -> empty dict
![1](/5.png)
#### Calculating Survival Rate by dividing the total number of people that survived in that cluster to the total number of people in that cluster.
![1](/6.png)
![1](/7.png)

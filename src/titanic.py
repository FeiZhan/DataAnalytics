import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# get train & test csv files as a DataFrame
train_df = pd.read_csv("../data/train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("../data/test.csv", dtype={"Age": np.float64}, )

# drop unnecessary columns
train_df = train_df.drop(["PassengerId", "Name", "Ticket"], axis=1)
test_df = test_df.drop(["Name", "Ticket"], axis=1)


# Embarked

# fill the missing values with the most occurred value "S".
train_df["Embarked"] = train_df["Embarked"].fillna("S")

# remove "S" dummy variable, leave "C" & "Q", since they seem to have a good rate for Survival.
embark_dummies_titanic = pd.get_dummies(train_df["Embarked"])
embark_dummies_titanic.drop(["S"], axis=1, inplace=True)
embark_dummies_test = pd.get_dummies(test_df["Embarked"])
embark_dummies_test.drop(["S"], axis=1, inplace=True)

train_df = train_df.join(embark_dummies_titanic)
test_df = test_df.join(embark_dummies_test)
train_df.drop(["Embarked"], axis=1, inplace=True)
test_df.drop(["Embarked"], axis=1, inplace=True)


# Fare

# missing "Fare" values
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert from float to int
train_df["Fare"] = train_df["Fare"].astype(int)
test_df["Fare"] = test_df["Fare"].astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = train_df["Fare"][train_df["Survived"] == 0]
fare_survived = train_df["Fare"][train_df["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])


# Age

# get average, std, and number of NaN values in train_df
average_age_titanic = train_df["Age"].mean()
std_age_titanic = train_df["Age"].std()
count_nan_age_titanic = train_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# fill NaN values in Age column with random values generated
train_df["Age"][np.isnan(train_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

# convert from float to int
train_df["Age"] = train_df["Age"].astype(int)
test_df["Age"]    = test_df["Age"].astype(int)


# Cabin

# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
train_df.drop("Cabin", axis=1, inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)


# Family

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
train_df["Family"] =  train_df["Parch"] + train_df["SibSp"]
train_df["Family"].loc[train_df["Family"] > 0] = 1
train_df["Family"].loc[train_df["Family"] == 0] = 0

test_df["Family"] =  test_df["Parch"] + test_df["SibSp"]
test_df["Family"].loc[test_df["Family"] > 0] = 1
test_df["Family"].loc[test_df["Family"] == 0] = 0

# drop Parch & SibSp
train_df = train_df.drop(["SibSp","Parch"], axis=1)
test_df    = test_df.drop(["SibSp","Parch"], axis=1)


# Sex

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age, sex = passenger
    return "child" if age < 16 else sex

train_df["Person"] = train_df[["Age", "Sex"]].apply(get_person, axis=1)
test_df["Person"] = test_df[["Age", "Sex"]].apply(get_person, axis=1)

# No need to use Sex column since we created Person column
train_df.drop(["Sex"], axis=1, inplace=True)
test_df.drop(["Sex"], axis=1, inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic = pd.get_dummies(train_df["Person"])
person_dummies_titanic.columns = ["Male", "Female", "Child"]
person_dummies_titanic.drop(["Male"], axis=1, inplace=True)

person_dummies_test = pd.get_dummies(test_df["Person"])
person_dummies_test.columns = ["Male","Female","Child"]
person_dummies_test.drop(["Male"], axis=1, inplace=True)

train_df = train_df.join(person_dummies_titanic)
test_df = test_df.join(person_dummies_test)

train_df.drop(["Person"],axis=1,inplace=True)
test_df.drop(["Person"],axis=1,inplace=True)


# Pclass

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(train_df["Pclass"])
pclass_dummies_titanic.columns = ["Class_1", "Class_2", "Class_3"]
pclass_dummies_titanic.drop(["Class_3"], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df["Pclass"])
pclass_dummies_test.columns = ["Class_1","Class_2","Class_3"]
pclass_dummies_test.drop(["Class_3"], axis=1, inplace=True)

train_df.drop(["Pclass"], axis=1, inplace=True)
test_df.drop(["Pclass"], axis=1, inplace=True)
train_df = train_df.join(pclass_dummies_titanic)
test_df = test_df.join(pclass_dummies_test)


# define training and testing sets
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()


# Logistic Regression
# Your submission scored 0.75598
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
#Y_pred = logreg.predict(X_test)
#score = logreg.score(X_train, Y_train) # 0.812570145903


# Support Vector Machines
# Your submission scored 0.65072
svc = SVC()
svc.fit(X_train, Y_train)
#Y_pred = svc.predict(X_test)
#score = svc.score(X_train, Y_train) # 0.883277216611


# Random Forests
# Your submission scored 0.75120
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
score = random_forest.score(X_train, Y_train) # 0.964085297419

# KNN
# Your submission scored 0.63636
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
#Y_pred = knn.predict(X_test)
#score = knn.score(X_train, Y_train) # 0.837261503928


# Gaussian Naive Bayes
# Your submission scored 0.74163
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
#Y_pred = gaussian.predict(X_test)
#score = gaussian.score(X_train, Y_train) # 0.800224466891


# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(train_df.columns.delete(0))
coeff_df.columns = ["Features"]
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# print results
print("score", score)
print("coeff_df", coeff_df)

# create submission dataset
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv("../data/result.csv", index=False)
import pandas as pd
#column names form the adult dataset
print("========================")
print ("==============================================")


columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]
##loading the data using the pandas library

print("======================================================")
df = pd.read_csv(
    "adult.data",
    names = columns,
    skipinitialspace = True
)

##Showing the basic information of the data 
print ("Dataset shape: ")
print (df.shape)


print ("\nFirst 5 rows: ")
print(df.head())

print ("\nColumn names: ")
print(df.columns.tolist())

print("\nData types: ")
print(df.dtypes)  ##printing the data types present in the data 

print("\nMissing values: ")
print (df.isnull().sum) #finding the values that are missing inside our data

print ("\nQuestion marks in each column: ")
for column in df.columns:
    print(column, ":", (df[columns]=="?").sum())
##Features
x = df.drop("income", axis=1)  #drop all other columns except this because is what we will be predicting

#Target
y = df["income"]

print ("\nFeatures (x): ")
print (y.head())

##converting the categorical columns into numbers /integers
X_encoded = pd.get_dummies(x)

print("\nEncoded data: ")
print(X_encoded.head())


print("\nOriginal number of feartures: ", x.shape[1])
print("Number of feartures after encoding: ", X_encoded.shape[1])

from sklearn.model_selection import train_test_split

#splitting the data into training and testing sets
print("===========================================================")
print("===================================================================")

x_train, x_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)
print ("\nTraining data: ")
print("x_train: ",x_train.shape)
print("y_train: ", y_train.shape)

print("\nTesting data: ")
print ("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)

##Scaling the data for good results
from sklearn.preprocessing import StandardScaler

#create the scaler
scaler = StandardScaler()

#Learn scaling parameters from the training data
X_train_scaled = scaler.fit_transform(x_train)

##Apply the same scaling to testing data
X_test_scaled = scaler.transform(x_test)
print("\nScaled data: ")
print ("X_train_scaled: ", X_train_scaled.shape)
print ("X_test_scaled: ", X_test_scaled.shape)

#After scaling we now create the  knnn model 
##we use the model built in the sklearn
print ("=================")
print ("==================================================")

print ("The project is doing goood..................")
print ("=======================================================")


##creating our firts knn model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5)

#Training the model
model.fit(X_train_scaled, y_train)

print ("\nKNN model trained succefully!!!!")

print("===========================================")
#making the predictions
y_pred = model.predict (X_test_scaled)

print("\nFirst 20 Predictions: ")
print(y_pred[:20])

print("\nFirst 20 actual values: ")
print (y_test.iloc[:20].values)

print("=============================")
#measuring the accuracy of the model
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("\nModel accuracy: ", accuracy)
print("Model accuracy (%): ", accuracy*100)

print("==================")
from sklearn.metrics import classification_report

print ("\nClassification report: ")
print (classification_report(y_test, y_pred))

print("====================================================")
#indentfying different values of k

k_values = [1, 3, 5, 7, 9, 11 , 21, 31, 51, 101]
accuracies = []

for k in k_values :
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print (
        "k = ", k,
        "| Accuracy =", round(accuracy * 100, 2), "%"
    )

print("=========================================")
##Testing the cross validation
from sklearn.model_selection import cross_val_score

model = KNeighborsClassifier(n_neighbors=11)

scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv= 5,
    scoring = "accuracy"
)
print("\nCross_validation scores: ")
print(scores)

print("Average CV accuracy: ", scores.mean())
print("Average Cv accuracy(%): ", scores.mean() * 100)
print("========================================")

print("======================================")
##comparing the value of k using the cross validation
k_values = [1, 3, 5, 7, 9, 11 , 21, 31, 51, 101]
cv_results = []

for k in k_values :
    model =KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv= 5,
        scoring = "accuracy"
    )
    average_score = scores.mean()
    cv_results.append(average_score)
    print (
        "k =", k,
        "|Average Cv Accuracy =", 
        round(average_score * 100, 2),
        "%"
    )    

print("==========================================")

#Comparing the values of the training and those of the cross validation
from sklearn.model_selection import cross_validate
k_values = [1, 3, 5, 7, 9, 11 , 21, 31, 51, 101]
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    results = cross_validate(
        model,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy",
        return_train_score = True

    )
    train_accuracy =results["train_score"].mean()
    validation_accuracy = results["test_score"].mean()

    print(
        "k =", k,
        "|Training =", round(train_accuracy*100, 2), "%",
        "|Validation = ", round(validation_accuracy * 100, 2), "%"
    )

print("=========================================")
##Hyperparameter tuning using the gridsearchcv

from sklearn.model_selection import GridSearchCV

model = KNeighborsClassifier()
param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 21, 31, 51, 101],
    "weights": ["uniform", "distance"],
    "p": [1,2]

}
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring = "accuracy",
    n_jobs = 1

)

grid_search.fit (X_train_scaled, y_train)

print("\nBest parameters: ")
print(grid_search.best_params_)

print ("\nBest cross-validation accuracy: ")
print (grid_search.best_score_)

print(
    "\nBest cross-validation accuracy (%): ",
    grid_search.best_score_ * 100
)

##Data visualization


import matplotlib.pyplot as plt
k_graph = [1, 3, 5, 7, 9, 11 , 21, 31, 51, 101]


train_graph = []
validation_graph = []

for k in k_values :
    model =KNeighborsClassifier(n_neighbors= k)
    results = cross_validate(
        model,
        X_train_scaled,
        y_train,
        cv = 5,
        scoring= "accuracy",
        return_train_score =True

    )
    train_graph.append(
        results["train_score"].mean() * 100

    )
    validation_graph.append(
        results["test_score"].mean() * 100

    )
print("\nNumber of k values :", len(k_graph))    
print("Training scores : ", len(train_graph))
print("Validation scores:", len(validation_graph))

plt.plot(
    k_graph,
    train_graph,
    marker="o",
    label = "Training Accuracy"
)

plt.plot(
    k_graph,
    validation_graph,
    marker = "o",
    label ="validation Accuracy"

)
plt.xlabel("k value")
plt.ylabel ("Accuracy (%)")

plt.title("KNN Training vs Validation Accuracy ")

plt.legend ()
plt.show()
                

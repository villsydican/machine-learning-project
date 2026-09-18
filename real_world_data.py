from ucimlrepo import fetch_ucirepo

#Downloading the adult dataset from the uci
adult = fetch_ucirepo(id=2)

##Separating the features and targets of the data
x = adult.data.features 
y = adult.data.targets

##Displaying basic information of the data 
print("Dataset loaded successfully!!")
print ()

print ("Number of rows:", x.shape[0])
print("Number of features: ", x.shape[1])

print ()
print ("Feartures :")
print (x.columns.tolist())

print()
print ("Target: ")
print(y.columns.tolist())

print ()
print ("First 5 rows : ")
print(x.head())

print ()
print ("Target values: ")
print(y.head())
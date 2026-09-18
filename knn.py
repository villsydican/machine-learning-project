from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Traning data
#features
#[squarefeets, price per unit]
x =[
    [20, 12000],
    [30, 15000],
    [12, 10000],
    [18, 10500],
    [50, 20000],
    [100, 40000],
    [120, 45000],
    [200, 85000],
    [250, 100000],
    [300, 125000]

]

y =[
    "Kiserian",
    "Muthaiga",
    "Kilimani",
    "Upper hill",
    "Syokimau",
    "Limuru",
    "Ngong",
    "Lavington",
    "Karen",
    "Runda"
]

##Splittig the data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.5,
    random_state = 42

)
scaler= StandardScaler()

##Scaling the feartures 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
##labels of the data

model =KNeighborsClassifier(n_neighbors=1)

##Training the model

model.fit (x_train,y_train)




##Making the prediction
y_pred = model.predict(x_test)
##calculating the accuracy of our model
accuracy = accuracy_score(y_test, y_pred)
print ("Actual results: ", y_test)
print("Predicted results: ", y_pred)
print("Accuracy: ", accuracy)


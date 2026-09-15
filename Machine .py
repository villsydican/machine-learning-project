#####  target classes
#iris-setosa 
#iris-versicolor 
#iris-verginica
######feartures
#sepal length in cm
#sepal width in cm
#petal length in cm
#petal width in cm
print("Hello! Python is working perfectly.")
print("My machine learning environment is ready!")
import numpy as np

numbers = np.array([10, 20, 30, 40, 50])

print("Numbers:", numbers)
print("Mean:", np.mean(numbers))
print("Maximum:", np.max(numbers))
print("Minimum:", np.min(numbers))
import pandas as pd

data = {
    "Name": ["John", "Mary", "Peter"],
    "Age": [20, 22, 21],
    "Score": [75, 88, 92]
}

df = pd.DataFrame(data)

print(df)
print("\nAverage score:", df["Score"].mean())
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("My First Machine Learning Graph")

plt.show()
import torch

# Create two tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Add them
result = a + b

print("Tensor A:", a)
print("Tensor B:", b)
print("A + B:", result)

print("PyTorch version:", torch.__version__)
print("Using GPU:", torch.cuda.is_available())
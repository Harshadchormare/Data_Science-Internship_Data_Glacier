import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# load the csv file
df = pd.read_csv("IRIS.csv")

print(df.head())

# select independent and dependent variable
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["class"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# feature selection
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# fit the model
classifier.fit(X_train, y_train)

# Make pickle file of the model
pickle.dump(classifier, open("model.pkl", "wb"))

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


# Load data
df = pd.read_csv('iris.csv')

print(df.head())

# select dependent and independent variables
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
Y = df["Class"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate model
classifier = RandomForestClassifier()

# fit model
classifier.fit(X_train, y_train)

# make pickle file of model
pickle.dump(classifier, open('model.pkl', 'wb'))


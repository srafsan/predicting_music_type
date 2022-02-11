import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre']) # Input
y = music_data['genre'] # Output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # train_set is 80% and test_set is 20%

model = joblib.load('music-recommender.joblib') # Loading the saved model

predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)

print("The predicted outcome is:", predictions)
print("The accuracy of the model is:", score)
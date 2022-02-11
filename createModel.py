import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre']) # Input
y = music_data['genre'] # Output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # For measuring accuracy

# Building a model using machine learning algorithm 
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Creating the graphviz
tree.export_graphviz(model, out_file='music-recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True
                    )

joblib.dump(model, 'music-recommender.joblib') # Saving the model
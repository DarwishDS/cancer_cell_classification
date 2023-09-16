import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# import the GaussianNB module 
from sklearn.naive_bayes import GaussianNB

data = load_breast_cancer()

#organize our data - Since, there are 'Target_names' and 'Target', also Features and Their values
label_names=data['target_names']
labels=data['target']
feature_names=data['feature_names']
features=data['data']

print(label_names)
print(labels)
print(features)
print(feature_names)

#Splitting the available data into train and test sets
train, test, train_labels, test_labels= train_test_split(features, labels, test_size = 0.33, random_state = 42)

gnb = GaussianNB()
model = gnb.fit(train, train_labels)

predictions = gnb.predict(test)

print(predictions)
print(accuracy_score(test_labels, predictions))

#Giving New Value As Input and Predicting If Its Cancerous or Not.
sample_feature_values = [
    1.12,   # mean radius
    1.34,   # mean texture
    9.67,   # mean perimeter
    7.2,   # mean area
    0.0897,  # mean smoothness
    0.1024,  # mean compactness
    0.0977,  # mean concavity
    0.0787,  # mean concave points
    0.1753,  # mean symmetry
    0.0621,  # mean fractal dimension
    0.5701,  # radius error
    1.435,   # texture error
    4.707,   # perimeter error
    8.69,   # area error
    0.004,   # smoothness error
    0.0123,  # compactness error
    0.0132,  # concavity error
    0.0059,  # concave points error
    0.0194,  # symmetry error
    0.0032,  # fractal dimension error
    6.89,   # worst radius
    5.45,   # worst texture
    2.4,   # worst perimeter
    2.0,  # worst area
    0.141,   # worst smoothness
    0.346,   # worst compactness
    0.32,    # worst concavity
    0.162,   # worst concave points
    0.32,    # worst symmetry
    0.0897   # worst fractal dimension
]

# Assuming 'new_data' contains the new data you want to classify
new_data = [sample_feature_values]  # Replace with actual feature values

# Use the trained model to predict the class of the new data
new_predictions = gnb.predict(new_data)

# Display the predicted results
if new_predictions[0] == 0:
    print("The cancer cell is predicted to be non-cancerous (Benign).")
else:
    print("The cancer cell is predicted to be cancerous (Malignant).")


# THANK_YOU #
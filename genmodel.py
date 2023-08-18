import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load .pkl
hogvectors_train = pickle.load(open('hogvectors_train.pkl', 'rb'))
hogvectors_test = pickle.load(open('hogvectors_test.pkl', 'rb'))

#extract the features on column 8100
X_train_data = [hogfeature_Xtrain[0:8100] for hogfeature_Xtrain in hogvectors_train]
X_test_data = [hogfeature_Xtest[0:8100] for hogfeature_Xtest in hogvectors_test]

#extract classes
Y_train_data = [hogfeature_Ytrain[-1] for hogfeature_Ytrain in hogvectors_train]
Y_test_data = [hogfeature_Ytest[-1] for hogfeature_Ytest in hogvectors_test]

# Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder() 
y_labelNum_train = label_encoder.fit_transform(Y_train_data)
y_labelNum_test = label_encoder.transform(Y_test_data)

# Train a DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_data, y_labelNum_train)


# model = {0:"Audi",1:"Hyundai Creta",2:"Mahindra Scorpio",3:"Rolls Royce",4:"Swift",5:"Tata Safari",6:"Toyota Innova"}

# Predict labels
y_pred = clf.predict(X_test_data)
# print(model[y_pred[0]])


accuracy = accuracy_score(y_labelNum_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

write_path = "car_model.pkl"
pickle.dump(clf, open(write_path,"wb"))
print("data preparation is done")

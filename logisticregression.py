import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
data = pd.read_csv("machine_failure.csv")
labelencoder = LabelEncoder()
data['type_encoded'] = labelencoder.fit_transform(data['Type'])

x= data[["type_encoded", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",  "TWF", "HDF", "PWF", "OSF", "RNF", "type_encoded"]].values
y= data[['Machine failure']].values
model = LogisticRegression()
model.fit(x,y)
print(model.coef_)
print(model.intercept_)



# Save the model and label encoder
joblib.dump(model, 'model.pkl')
print("Model and label encoder saved as model.pkl ")
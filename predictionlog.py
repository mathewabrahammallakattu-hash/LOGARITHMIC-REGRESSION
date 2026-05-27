import joblib
model = joblib.load("model.pkl")
input_data = [[1, 300, 310, 1500, 20, 10, 0, 0, 0, 0, 0, 0, 1]]
prediction = model.predict(input_data)
print("Prediction for the input data:", prediction)
if prediction[0] == 1:
    print("The machine is likely to fail.")
else:    print("The machine is unlikely to fail.")
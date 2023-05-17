from sklearn.metrics import mean_squared_error

def evaluate_accuracy(model, X_train, y_train):
  y_pred = model.predict(X_train)
  accuracy = mean_squared_error(y_train, y_pred)
  return accuracy
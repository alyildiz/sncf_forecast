def inference(model, scaler, inputs):
    inputs = scaler.transform(inputs)
    preds = model.predict(inputs)
    preds = scaler.inverse_transform([preds])[0]
    return preds

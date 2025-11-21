# --- evolution_model.py ---
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class EvolutionModel:
    def __init__(self, input_dim, output_dim, mapping):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.evolution_map = mapping
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=self.input_dim))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.output_dim, activation="softmax"))
        return model

    def compile_model(self):
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def fit_model(self, X_train, y_train, epochs, batch_size, X_test, y_test):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    def predict_id(self, input_features):
        prediction_probabilities = self.model.predict(input_features, verbose=0)
        predicted_id = np.argmax(prediction_probabilities, axis=1)[0]
        return predicted_id

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return accuracy
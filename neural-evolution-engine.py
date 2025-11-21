import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# --- 1. UPLOADING DATA ---
github_data_url = "https://raw.githubusercontent.com/ralolooafanxyaiml/neural-evolution-sim/refs/heads/main/data.csv"
df = pd.read_csv(github_data_url)

X = df[["METABOLISM", "SKIN", "HABITAT", "SIZE", "DIET", "THREAT"]]
y = df["EVOLUTION_TARGET"]

# --- 2. DATABASES ---
ARCHETYPES = {
    "BIG_CAT": [1, 1, 1, 3, 1],
    "CANINE": [1, 1, 1, 2, 1],
    "HERBIVORE_MEGA": [1, 4, 1, 4, 2],
    "GRAZER": [1, 1, 1, 3, 2],
    "RODENT": [1, 1, 1, 1, 3],
    "RAPTOR_BIRD": [1, 3, 3, 2, 1],
    "WATER_BIRD": [1, 3, 2, 2, 3],
    "SNAKE_LIKE": [0, 2, 1, 2, 1],
    "LIZARD_SMALL": [0, 2, 1, 1, 3],
    "AMPHIBIAN_STD": [0, 4, 2, 1, 3],
    "FISH_PREDATOR": [0, 2, 2, 3, 1],
    "FISH_PREY": [0, 2, 2, 1, 2],
    "INSECT_GIANT": [0, 2, 1, 1, 3],
    "WHALE_LIKE": [1, 4, 2, 4, 1],
    "PRIMATE": [1, 1, 1, 2, 3],
    "BEAR_LIKE": [1, 1, 1, 3, 3],
    "MARINE_REPTILE": [0, 2, 2, 4, 1],
    "MARSUPIAL": [1, 1, 1, 2, 2],
    "DEEP_SEA_FISH": [0, 4, 2, 1, 1],
    "WINGED_INSECT": [0, 2, 3, 0, 3]
}

ANIMAL_DATABASE = {}
ANIMAL_DATABASE["dragon"] = [0, 2, 3, 4, 1]
ANIMAL_DATABASE["alien"] = [1, 4, 1, 2, 3]

THREAT_DATABASE = {}

EVOLUTION_MAPPING = {
    0: [
        "Significant thickening of the dermal layer and fur density for extreme cold insulation.",
        "Development of a dense undercoat and increased subcutaneous fat reserves for thermal regulation.",
        "Adaptation to cold by minimizing surface area and maximizing thermal retention."
    ],
    1: [
        "Skin hardens into keratinous scales or a protective shell, providing robust armor against toxins and heat.",
        "Rapid growth of tough scales or a protective carapace structure to prevent rapid dehydration.",
        "Cellular structure becomes highly resistant to corrosive elements and high UV exposure."
    ],
    2: [
        "Shift to a prolonged torpor or hibernation state to dramatically conserve energy during resource scarcity.",
        "Reduction in basal metabolic rate by 40% to survive extended periods of low food and water.",
        "Evolution of a super-efficient energy storage organ (e.g., camel's hump-like fat reserve)."
    ],
    3: [
        "Lungs/gills become hyper-efficient, maximizing oxygen uptake in environments with depleted air quality.",
        "Development of larger wings or stronger aquatic fins for rapid migration to a new, cleaner habitat.",
        "Evolution of a secondary, low-oxygen tolerance organ to support survival in polluted air/water."
    ],
    4: [
        "Glands develop to secrete defensive toxins, often coupled with aposematic (bright warning) coloration.",
        "Cellular mechanisms rapidly evolve to neutralize pollutants and heavy metals consumed through diet.",
        "Development of a bitter taste or unpalatable texture to deter most common predators."
    ],
    5: [
        "Significant enlargement of eyes (for low-light) or auditory organs (for high-noise) to enhance sensory perception.",
        "Development of specialized sensory organs (e.g., electroreception) to navigate murky or chaotic environments.",
        "Increased brain capacity dedicated to processing complex sensory data and creating a cognitive map."
    ]
}

# --- 3. EVOLUTION CATEGORIES ---
ATTRIBUTE_CATEGORIES = {
    0: "SKIN_INSULATION", 
    1: "SKIN_ARMOR", 
    2: "METABOLISM",
    3: "RESPIRATORY",
    4: "DEFENSE",
    5: "SENSORY"
}

# --- 4. PREPROCESSING ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_encoded = to_categorical(y_train, num_classes=6)
y_test_encoded = to_categorical(y_test, num_classes=6)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. ANIMALS/THREATS OOP ---
def add_animals(archetype_key, animal_list):
    for animal in animal_list:
        if archetype_key in ARCHETYPES:
            ANIMAL_DATABASE[animal] = ARCHETYPES[archetype_key]

def add_threats(threat_id, keywords):
    for word in keywords:
        THREAT_DATABASE[word] = threat_id

# --- 5. DATA TO DATABASES ---
add_animals("BIG_CAT", ["lion", "tiger", "leopard", "jaguar", "cheetah", "panther", "cougar", "lynx"])
add_animals("CANINE", ["wolf", "dog", "fox", "coyote", "jackal", "hyena", "dingo"])
add_animals("HERBIVORE_MEGA", ["elephant", "rhino", "hippo", "giraffe", "dinosaur", "brachiosaurus", "mammoth"])
add_animals("GRAZER", ["horse", "cow", "zebra", "deer", "moose", "camel", "buffalo", "gazelle", "donkey", "sheep", "goat"])            
add_animals("RODENT", ["mouse", "rat", "hamster", "squirrel", "beaver", "rabbit", "hare", "guinea pig"])
add_animals("RAPTOR_BIRD", ["eagle", "hawk", "falcon", "owl", "vulture", "condor"])
add_animals("WATER_BIRD", ["penguin", "duck", "swan", "goose", "pelican", "seagull", "albatross"])
add_animals("SNAKE_LIKE", ["snake", "cobra", "python", "viper", "anaconda", "boa"])
add_animals("LIZARD_SMALL", ["lizard", "gecko", "chameleon", "iguana", "skink"])
add_animals("AMPHIBIAN_STD", ["frog", "toad", "salamander", "newt", "axolotl"])
add_animals("FISH_PREDATOR", ["shark", "great white", "barracuda", "swordfish", "piranha"])
add_animals("FISH_PREY", ["goldfish", "salmon", "tuna", "trout", "cod", "sardine", "clownfish"])
add_animals("WHALE_LIKE", ["whale", "blue whale", "orca", "dolphin", "beluga", "manatee"])
add_animals("PRIMATE", ["human", "monkey", "chimpanzee", "gorilla", "orangutan", "lemur", "baboon"])
add_animals("BEAR_LIKE", ["bear", "grizzly", "polar bear", "panda", "koala"])
add_animals("MARINE_REPTILE", ["crocodile", "alligator", "caiman", "sea turtle", "ichthyosaur"])
add_animals("MARSUPIAL", ["kangaroo", "wallaby", "opossum", "tasmanian devil"])
add_animals("DEEP_SEA_FISH", ["anglerfish", "lanternfish", "hagfish", "jellyfish"])
add_animals("WINGED_INSECT", ["fly", "bee", "wasp", "moth", "butterfly", "dragonfly"])

add_threats(1, ["cold", "freezing", "ice", "ice age", "snow", "blizzard", "arctic", "glacier", "subzero", "frost", "polar vortex", "winter", "hail", "hypothermia", "frozen", "chill", "absolute zero"])
add_threats(2, ["heat", "hot", "fire", "lava", "magma", "volcano", "eruption", "sun", "solar flare", "global warming", "desert", "drought", "dry", "arid", "burning", "scorch", "heatwave", "supernova", "boiling"])
add_threats(3, ["toxin", "toxic", "poison", "pollution", "plastic", "oil spill", "acid", "acid rain", "radiation", "nuclear", "radioactive", "waste", "smog", "virus", "bacteria", "plague", "pandemic", "chemical", "contamination", "venom", "garbage", "trash"])
add_threats(4, ["scarcity", "famine", "starvation", "hunger", "no food", "empty", "poverty", "competition", "overpopulation", "barren", "desolate", "no water", "thirst"])
add_threats(5, ["air", "no air", "oxygen", "suffocation", "choking", "asphyxiation", "vacuum", "space", "underwater", "drowning", "pressure", "predator", "hunter", "enemy", "monster", "alien", "dinosaur", "attack", "war", "hunt"])

# --- 6. AI MODEL ---
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

# --- 7. MACHINE LEARNING ---
input_feature_count = X_train_scaled.shape[1]
output_class_count = y_train_encoded.shape[1]

evolution_sim = EvolutionModel(
    input_dim=input_feature_count, 
    output_dim=output_class_count, 
    mapping=EVOLUTION_MAPPING
)
evolution_sim.compile_model()

evolution_sim.fit_model(
    X_train_scaled, y_train_encoded, 
    epochs=50, batch_size=32, 
    X_test=X_test_scaled, y_test=y_test_encoded
)
final_accuracy = evolution_sim.evaluate_model(X_test_scaled, y_test_encoded)


# --- 7. SIMULATION ---
def start_game_interface():
    print("\n\n####################################################")
    print("#      AI EVOLUTION SIMULATOR (V1.0) - READY       #")
    print("####################################################")
    
    # PICKING ANIMAL
    while True:
        print("\n--- NEW EVOLUTIONARY LINEAGE STARTED ---")
        features = []
        animal_name_display = ""
        
        # PICKING ANIMAL
        while True:
            user_animal = input("\n>> ENTER ANIMAL NAME (or 'exit' to close): ").lower().strip()
            if user_animal == 'exit':
                print("Goodbye!")
                return # CLOSING PROGRAM

            if user_animal in ANIMAL_DATABASE:
                features = ANIMAL_DATABASE[user_animal]
                animal_name_display = user_animal.capitalize()
                print(f"   Organism Selected: {animal_name_display}")
                break
            else:
                print(f"   Unknown Animal. Try: Lion, Wolf, Snake, Shark, Dragon...")

        # INVENTORY)
        current_evolution_attributes = {} 

        # THREAT AND EVOLUTION
        while True:
            print(f"\n   --- Current Organism: {animal_name_display} ---")
            user_threat = input(">> ENTER THREAT (or type 'quit' to change animal): ").lower().strip()
            
            if user_threat == 'quit':
                break 
            
            threat_id = None
            for key in THREAT_DATABASE:
                if key in user_threat:
                    threat_id = THREAT_DATABASE[key]
                    break
            
            if not threat_id:
                print("   Unknown Threat. Try: Cold, Heat, Virus, Predator...")
                continue

            # PREDICT
            input_vector = np.array([features + [float(threat_id)]])
            input_scaled = scaler.transform(input_vector)
            
            # ID
            predicted_id = evolution_sim.predict_id(input_scaled)

            # WORDS TO NUMBERS
            evolution_options = EVOLUTION_MAPPING.get(predicted_id, ["Error"])
            final_description = random.choice(evolution_options)
            
            # CATEGORIES
            category = ATTRIBUTE_CATEGORIES.get(predicted_id, "UNKNOWN")
            current_evolution_attributes[category] = final_description
            
            # RESULTS OF EVOLUTION
            print(f"\n   EVOLUTION TRIGGERED: {category}")
            print(f"   Result: {final_description}")
            
            print("\n   [CURRENT DNA INVENTORY]:")
            if not current_evolution_attributes:
                print("      (No mutations yet)")
            else:
                for cat, desc in current_evolution_attributes.items():
                    print(f"      * {cat}: {desc}")
            print("-" * 50)

# STARTING ENGINE
start_game_interface()
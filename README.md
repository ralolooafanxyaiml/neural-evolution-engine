# Neural Evolution Engine V3.0: Multi-Modal AI Simulation

### Project Overview
**Neural Evolution Engine V3.0** is an advanced Deep Learning application built from scratch using **TensorFlow** and **Keras**. It simulates evolutionary biology principles by predicting the optimal adaptation strategy for a species when faced with catastrophic environmental threats.

**What's New in V3.0? (The Cognitive Leap!)**
V3.0 features a revolutionary **Triple-Branch Multi-Modal Architecture** that handles **three distinct tasks**: processing **Biological Data**, analyzing **Environmental Imagery**, and providing **Natural Language assistance** via a chatbot.

---

### Technical Architecture (Triple-Core Brain)

The system utilizes a sophisticated architecture that combines two main model structures for prediction and one for conversational AI.

#### 1. Simulation Core (Hybrid Prediction)

| Branch | Architecture | Input | Function |
| :--- | :--- | :--- | :--- |
| **Visual Branch (The "Eye")** | CNN (2x Conv2D + MaxPooling) | 64x64 RGB Images | Analyzes visual patterns (snow, fire, toxic waste) to identify the threat. |
| **Biological Branch (The "Brain")** | ANN (Dense Layers) | Encoded Biological Features | Processes organism's physiological constraints and traits. |
| **Fusion Layer** | Concatenate + Softmax | Merged CNN/ANN features | Generates probability distribution for 6 evolutionary outcomes. |

#### 2. Chatbot Core (NLP / Knowledge Assistant)

| Branch | Architecture | Input | Function |
| :--- | :--- | :--- | :--- |
| **Language Branch** | **LSTM (Recurrent Neural Network)** | User Text (Tokenized, Embedded) | Understands questions about Evolutionary Biology (e.g., "What is genetic drift?"). |
| **Function** | **Intent Classification** | Chatbot Model (`evolutionchatbotmodel.h5`) | Provides relevant, pre-trained biological explanations. |

---

### Usage: Hybrid Interaction Modes

When running `main.py`, the user is presented with **THREE MAIN OPTIONS**:

| Option | Mode | Primary AI Used | Description |
| :---: | :--- | :--- | :--- |
| **1** | **Simulation Mode (Hybrid)** | **CNN + ANN** | Predicts the optimal evolutionary adaptation based on visual threats and organism traits. |
| **2** | **AI Assistant Mode** | **LSTM (NLP)** | Answers user queries regarding evolutionary concepts and biological definitions. |
| **3** | **Quit** | - | Exits the program. |

---

### Tech Stack
* **Deep Learning:** TensorFlow, Keras (Functional API)
* **Sequence Modeling:** **LSTM (New!)**
* **Computer Vision:** OpenCV (Image Preprocessing)
* **Data Engineering:** Pandas, NumPy
* **Preprocessing:** Scikit-Learn (StandardScaler, LabelEncoder)

---

### Installation & Run

1. **Clone the Repository**
```bash
git clone [https://github.com/ralolooafanxyaiml/neural-evolution-engine]
cd Neural-Evolution-Engine
pip install tensorflow pandas numpy scikit-learn opencv-python
```
2. **Train the NLP Chatbot Model (One Time Setup)**
```bash
python chatbot_train.py
```
3. **Run the Main Engine**
```bash
python main.py
```
Data Sources & Acknowledgements
This project utilizes external datasets for training the Visual Threat Detection (CNN) module:

Intel Image Classification by Puneet Bansal (Cold/Ice)

Natural Disaster Images by Aseem Arora (Heat/Fire)

Garbage Classification by Sashaank Sekar (Toxin/Pollution)

Underwater Image Classification by Great Sharma (Airless/Aquatic)

US Drought Data (Scarcity)

Developed by Mustafa İlker Aktaş - Global AI Contributor

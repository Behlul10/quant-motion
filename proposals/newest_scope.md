# Newer scope

Aside from comparing smoothing from a quantum computer and a classical computation. A new

coordinates given a processed video input and not generate novel motion, but perform pose estimation.

# Data Flow


Gesture Video —> MediaPipe —> Generated Landmarks
Gesture Video —> PCA extractor —> PCA Data

The quantum model learns the mapping from some representation (PCA data) of the video to the
landmarks like a quantum machine learning regressor.

# Evaluation


The quantum model will train on landmark data, and PCA data of 10 videos. To test the trained quantum
model I will input PCA data of 2 new videos where

-​ One video has very similar motion to the other videos
-​ One video has more unique motion compared to the other videos

# Deliverables


-​ GitHub Repository:

-​ notebooks/

-​ 01_data_loading.ipynb (visualising data)
-​ (still figuring this out)
-​ src/ module with reusable functions

-​ landmarks.py
-​ pcaExtract.py
-​ qiskit_test.py (testing quantum circuit)
-​ qiskit_main.py (quantum circuit, training & generating landmarks)
-​ data/

-​ pca/
-​ landmarks/
-​ quant-gen-vids/ (2 new videos for quantum LM gen)
-​ training-vids/ (10 videos for training)
-​ Dataset: 12 sample hand motions (raw + classical + quantum processed)

-​ PCA Data


-​ Landmark Data
-​ Quantum Simulated Landmark Data
-​ Quantum Generated Landmark Data
-​ Godot Visualization Simulation Demo

-​ Imported robotic hand
-​ Animated landmarks showcasing generated landmarks trained on, and the quantum
generated landmarks
-​ This will be available on itch as a web demo


Problem (Cant feed the video directly to a quantum computer):
IBM gives free 10 minutes of time and 100 Quibits a month for students. A single video frame can be
1920×1080 = over 2 million values. This is too many to feed to a quantum computer with limited qubits.

Solution: Principle Component Analysis (PCA) is a technique in machine learning that finding the most
important patterns in the data and discarding the rest. There was a research paper arXiv:2408.03351

[quant-ph] that compressed images from 784 pixels down to 5 core principle components to match 5
qubits. Instead of feeding in the video I will feed in the important patterns in the video that the PCA
technique generated.

Overview: ​
I will be generating PCA Data from videos where the data represents the most important patterns in the
video. Then I will have both Landmarks from the video and PCA data. Where I will be using QML
Techniques to generate landmarks of two new videos given their PCA data to the quantum computer.

Game Engine Hand Visualisation Implementation:
Import mechanical robot hand files into Godot:

-​ [https://github.com/RobotLocomotion/models/tree/master/allegro_hand_description/urdf](https://github.com/RobotLocomotion/models/tree/master/allegro_hand_description/urdf)
Record the animated hand with the following landmarks applied:

-​ Landmarks generated from the original video



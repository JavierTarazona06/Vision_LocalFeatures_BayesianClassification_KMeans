# ComputerVision_LocalFeatures_BayesianClassification_and_KMeansClustering

## 3.1. Bayesian pixel classification

Run these steps from the project root.

1. Create and activate the virtual environment (if it does not exist yet):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies in the virtual environment:

```bash
pip install -r requirements.txt
```

3. Visualize and save color components (RGB, HSV, YCrCb):

```bash
python3 src/Display_Components.py img/Essex_Faces/94/asamma.19.jpg --space all --save-dir outputs/components
```

4. Train and test Bayesian pixel classification:

```bash
python3 src/Bayes_Model_Training.py img/Essex_Faces/94/asamma.19.jpg --features ycrcb --model qda --decision map --test-image img/Essex_Faces/96/arwebb.19.jpg
```

Notes:
- During training, draw rectangular RoIs with the mouse.
- Press `p` for skin, `n` for non-skin, and `q` to finish training.

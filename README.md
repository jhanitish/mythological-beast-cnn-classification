# Mythological Creature Classification

This repository contains code and instructions for training and testing a machine learning model that classifies mythological creatures using a modified MobileNetV2 CNN architecture. The project includes two main notebooks:
- **`hyperparameters-mythological-beast-vision.ipynb`**: Used for hyperparameter tuning.
- **`mythological_beast_vision.ipynb`**: Contains the main model for training and testing on the mythological creature dataset.

The trained models are saved in the following directories:
- **Hyperparameter-tuned model**: `hyperparams-model/`
- **Standard model**: `models/`

---

## System Requirements

- **Operating System**: Windows, Linux, or macOS.
- **Python Version**: Python 3.8 or higher.
- **Libraries/Dependencies**:
  - TensorFlow >= 2.0
  - Keras
  - NumPy
  - Matplotlib
  - OpenCV
  - scikit-learn
  - Google Colab (for cloud execution)
  - CUDA-enabled GPU (for local GPU execution)

---

## Running on Google Colab

Google Colab provides free GPU access, making it a good option for training models on large datasets.

### Steps to Run the Code in Google Colab:

1. **Access Google Colab**:
   - Open your web browser and navigate to [Google Colab](https://colab.research.google.com/).

2. **Upload the Notebooks**:
   - Upload the notebooks `hyperparameters-mythological-beast-vision.ipynb` and `mythological_beast_vision.ipynb` by clicking on the “Upload” button under the “File” menu.

3. **Setting up the GPU**:
   - Go to **Runtime** > **Change runtime type**.
   - Set **Hardware accelerator** to **GPU**.

4. **Install Dependencies**:
   ```bash
   !pip install tensorflow keras opencv-python scikit-learn matplotlib
   ```

5. **Load the Dataset**:
   - Upload your dataset to Colab or link it from a cloud service like Google Drive. You can mount Google Drive using the following code:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Once mounted, navigate to the location of your dataset.

6. **Running the Hyperparameter Tuning Notebook**:
   - Open `hyperparameters-mythological-beast-vision.ipynb`.
   - Run all cells sequentially to train the model with different hyperparameter configurations.
   - After tuning, the trained model will be saved in the `hyperparams-model/` directory.

7. **Running the Core Model Notebook**:
   - Open `mythological_beast_vision.ipynb`.
   - Run all cells to initiate the training, and the model will be saved in the `models/` directory.

8. **Model Evaluation**:
   - After training, evaluate the model's performance using validation data. The notebook will generate confusion matrices and other metrics.
   - Visualize model predictions using provided sample images.

9. **Download the Model**:
   ```python
   from google.colab import files
   files.download('/content/models/models.h5')
   ```

---

## Running on Local GPU Machine

For users with access to a local machine equipped with a CUDA-enabled GPU, follow these steps:

### Steps to Run the Code on Local GPU:

1. **Install CUDA and cuDNN**:
   - Ensure you have installed CUDA and cuDNN compatible with TensorFlow. You can check the [official TensorFlow GPU support page](https://www.tensorflow.org/install/gpu) for detailed instructions.

2. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv myth-creature-env
   source myth-creature-env/bin/activate  # For Linux/Mac
   myth-creature-env\Scripts\activate  # For Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install tensorflow-gpu keras opencv-python scikit-learn matplotlib
   ```

4. **Clone the Repository or Copy Files**:
   - Copy the notebooks `hyperparameters-mythological-beast-vision.ipynb` and `mythological_beast_vision.ipynb` to your local machine.

5. **Set Up Dataset**:
   - Ensure the dataset is available on your machine and placed in the correct directory.
   - Modify the paths in the notebooks to point to the dataset on your local system.

6. **Running the Hyperparameter Tuning Notebook**:
   - Open `hyperparameters-mythological-beast-vision.ipynb` in your Jupyter environment or directly in your IDE (e.g., VSCode, PyCharm).
   - Execute the notebook to train the model with different hyperparameter settings.
   - The tuned model will be saved in the `hyperparams-model/` directory.

7. **Running the Core Model Notebook**:
   - Open `mythological_beast_vision.ipynb` and execute the notebook to train the core model.
   - The final model will be saved in the `models/` directory after training.

8. **Training the Model with GPU**:
   - TensorFlow should automatically detect the GPU. You can verify by running:
     ```python
     from tensorflow.python.client import device_lib
     print(device_lib.list_local_devices())
     ```

9. **Model Evaluation**:
   - The notebook includes code to generate evaluation metrics and visualize predictions. Execute the respective cells after training is complete.

---

## Common Troubleshooting

- **Issue: "No GPU detected" in Google Colab or Local Machine**:
  - Ensure you have selected the GPU in Colab runtime settings or installed the correct GPU drivers (CUDA and cuDNN) for your local machine.

- **Issue: Model overfitting**:
  - If you notice overfitting, consider increasing data augmentation in the code, adjusting learning rates, or adding more regularization (dropout, weight decay).

- **Issue: Low accuracy**:
  - If accuracy is low, review your dataset quality. You may need to expand your dataset or fine-tune the learning rate or batch size.

---

## Contact and Further Assistance

If you encounter any issues or need further assistance, you can contact the project author at [pingnitish@gmail.com]. For updates or to contribute to this project, please visit the project repository on [GitHub]([https://github.com/username/repository](https://github.com/jhanitish/mythological-beast-cnn-classification)).



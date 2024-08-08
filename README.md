# Detect_cassava_diesease_using_computer_vision

# Cassava Leaf Disease Classification

I developed this project to classify cassava leaf diseases using deep learning techniques. It was part of a competition hosted on Kaggle, which aimed to improve the accuracy of detecting various diseases that affect cassava plants. The model I built can classify cassava leaves into one of five categories: Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD), Cassava Green Mottle (CGM), Cassava Mosaic Disease (CMD), and Healthy.

#Project Overview

Cassava is a vital crop in many developing countries, providing a significant source of calories. However, its production is often threatened by various diseases. The purpose of this project is to leverage machine learning, specifically convolutional neural networks (CNNs), to accurately classify cassava leaf diseases, aiding in timely and precise treatment to ensure crop health and yield.

### Dataset

The dataset used for this project is provided by the Cassava Leaf Disease Classification competition on Kaggle. It contains 21,367 labeled images of cassava leaves categorized into one of five classes.

Number of classes: 5
Total images: 21,367
Training images: 17,115
Validation images: 4,282

### Model Architecture

I built the model using TensorFlow and Keras. I experimented with various architectures, including:

ResNet50
EfficientNet
VGG16
I employed transfer learning, using pre-trained weights from ImageNet, and fine-tuned them for this specific classification task.


## Training

The model was trained on Google Colab with the following setup:

Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Epochs: 50
Batch Size: 32
I applied data augmentation techniques to enhance the model's robustness and prevent overfitting.

## Results

The final model achieved an accuracy of 85% on the validation set. Below is a visualization of the model’s performance:

### Acknowledgements

* Kaggle for hosting the competition and providing the dataset.

* TensorFlow and Keras for the powerful deep learning tools.

* The open-source community for the resources and tutorials that helped make this project possible.



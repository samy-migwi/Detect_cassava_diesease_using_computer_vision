# Cassava Leaf Disease Classification

![Cassava Leaf Disease Classification](https://github.com/samy-migwi/Detect_cassava_diesease_using_computer_vision/blob/main/data/thumbnail.png?raw=true)

*Classify cassava leaf diseases using deep learning techniques.*

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview
Cassava is a vital crop in many developing countries, providing a significant source of calories. This project leverages convolutional neural networks (CNNs) to accurately classify cassava leaf diseases, aiding in timely and precise treatment to ensure crop health and yield.

## Dataset
The dataset used for this project is from the Cassava Leaf Disease Classification competition on Kaggle.

- **Number of classes**: 5
- **Total images**: 21,367
- **Training images**: 17,115
- **Validation images**: 4,282

## Model Architecture
The model was built using TensorFlow and Keras. Key architectures experimented with include:

- ResNet50
- EfficientNet
- VGG16

Transfer learning was utilized with pre-trained weights from ImageNet, which were fine-tuned for this specific task.

## Training
Training details:

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy//85
- **Epochs**: 10
- **Batch Size**: 16

Data augmentation techniques were used to improve model performance and reduce overfitting.

## Evaluation
The model achieved an accuracy of 85% on the validation set. Metrics:

## Results
The final model achieved 85% accuracy. Visualizations of model performance can be found below:

*Include relevant visualizations here*

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
- Kaggle for hosting the competition and providing the dataset.
- TensorFlow and Keras for the deep learning tools.
- The open-source community for their resources and tutorials.

---

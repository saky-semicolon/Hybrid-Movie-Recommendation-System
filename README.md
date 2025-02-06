# Hybrid Movie Recommendation System Using Deep Learning and Transfer Learning Approaches

## Overview
This repository contains the implementation of a **Hybrid Movie Recommendation System** that leverages **Deep Learning** and **Transfer Learning** techniques. The system is designed to predict movie ratings and provide personalized movie recommendations based on user preferences and movie metadata.

## Features
- **Deep Learning Model**: Utilizes embedding layers for movie IDs and titles to learn intricate patterns.
- **GloVe Transfer Learning Model**: Employs pretrained GloVe embeddings to incorporate semantic information from movie titles.
- **Fine-Tuned GloVe Model**: Enhances pretrained embeddings by fine-tuning them on the dataset for improved accuracy.
- **Evaluation Metrics**: Models are assessed using **Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score**.
- **Scalability**: Designed to support future enhancements with additional features like user interactions and genre preferences.

## Technologies Used
- **Python**
- **TensorFlow & Keras**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **GloVe Pretrained Embeddings**

## Dataset
The models were trained on **TMDB 5000 Movies Dataset** and **TMDB 5000 Credits Dataset**, containing:
- **Movie Metadata**: Budget, revenue, genres, popularity, runtime, vote counts, etc.
- **Cast & Crew Data**: Actors, directors, and production crew information.

## Model Architectures
### 1. Deep Learning Model
- **Embedding layers** for movie IDs and titles
- **Dense layers** for feature extraction
- **Dropout layers** for regularization
- **Adam optimizer** with MSE loss function

### 2. GloVe Transfer Learning Model
- Uses **pretrained GloVe embeddings** for movie titles
- Embeddings are **frozen** to retain pretrained semantic knowledge

### 3. Fine-Tuned GloVe Model
- Similar to GloVe model but allows **trainable embeddings**
- Embeddings are updated during training for **domain-specific adaptation**

## Performance Evaluation
| Model | MAE | RMSE | R² |
|--------|------|------|------|
| **Deep Learning Model** | 0.9935 | 1.3082 | -0.3203 |
| **GloVe Transfer Learning Model** | 1.3894 | 1.8442 | -1.6241 |
| **Fine-Tuned GloVe Model** | 1.1930 | 1.4838 | -0.6986 |

**Key Takeaways:**
- The **Deep Learning Model** achieved the **lowest error rates**, making it the most effective.
- The **Fine-Tuned GloVe Model** improved over the frozen GloVe model by adapting embeddings to the dataset.
- All models showed potential but need further refinements for better **explanatory power (R² values)**.

## Installation & Usage
### Prerequisites
- Python 3.8+
- TensorFlow & Keras
- Pandas, NumPy, Matplotlib, Seaborn

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/hybrid-movie-recommendation.git
   cd hybrid-movie-recommendation
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```sh
   python train.py
   ```

## Future Enhancements
- **Incorporating User Behavior**: Adding user interactions and watch history.
- **Hybrid Techniques**: Combining deep learning with traditional collaborative filtering.
- **Deployment**: Creating a web app for real-time recommendations.

## Contributors
- **S M Asiful Islam Saky** (Developer & Researcher)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

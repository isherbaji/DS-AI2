Project README: A Cross-Domain Evaluation of Local and Global SHAP Explanations

This project conducts a cross-domain evaluation of SHAP (SHapley Additive exPlanations) to assess the transparency, explainability, and accountability of AI models across tabular, text, and image data.

Introduction

As machine learning models are increasingly deployed in critical sectors like finance, healthcare, and legal services, their "black box" nature poses challenges to fairness, accountability, and trust. Regulations such as the General Data Protection Regulation (GDPR) and the EU AI Act emphasize the right of individuals to understand automated decisions, making explainability a legal and ethical imperative. This project evaluates the effectiveness of SHAP, a widely used explainability tool, across different data modalities to explore its consistency and robustness.

Research Questions

This project aims to answer the following research questions:

1. Cross-Modality Consistency: How consistently do SHAP local and global explanations identify the most influential input features across tabular, text, and image prediction tasks?
2. Local Fidelity Under Feature Removal: To what extent does removing the top-k features ranked by SHAP affect model performance?

Methodology

The project employs a cross-domain evaluation using three distinct datasets and models:

Dataset: NYPD Complaint Data
Type: Tabular
Model: XGBoost

Dataset: IMDb Movie Reviews
Type: Text
Model: DistilBERT

Dataset: ImageNet
Type: Image
Model: ResNet50

Tabular Data: NYPD Complaint Data

An XGBoost model is trained on the NYPD Complaint Data Historic dataset to predict the category of law violations (felony, misdemeanor, violation). Feature selection is performed using ANOVA F-scores to identify the most relevant features. The model's predictions are then explained using SHAP to determine feature importance.

Text Data: IMDb Movie Reviews

A DistilBERT model is used for sentiment analysis on the IMDb movie review dataset. SHAP is applied to explain the model's predictions, identifying words that contribute most to positive or negative sentiment classifications.

Image Data: ImageNet

A ResNet50 model is used for image classification on the ImageNet dataset. SHAP is employed to generate explanations for the model's predictions, highlighting the pixels that are most influential in determining the image's class.

Setup and Usage

To run the scripts and notebooks in this project, please follow these steps:

1. Clone the repository.
2. Install the required dependencies:
pip install -r requirements.txt
3. Download the "crime dataset.csv" for the "1_" scripts. This dataset must be downloaded from [https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data) before you can run the code.
4. Run the notebooks in the specified order:
- Start with the presentation to get an overview of the project.
- Run the "1_" series of notebooks for the tabular data analysis.
- Run the "2_" series of notebooks for the text data analysis.
- Run the "3_" notebook for the image data analysis.

For a detailed analysis of the results, please refer to the "Research Questions Answers.txt" file.

License

This project is licensed under the MIT License.

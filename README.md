# Loan Eligibility Predictor

![GitHub](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

An application for prediction loan eligibility for applicants based on their personal, financial, and property-related information.

![image](./assets/main.png)
![image](./assets/results.png)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Using pip](#using-pip)
  - [Using Conda](#using-conda)
- [Usage](#usage)
- [License](#license)

---

## Overview

The **Loan Eligibility Predictor** is a tool built to assist banks and financial institutions in determining whether an applicant qualifies for a loan. Trained on a Random Forest Classifier, which is then deployed as a web application using Streamlit. The app allows users to input their details and receive instant feedback on their loan eligibility, along with a confidence score indicating the model's certainty.

---

## Features

- Predicts loan eligibility in real-time.
- Provides a confidence score for each prediction.
- User-friendly interface powered by Streamlit.
- Support for both neural network and random forest classifier models.

---

## Installation

You can install and run this application using either pip or conda.

### Using pip

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SamarthPyati/LoanEligibilityPredictor.git
   cd loan-eligibility-predictor
   ```

2. **Set Up a Virtual Environment** (Optional but Recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Using Conda

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SamarthPyati/LoanEligibilityPredictor.git
   cd loan-eligibility-predictor
   ```

3. **Create and Activate a New Conda Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate loan-predictor
   ```

4. **Alternative: Manual Conda Setup** (if environment.yml is not working):
   ```bash
   conda create -n loan-predictor 
   conda activate loan-predictor
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Enter Applicant Information**:
   Fill in the required fields with the applicant's details.

3. **Get Prediction**:
   Click the "Predict" button to receive the loan eligibility prediction.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

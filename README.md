# multiple-linear-regression-normal-equation
#

A machine learning project that implements Multiple Linear Regression from scratch using the Normal Equation without relying on scikit-learn regression models.


# Project Idea

The goal of this project is to predict product sales based on different advertising platforms:

- YouTube
- Facebook
- Newspaper

Instead of using ready-made regression models, the project manually computes the optimal theta parameters using linear algebra and the Normal Equation.


# Dataset

The dataset contains marketing spending data and corresponding sales values.

Features:
- YouTube advertising budget
- Facebook advertising budget
- Newspaper advertising budget

Target:
- Sales


# Technologies Used

- Python
- NumPy
- Pandas


# Machine Learning Concepts Used

This project demonstrates understanding of:

- Multiple Linear Regression
- Normal Equation
- Matrix Multiplication
- Matrix Transpose
- Matrix Inverse
- Feature Matrix Construction
- Bias Term
- Theta Optimization
- Linear Algebra in Machine Learning

---

# Mathematical Formula

The model uses the Normal Equation:

θ = (XᵀX)⁻¹ Xᵀ y

Where:

- X → Feature matrix
- Xᵀ → Transpose of X
- y → Target values
- θ → Optimal model parameters



# Project Workflow

## 1. Load Dataset

The dataset is loaded using pandas.

```python
data = pd.read_csv("market_data.csv")
```

---

## 2. Extract Features and Labels

```python
y = data["sales"]
X = data.drop("sales", axis=1)
```

---

## 3. Add Bias Column

A column of ones is added manually to represent θ₀.

```python
x_bias = np.ones((n,1))
```

---

## 4. Build Final Feature Matrix

```python
final = np.append(x_bias, X, axis=1)
```

---

## 5. Compute Theta using Normal Equation

```python
theta = (XᵀX)⁻¹ Xᵀ y
```

Implemented manually using NumPy operations.

---

## 6. Prediction

The learned theta values are used to predict sales values for new advertising budgets.

---

# Example Prediction

Input:
- YouTube = 84
- Facebook = 19
- Newspaper = 48

Predicted Sales:

```python
11.09
```

---

# Learning Outcomes

Through this project I learned:

- How Linear Regression works mathematically
- How Gradient Descent differs from Normal Equation
- How theta values are computed manually
- The importance of matrix operations in machine learning
- How machine learning models optimize cost functions

---

# Future Improvements

Possible future improvements:

- Add Gradient Descent implementation
- Compare Normal Equation with sklearn LinearRegression
- Add MSE and R² evaluation metrics
- Visualize predictions and regression lines
- Add feature normalization
- Implement Polynomial Regression

---

# Author

saeed hafez

ml and sw engineer Student  
Interested in AI, Machine Learning, and NLP.

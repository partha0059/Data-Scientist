# 🍕 Pizza Predictor - Decision Tree Classifier

A sleek and interactive web application that uses a Decision Tree machine learning model to predict whether you should eat pizza based on your hunger level and the day of the week. Built with Flask, featuring a professional glassmorphic UI with stunning animations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![ML](https://img.shields.io/badge/ML-Decision%20Tree-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-red.svg)

## 🎯 Project Overview

This project demonstrates the implementation of a **Decision Tree Classifier** using the **Entropy criterion** to make binary predictions. The application features a modern glassmorphic UI design and provides an intuitive interface for making predictions.

### Key Features

- ✨ **Modern Glassmorphic UI** - Professional, responsive design with smooth animations
- 🤖 **Machine Learning Model** - Decision Tree Classifier with Entropy criterion
- 🚀 **Real-time Predictions** - Instant results based on user input
- 📊 **Simple & Intuitive** - Easy-to-use interface for demonstrations

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn (Decision Tree Classifier)
- **Data Processing**: pandas
- **Frontend**: HTML5, CSS3 (Glassmorphism design)
- **Model Persistence**: pickle

## 📋 Prerequisites

Before running this project, ensure you have:

- Python 3.8 or higher
- pip (Python package installer)

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/partha0059/-Pizza-Predictor-Decision-Tree.git
cd -Pizza-Predictor-Decision-Tree
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional)

The repository includes a pre-trained model (`decision_tree_model.pkl`), but you can retrain it:

```bash
python train_model.py
```

### 5. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## 🎮 How to Use

1. Open your web browser and navigate to `http://localhost:5000`
2. Select your hunger status: **Am I Hungry?** (Yes/No)
3. Select the day type: **Is It Weekend?** (Yes/No)
4. Click the **PREDICT** button
5. View the prediction result! 🍕

## 📁 Project Structure

```
Pizza-Predictor-Decision-Tree/
│
├── app.py                      # Flask application (main file)
├── train_model.py              # Model training script
├── decision_tree_model.pkl     # Pre-trained Decision Tree model
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
│
├── templates/
│   └── index.html             # Main HTML template
│
└── static/
    ├── style.css              # Glassmorphic CSS styling
    └── images/
        └── pizza_slice.png    # Pizza icon
```

## 🧠 How It Works

### Decision Tree Model

The application uses a **Decision Tree Classifier** with the following configuration:

- **Criterion**: Entropy (Information Gain)
- **Max Depth**: 3
- **Random State**: 42 (for reproducibility)

### Training Data

The model is trained on a simple dataset with three features:

| Am I Hungry | Is It Weekend | Shall I Eat Pizza |
|-------------|---------------|-------------------|
| Yes (1)     | Yes (1)       | Yes (1)          |
| Yes (1)     | No (0)        | Yes (1)          |
| Yes (1)     | Yes (1)       | Yes (1)          |
| No (0)      | Yes (1)       | No (0)           |
| No (0)      | No (0)        | No (0)           |

The model learns patterns from this data to make predictions on new inputs.

## 🎨 UI Design

The application features a **professional glassmorphic design** with:

- Soft gradient backgrounds
- Translucent glass-effect cards
- Smooth animations and transitions
- Responsive layout for all devices
- Modern typography (Poppins font)

## 📊 Model Performance

The Decision Tree model achieves perfect accuracy on the training dataset, making it ideal for this demonstration project. The simple decision rules are:

```
IF Hungry = Yes THEN Pizza = Yes
IF Hungry = No THEN Pizza = No
```

## 🔧 Customization

### Modify the Training Data

Edit the dataset in `train_model.py` to change the model's behavior:

```python
data = {
    'Am_I_Hungry': ['Yes', 'Yes', 'Yes', 'No', 'No'],
    'Is_It_Weekend': ['Yes', 'No', 'Yes', 'Yes', 'No'],
    'Shall_I_Eat_Pizza': ['Yes', 'Yes', 'Yes', 'No', 'No']
}
```

### Change the UI Theme

Modify `static/style.css` to customize colors, animations, and styling.

## 🚢 Deployment

This application can be deployed to various platforms:

- **Heroku**: Add a `Procfile` with `web: python app.py`
- **Render**: Use the Flask configuration
- **PythonAnywhere**: Upload files and configure WSGI

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open-source and available under the MIT License.

## 👨‍💻 Author

**Partha Sarathi R**

- GitHub: [@partha0059](https://github.com/partha0059)

## 🙏 Acknowledgments

- Built with Flask and scikit-learn
- Glassmorphic UI inspired by modern design trends
- Created as an educational project to demonstrate Decision Tree classification

---

<div align="center">
  <strong>Made with ❤️ and 🍕</strong>
  <br>
  <em>A Decision Tree Classifier Project</em>
</div>

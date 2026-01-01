# InsurSence - Insurance Prediction Web App

InsureSence is a comprehensive web application built with Flask that provides personalized insurance premium predictions, policy recommendations, and AI-powered support. It uses machine learning models to analyze user data and predict insurance costs, claim probabilities, and suitable policy types.

## Features

- **User Authentication**: Secure signup and login with OTP verification via email
- **Insurance Prediction**: Predict annual premiums, monthly costs, and claim probabilities based on user health data
- **Policy Recommendations**: Get personalized policy suggestions with probability scores
- **AI-Powered Support**: Integrated Gemini AI for real-time customer support and personalized health advice
- **Prediction History**: View and manage past insurance predictions
- **PDF Reports**: Generate downloadable PDF reports of insurance predictions and advice
- **Responsive Design**: Clean, user-friendly interface with responsive templates

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: Scikit-learn, Joblib (for model loading and predictions)
- **AI Integration**: Google Gemini AI API
- **Database**: SQLite (via custom database utilities)
- **Email Service**: SMTP for OTP verification
- **Frontend**: HTML, CSS, Jinja2 templates
- **PDF Generation**: ReportLab
- **Environment Management**: python-dotenv

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- A Google Gemini API key
- SMTP email credentials for OTP functionality

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/utk712/Major-p.git
   cd Major-p
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Fill in your configuration:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     EMAIL_USER=your_email@gmail.com
     EMAIL_PASS=your_email_password
     ```

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Access the app:**
   Open your browser and go to `http://127.0.0.1:5000`

## Usage

1. **Home Page**: Introduction to the application
2. **Sign Up**: Create an account with email verification via OTP
3. **Login**: Access your account
4. **Predict**: Fill in your health details to get insurance predictions
5. **History**: View your past predictions
6. **Ask Support**: Get AI-powered assistance
7. **Download PDF**: Generate reports of your predictions

## Project Structure

```
Major-p/
├── app.py                      # Main Flask application
├── training.py                 # ML model training script
├── debug_gemini.py             # Gemini AI debugging utilities
├── list_models.py              # Model listing utilities
├── test_gemini_call.py         # Gemini API testing
├── requirements.txt            # Python dependencies
├── insurance.db                # SQLite database
├── .env                        # Environment variables
├── .env.example                # Environment template
├── models/                     # Trained ML models
│   ├── premium_model.joblib
│   ├── policy_model.joblib
│   ├── claim_model.joblib
│   └── policy_label_encoder.joblib
├── data/                       # Training data
│   └── data.csv
├── utils/                      # Utility modules
│   ├── database.py             # Database operations
│   └── emails_otp.py           # Email and OTP functionality
├── template/                   # HTML templates
│   ├── home.html
│   ├── signup.html
│   ├── login.html
│   ├── verify.html
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   ├── ask.html
│   └── support.html
└── static/                     # Static files (CSS, JS, images)
    └── style.css
```

## API Endpoints

- `GET /` - Redirect to home
- `GET /home` - Home page
- `GET/POST /signup` - User registration
- `GET/POST /verify` - OTP verification
- `GET/POST /login` - User login
- `GET /logout` - User logout
- `GET /index` - Prediction form
- `POST /predict` - Process prediction
- `GET /history` - View prediction history
- `GET/POST /ask` - AI support chat
- `GET /support` - Support page
- `POST /support_api` - API endpoint for support
- `GET /download_pdf` - Download PDF report

## Machine Learning Models

The application uses three trained models:

1. **Premium Model**: Predicts annual insurance premium
2. **Policy Model**: Recommends suitable insurance policies
3. **Claim Model**: Estimates claim probability

Models are trained on health insurance data including age, BMI, smoking status, region, and other factors.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Flask web framework
- Powered by Google Gemini AI
- Machine learning models trained with scikit-learn
- PDF generation using ReportLab

## Support

For support, please contact the development team or use the in-app AI support feature.

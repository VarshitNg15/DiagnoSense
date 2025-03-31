# DiagnoSense - Disease Prediction System

A web application that helps predict diseases based on patient symptoms using machine learning models. The system provides separate interfaces for patients and doctors, allowing for disease prediction, history tracking, and medical insights.

## Features

- Patient and Doctor authentication
- Disease prediction using NLP and ML models
- Patient disease history tracking
- Doctor's insights and notes
- User profiles linked to login credentials

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diagnosense
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python app.py
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Register as either a patient or doctor
4. Login with your credentials
5. Use the appropriate features based on your role:
   - Patients can:
     - Predict diseases based on symptoms
     - View their disease history
     - Access doctor's notes
   - Doctors can:
     - View patient records
     - Add medical insights and notes
     - Track patient history

## Project Structure

```
diagnosense/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── static/            # Static files (CSS, JS)
├── templates/         # HTML templates
├── models/           # ML models
└── healthcare.db     # SQLite database
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
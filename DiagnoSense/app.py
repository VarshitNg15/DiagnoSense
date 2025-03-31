from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
from prediction import DiseasePredictor
from flask_login import LoginManager, UserMixin, login_required, current_user, login_user, logout_user

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///healthcare.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize the disease predictor
predictor = DiseasePredictor()

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'patient' or 'doctor'
    profile = db.relationship('Profile', backref='user', uselist=False)
    patient = db.relationship('Patient', backref='user', uselist=False, foreign_keys='Patient.user_id')
    doctor_patients = db.relationship('Patient', backref='doctor', foreign_keys='Patient.doctor_id')

    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    specialization = db.Column(db.String(100))  # For doctors only
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class DiseaseHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    diagnosis_date = db.Column(db.DateTime, default=datetime.utcnow)
    doctor_notes = db.Column(db.Text)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    confidence = db.Column(db.Float)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    contact = db.Column(db.String(20))
    email = db.Column(db.String(120))
    last_visit = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='Active')
    diagnosis = db.Column(db.Text)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)

class Diagnosis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), nullable=False)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        specialization = request.form.get('specialization') if role == 'doctor' else None

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))

        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            role=role
        )
        db.session.add(user)
        db.session.flush()

        profile = Profile(
            user_id=user.id,
            name=name,
            email=email,
            phone=phone,
            specialization=specialization
        )
        db.session.add(profile)
        db.session.flush()

        # If registering as a patient, create a Patient record and assign to a doctor
        if role == 'patient':
            # Find a doctor with the least number of patients
            doctors = User.query.filter_by(role='doctor').all()
            if doctors:
                # Get the doctor with the least number of patients
                doctor = min(doctors, key=lambda d: len(d.doctor_patients))
                
                patient = Patient(
                    user_id=user.id,
                    doctor_id=doctor.id,
                    name=name,
                    email=email,
                    contact=phone,
                    status='Active'
                )
                db.session.add(patient)
                flash(f'Registration successful! You have been assigned to Dr. {doctor.profile.name}.')
            else:
                flash('Registration successful! A doctor will be assigned to you soon.')
        
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'doctor':
        # Get doctor's patients
        patients = Patient.query.filter_by(doctor_id=current_user.id).all()
        
        # Get today's appointments
        today = datetime.now().date()
        today_appointments = Appointment.query.filter(
            Appointment.doctor_id == current_user.id,
            Appointment.date == today
        ).count()
        
        # Get total patients
        total_patients = len(patients)
        
        # Get pending reports
        pending_reports = Diagnosis.query.filter(
            Diagnosis.doctor_id == current_user.id,
            Diagnosis.status == 'pending'
        ).count()
        
        return render_template('doctor_dashboard.html',
                             user=current_user,
                             patients=patients,
                             total_patients=total_patients,
                             today_appointments=today_appointments,
                             pending_reports=pending_reports)
    else:
        print("\n=== Loading Patient Dashboard ===")
        print(f"Current user: {current_user.id}, Role: {current_user.role}")
        
        # Get patient's recent history with doctor's notes
        recent_history = DiseaseHistory.query.filter_by(
            patient_id=current_user.id
        ).order_by(DiseaseHistory.diagnosis_date.desc()).limit(5).all()
        
        print(f"Found {len(recent_history)} history records")
        for record in recent_history:
            print(f"Record: {record.disease}")
            print(f"  - Date: {record.diagnosis_date}")
            print(f"  - Doctor ID: {record.doctor_id}")
            print(f"  - Doctor Notes: {record.doctor_notes}")
            print(f"  - Patient ID: {record.patient_id}")
            print("---")
        
        # Get patient's doctor information
        patient = Patient.query.filter_by(user_id=current_user.id).first()
        doctor = None
        if patient and patient.doctor_id:
            doctor = User.query.get(patient.doctor_id)
            print(f"Found doctor: {doctor.profile.name if doctor else 'None'}")
            print(f"Patient's doctor_id: {patient.doctor_id}")
            print(f"Patient's user_id: {patient.user_id}")
        
        print("=== End Patient Dashboard ===\n")
        return render_template('patient_dashboard.html',
                             user=current_user,
                             recent_history=recent_history,
                             doctor=doctor)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if current_user.role != 'patient':
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        
        # Get prediction from the model
        prediction = predictor.predict(symptoms)
        
        # Save to history
        history = DiseaseHistory(
            patient_id=current_user.id,
            disease=prediction['disease'],
            symptoms=symptoms,
            confidence=prediction['confidence']
        )
        db.session.add(history)
        
        # Update patient's current diagnosis
        patient = Patient.query.filter_by(user_id=current_user.id).first()
        if patient:
            patient.diagnosis = prediction['disease']
            patient.last_visit = datetime.utcnow()
        
        db.session.commit()
        
        return render_template('prediction_result.html', prediction=prediction)
    return render_template('predict.html')

@app.route('/history')
@login_required
def history():
    if current_user.role == 'doctor':
        # Get all patients assigned to this doctor
        patients = Patient.query.filter_by(doctor_id=current_user.id).all()
        patient_ids = [patient.user_id for patient in patients]
        
        # Get all diagnoses for these patients
        diagnoses = DiseaseHistory.query.filter(
            DiseaseHistory.patient_id.in_(patient_ids)
        ).order_by(DiseaseHistory.diagnosis_date.desc()).all()
        
        return render_template('history.html', diagnoses=diagnoses, is_doctor=True)
    else:
        # Get patient's own history
        diagnoses = DiseaseHistory.query.filter_by(
            patient_id=current_user.id
        ).order_by(DiseaseHistory.diagnosis_date.desc()).all()
        
        return render_template('history.html', diagnoses=diagnoses, is_doctor=False)

@app.route('/api/patient/<int:patient_id>')
@login_required
def get_patient_details(patient_id):
    print(f"Getting details for patient ID: {patient_id}")
    print(f"Current user role: {current_user.role}")
    
    if current_user.role != 'doctor':
        print("Unauthorized: Not a doctor")
        return jsonify({'error': 'Unauthorized'}), 403
    
    patient = Patient.query.get_or_404(patient_id)
    print(f"Found patient: {patient.name}")
    print(f"Patient's doctor_id: {patient.doctor_id}")
    print(f"Current user id: {current_user.id}")
    
    if patient.doctor_id != current_user.id:
        print("Unauthorized: Not the assigned doctor")
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Get the latest diagnosis using patient's user_id
    latest_diagnosis = DiseaseHistory.query.filter_by(patient_id=patient.user_id).order_by(DiseaseHistory.diagnosis_date.desc()).first()
    print(f"Latest diagnosis: {latest_diagnosis.disease if latest_diagnosis else 'None'}")
    
    response_data = {
        'id': patient.id,
        'name': patient.name,
        'age': patient.age,
        'contact': patient.contact,
        'email': patient.email,
        'last_visit': patient.last_visit.isoformat(),
        'status': patient.status,
        'diagnosis': patient.diagnosis,
        'latest_diagnosis': {
            'disease': latest_diagnosis.disease,
            'symptoms': latest_diagnosis.symptoms,
            'confidence': latest_diagnosis.confidence,
            'diagnosis_date': latest_diagnosis.diagnosis_date.isoformat(),
            'doctor_notes': latest_diagnosis.doctor_notes
        } if latest_diagnosis else None
    }
    print(f"Sending response: {response_data}")
    return jsonify(response_data)

@app.route('/api/diagnosis/feedback', methods=['POST'])
@login_required
def add_diagnosis_feedback():
    print("\n=== Starting add_diagnosis_feedback ===")
    print(f"Current user: {current_user.id}, Role: {current_user.role}")
    
    if current_user.role != 'doctor':
        print("Unauthorized: Not a doctor")
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Handle both form data and JSON requests
    if request.is_json:
        data = request.get_json()
        patient_id = data.get('patient_id')
        doctor_notes = data.get('doctor_notes')
        print(f"Received JSON data - patient_id: {patient_id}, notes: {doctor_notes}")
    else:
        patient_id = request.form.get('patient_id')
        doctor_notes = request.form.get('doctor_notes')
        print(f"Received form data - patient_id: {patient_id}, notes: {doctor_notes}")
    
    if not patient_id or not doctor_notes:
        print("Missing required fields")
        return jsonify({'error': 'Missing required fields'}), 400
    
    # First try to find the patient by user_id
    patient = Patient.query.filter_by(user_id=patient_id).first()
    if not patient:
        print(f"Patient not found with user_id: {patient_id}")
        return jsonify({'error': 'Patient not found'}), 404
    
    print(f"Found patient: {patient.name}, user_id: {patient.user_id}, doctor_id: {patient.doctor_id}")
    
    if patient.doctor_id != current_user.id:
        print("Unauthorized: Not the assigned doctor")
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Get the latest diagnosis using patient's user_id
    latest_diagnosis = DiseaseHistory.query.filter_by(
        patient_id=patient.user_id
    ).order_by(DiseaseHistory.diagnosis_date.desc()).first()
    
    if not latest_diagnosis:
        print("No diagnosis found for patient")
        return jsonify({'error': 'No diagnosis found'}), 404
    
    print(f"Found latest diagnosis: {latest_diagnosis.disease}")
    print(f"Current doctor_notes: {latest_diagnosis.doctor_notes}")
    
    # Update the latest diagnosis with doctor's notes
    latest_diagnosis.doctor_notes = doctor_notes
    latest_diagnosis.doctor_id = current_user.id
    
    # Update the patient's current diagnosis
    patient.diagnosis = latest_diagnosis.disease
    patient.last_visit = datetime.utcnow()
    
    try:
        db.session.commit()
        print("Successfully saved doctor's notes")
        print(f"Updated diagnosis: {latest_diagnosis.disease}")
        print(f"Updated doctor's notes: {latest_diagnosis.doctor_notes}")
        print("=== End add_diagnosis_feedback ===\n")
        return jsonify({'success': True, 'message': 'Feedback submitted successfully'})
    except Exception as e:
        print(f"Error saving doctor's notes: {str(e)}")
        db.session.rollback()
        print("=== End add_diagnosis_feedback ===\n")
        return jsonify({'error': 'Failed to save feedback'}), 500

@app.route('/doctor/insights/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def doctor_insights(patient_id):
    if current_user.role != 'doctor':
        return redirect(url_for('dashboard'))
    
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        history_id = request.form.get('history_id')
        notes = request.form.get('notes')
        
        history = DiseaseHistory.query.get(history_id)
        if history and history.patient_id == patient.user_id:
            history.doctor_notes = notes
            db.session.commit()
            flash('Notes updated successfully')
    
    # Get patient's history
    history = DiseaseHistory.query.filter_by(patient_id=patient.user_id).order_by(DiseaseHistory.diagnosis_date.desc()).all()
    
    return render_template('doctor_insights.html', patient=patient, history=history)

@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out successfully.')
    return redirect(url_for('index'))

@app.route('/doctor/patient/<int:patient_id>')
@login_required
def view_patient(patient_id):
    if current_user.role != 'doctor':
        return redirect(url_for('dashboard'))
    
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        return redirect(url_for('dashboard'))
    
    # Get the latest diagnosis
    latest_diagnosis = DiseaseHistory.query.filter_by(
        patient_id=patient.user_id
    ).order_by(DiseaseHistory.diagnosis_date.desc()).first()
    
    return render_template('view_patient.html',
                         patient=patient,
                         latest_diagnosis=latest_diagnosis)

@app.route('/doctor-notes')
@login_required
def doctor_notes():
    if current_user.role != 'patient':
        return redirect(url_for('dashboard'))
    
    # Get patient's recent history with doctor's notes
    recent_history = DiseaseHistory.query.filter_by(
        patient_id=current_user.id
    ).order_by(DiseaseHistory.diagnosis_date.desc()).all()
    
    # Get patient's doctor information
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    doctor = None
    if patient and patient.doctor_id:
        doctor = User.query.get(patient.doctor_id)
    
    return render_template('doctor_notes.html',
                         user=current_user,
                         recent_history=recent_history,
                         doctor=doctor)

if __name__ == '__main__':
    with app.app_context():
        # Create tables if they don't exist
        db.create_all()
    app.run(debug=True) 
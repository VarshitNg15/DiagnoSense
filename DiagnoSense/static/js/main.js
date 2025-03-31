// Main JavaScript file for DiagnoSense

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    // Handle symptom button clicks
    const symptomBtns = document.querySelectorAll('.symptom-btn');
    const symptomsTextarea = document.getElementById('symptoms');
    
    if (symptomsTextarea && symptomBtns.length > 0) {
        symptomBtns.forEach(btn => {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                const symptom = this.textContent;
                const currentText = symptomsTextarea.value;
                
                if (currentText) {
                    symptomsTextarea.value = currentText + ', ' + symptom;
                } else {
                    symptomsTextarea.value = symptom;
                }
            });
        });
    }

    // Handle form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!form.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}); 
/**
 * Funciones para la interfaz web del Clasificador de Gomitas
 */

// Inicialización cuando el DOM esté cargado
document.addEventListener('DOMContentLoaded', function() {
    // Activar todos los tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Animate progress bars (for results page)
    const progressBars = document.querySelectorAll('.progress-bar');
    setTimeout(() => {
        progressBars.forEach(bar => {
            const width = bar.getAttribute('aria-valuenow') + '%';
            bar.style.width = width;
        });
    }, 100);
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Show current year in footer
    const yearElement = document.querySelector('.current-year');
    if (yearElement) {
        yearElement.textContent = new Date().getFullYear();
    }
});

// Función para validar formulario de subida de imagen
// Reemplaza la función validateImageForm que está causando la alerta en el archivo main.js
// o añade este script a la sección extra_js de predict.html

// Función para validar formulario de subida de imagen
function validateImageForm() {
    const fileInput = document.getElementById('file');
    if (!fileInput) return true;
    
    if (fileInput.files.length === 0) {
        alert('Por favor, selecciona una imagen');
        return false;
    }
    
    const file = fileInput.files[0];
    
    // Verificar tipo de archivo (incluir HEIC/HEIF)
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/heic', 'image/heif'];
    const validExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.heif'];
    
    // Verificar por tipo MIME o por extensión
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
        // En caso de HEIC, verificar específicamente por la extensión
        if (file.name.toLowerCase().endsWith('.heic') || file.name.toLowerCase().endsWith('.heif')) {
            // Es un archivo HEIC/HEIF por extensión, permitirlo
            return true;
        }
        
        alert('El archivo debe ser una imagen (JPEG, PNG, GIF o HEIC)');
        return false;
    }
    
    // Verificar tamaño (max 20MB)
    if (file.size > 20 * 1024 * 1024) {
        alert('La imagen es demasiado grande. Máximo 20MB.');
        return false;
    }
    
    return true;
}

// Sustituir la función anterior por esta en el evento submit del formulario
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            if (!validateImageForm()) {
                e.preventDefault();
            }
        });
    }
});
// Asignar validación al formulario si existe
const uploadForm = document.getElementById('upload-form');
if (uploadForm) {
    uploadForm.addEventListener('submit', function(e) {
        if (!validateImageForm()) {
            e.preventDefault();
        }
    });
}
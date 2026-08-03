// ============================================================
// Brain Tumor MRI Classifier - أكواد JavaScript مخصصة
// ============================================================

// ============================================================
// 1. انتظار تحميل الصفحة بالكامل
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ Brain Tumor MRI Classifier - JavaScript Loaded');
    
    // تهيئة جميع الوظائف
    initializeTooltips();
    initializeAnimations();
    initializeImagePreview();
});

// ============================================================
// 2. تهيئة الـ Tooltips (تلميحات الأدوات)
// ============================================================
function initializeTooltips() {
    // إضافة تلميحات لجميع العناصر التي تحمل class 'has-tooltip'
    const tooltipElements = document.querySelectorAll('.has-tooltip');
    
    tooltipElements.forEach(element => {
        const tooltipText = element.getAttribute('data-tooltip');
        if (tooltipText) {
            element.addEventListener('mouseenter', function(e) {
                showTooltip(e, tooltipText);
            });
            element.addEventListener('mouseleave', function(e) {
                hideTooltip(e);
            });
        }
    });
}

// ============================================================
// 3. عرض وإخفاء الـ Tooltip
// ============================================================
function showTooltip(event, text) {
    const tooltip = document.createElement('div');
    tooltip.className = 'custom-tooltip';
    tooltip.textContent = text;
    tooltip.style.position = 'absolute';
    tooltip.style.backgroundColor = '#333333';
    tooltip.style.color = '#FFFFFF';
    tooltip.style.padding = '8px 12px';
    tooltip.style.borderRadius = '6px';
    tooltip.style.fontSize = '12px';
    tooltip.style.zIndex = '1000';
    tooltip.style.maxWidth = '200px';
    tooltip.style.boxShadow = '0 4px 8px rgba(0,0,0,0.3)';
    tooltip.style.pointerEvents = 'none';
    tooltip.style.transition = 'opacity 0.3s ease';
    
    document.body.appendChild(tooltip);
    
    const rect = event.target.getBoundingClientRect();
    tooltip.style.top = (rect.top - tooltip.offsetHeight - 10 + window.scrollY) + 'px';
    tooltip.style.left = (rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + window.scrollX) + 'px';
    
    setTimeout(() => {
        tooltip.style.opacity = '0.9';
    }, 10);
}

function hideTooltip(event) {
    const tooltips = document.querySelectorAll('.custom-tooltip');
    tooltips.forEach(tooltip => tooltip.remove());
}

// ============================================================
// 4. تهيئة الرسوم المتحركة
// ============================================================
function initializeAnimations() {
    // إضافة تأثير ظهور تدريجي للعناصر التي تحمل class 'fade-in'
    const fadeElements = document.querySelectorAll('.fade-in');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.1
    });
    
    fadeElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        element.style.transform = 'translateY(20px)';
        observer.observe(element);
    });
}

// ============================================================
// 5. معاينة الصورة قبل التحميل
// ============================================================
function initializeImagePreview() {
    const fileInput = document.querySelector('input[type="file"]');
    
    if (fileInput) {
        fileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const previewContainer = document.getElementById('image-preview');
                    if (previewContainer) {
                        previewContainer.innerHTML = `
                            <img src="${e.target.result}" 
                                 alt="Preview" 
                                 style="max-width: 100%; max-height: 300px; border-radius: 12px; border: 2px solid #4A90D9; margin-top: 1rem;">
                        `;
                    }
                };
                reader.readAsDataURL(file);
            }
        });
    }
}

// ============================================================
// 6. تحسين عرض الجداول
// ============================================================
function initializeTableSearch() {
    const searchInput = document.querySelector('#table-search');
    if (searchInput) {
        searchInput.addEventListener('keyup', function() {
            const filter = this.value.toLowerCase();
            const tableRows = document.querySelectorAll('.custom-table tbody tr');
            
            tableRows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(filter) ? '' : 'none';
            });
        });
    }
}

// ============================================================
// 7. تحميل الصور بتأخير (Lazy Loading)
// ============================================================
function initializeLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.getAttribute('data-src');
                img.removeAttribute('data-src');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
}

// ============================================================
// 8. تنسيق الأرقام (عرضها بطريقة مفهومة)
// ============================================================
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// ============================================================
// 9. عرض رسائل التنبيه (مخصصة)
// ============================================================
function showAlert(message, type = 'info') {
    const colors = {
        'info': '#4A90D9',
        'success': '#4CAF50',
        'warning': '#FFD93D',
        'error': '#FF6B6B'
    };
    
    const alertBox = document.createElement('div');
    alertBox.className = 'custom-alert';
    alertBox.style.position = 'fixed';
    alertBox.style.bottom = '20px';
    alertBox.style.right = '20px';
    alertBox.style.backgroundColor = colors[type] || '#333333';
    alertBox.style.color = '#FFFFFF';
    alertBox.style.padding = '12px 20px';
    alertBox.style.borderRadius = '8px';
    alertBox.style.boxShadow = '0 4px 12px rgba(0,0,0,0.4)';
    alertBox.style.zIndex = '9999';
    alertBox.style.fontSize = '14px';
    alertBox.style.maxWidth = '300px';
    alertBox.textContent = message;
    
    document.body.appendChild(alertBox);
    
    setTimeout(() => {
        alertBox.style.opacity = '0';
        alertBox.style.transition = 'opacity 0.5s ease';
        setTimeout(() => alertBox.remove(), 500);
    }, 3000);
}

// ============================================================
// 10. تهيئة جميع الوظائف الإضافية
// ============================================================
initializeTableSearch();
initializeLazyLoading();

// ============================================================
// 11. تصدير الدوال للاستخدام في مكان آخر (اختياري)
// ============================================================
window.brainTumorApp = {
    showAlert: showAlert,
    formatNumber: formatNumber,
    showTooltip: showTooltip,
    hideTooltip: hideTooltip
};

console.log('✅ جميع الوظائف تم تهيئتها بنجاح');

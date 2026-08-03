# ============================================================
# app/components/sidebar.py
# مكون الشريط الجانبي - إدارة عناصر الشريط الجانبي في Streamlit
# ============================================================

import streamlit as st
from typing import Optional, Dict, List, Any
from PIL import Image
import os

# ============================================================
# 1. عرض الشعار في الشريط الجانبي
# ============================================================

def display_logo(logo_path: Optional[str] = None, width: int = 200) -> None:
    """
    عرض الشعار في أعلى الشريط الجانبي.
    
    Args:
        logo_path: مسار ملف الشعار (اختياري)
        width: عرض الصورة بالبكسل
    """
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path)
            st.sidebar.image(logo, width=width)
        except Exception as e:
            st.sidebar.error(f"❌ تعذر تحميل الشعار: {e}")
    else:
        # عرض شعار نصي بديل
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">🧠</div>
            <div style="color: #4A90D9; font-size: 1.2rem; font-weight: 700;">
                Brain Tumor MRI
            </div>
            <div style="color: #AAAAAA; font-size: 0.8rem;">
                Classification System
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 2. عرض عنوان المشروع
# ============================================================

def display_title(title: str = "🧠 Brain Tumor MRI Classifier", 
                   subtitle: Optional[str] = None) -> None:
    """
    عرض عنوان المشروع في الشريط الجانبي.
    
    Args:
        title: العنوان الرئيسي
        subtitle: العنوان الفرعي (اختياري)
    """
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 0.5rem 0;">
        <div style="color: #FFFFFF; font-size: 1.3rem; font-weight: 700;">
            {title}
        </div>
        {f'<div style="color: #AAAAAA; font-size: 0.8rem; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 3. عرض قائمة التنقل (Navigation Menu)
# ============================================================

def display_navigation_menu() -> str:
    """
    عرض قائمة التنقل في الشريط الجانبي.
    
    Returns:
        الصفحة المختارة (str)
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 التنقل")
    
    # تعريف الصفحات
    pages = {
        "🏠 الرئيسية": "Home",
        "📊 تحليل الصورة": "Analyze",
        "📈 التدريب": "Training",
        "📋 التقارير": "Reports",
        "📖 التوثيق": "Documentation",
        "⚙️ الإعدادات": "Settings"
    }
    
    # عرض الأزرار كـ Radio buttons
    selected = st.sidebar.radio(
        "اختر الصفحة",
        options=list(pages.keys()),
        index=0,
        label_visibility="collapsed"
    )
    
    return pages[selected]

# ============================================================
# 4. عرض معلومات النموذج
# ============================================================

def display_model_info(model_info: Dict[str, Any]) -> None:
    """
    عرض معلومات النموذج في الشريط الجانبي.
    
    Args:
        model_info: قاموس بمعلومات النموذج
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 معلومات النموذج")
    
    # عرض معلومات النموذج
    info_items = [
        ("الاسم", model_info.get('name', 'غير محدد')),
        ("الإصدار", model_info.get('version', '1.0.0')),
        ("الفئات", model_info.get('num_classes', 'غير محدد')),
        ("الدقة", f"{model_info.get('accuracy', 0):.2%}" if model_info.get('accuracy') else 'غير محدد'),
        ("المعلمات", f"{model_info.get('num_params', 0):,}" if model_info.get('num_params') else 'غير محدد')
    ]
    
    for label, value in info_items:
        st.sidebar.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
            <span style="color: #AAAAAA; font-size: 0.8rem;">{label}:</span>
            <span style="color: #FFFFFF; font-size: 0.8rem; font-weight: 500;">{value}</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 5. عرض إعدادات المستخدم
# ============================================================

def display_user_settings() -> Dict[str, Any]:
    """
    عرض إعدادات المستخدم في الشريط الجانبي.
    
    Returns:
        قاموس بإعدادات المستخدم
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ الإعدادات")
    
    # إعدادات العرض
    show_gradcam = st.sidebar.checkbox("🔥 عرض Grad-CAM", value=True)
    show_probabilities = st.sidebar.checkbox("📊 عرض الاحتمالات", value=True)
    
    # إعدادات الثقة
    confidence_threshold = st.sidebar.slider(
        "🎯 حد الثقة",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="الحد الأدنى لنسبة الثقة لقبول التنبؤ"
    )
    
    # إعدادات المظهر
    st.sidebar.markdown("#### 🎨 المظهر")
    theme = st.sidebar.selectbox(
        "الوضع",
        options=["داكن", "فاتح"],
        index=0
    )
    
    return {
        'show_gradcam': show_gradcam,
        'show_probabilities': show_probabilities,
        'confidence_threshold': confidence_threshold,
        'theme': theme
    }

# ============================================================
# 6. عرض رفع الملف (File Uploader)
# ============================================================

def display_file_uploader() -> Optional[Any]:
    """
    عرض مكون رفع الملف في الشريط الجانبي.
    
    Returns:
        الملف المرفوع (UploadedFile) أو None
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 رفع الصورة")
    
    # تعريف الأنواع المسموحة
    allowed_types = ['jpg', 'jpeg', 'png']
    
    # مكون رفع الملف
    uploaded_file = st.sidebar.file_uploader(
        "اختر صورة MRI",
        type=allowed_types,
        help="الصور المسموحة: JPG, JPEG, PNG"
    )
    
    return uploaded_file

# ============================================================
# 7. عرض معلومات الاتصال
# ============================================================

def display_contact_info() -> None:
    """
    عرض معلومات الاتصال في أسفل الشريط الجانبي.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <div style="color: #666666; font-size: 0.7rem;">
            📧 support@braintumor.ai
        </div>
        <div style="color: #666666; font-size: 0.7rem;">
            📱 +20 100 000 0000
        </div>
        <div style="color: #444444; font-size: 0.6rem; margin-top: 0.5rem;">
            v1.0.0 | © 2024 Brain Tumor AI
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 8. عرض إحصائيات سريعة
# ============================================================

def display_quick_stats(stats: Dict[str, Any]) -> None:
    """
    عرض إحصائيات سريعة في الشريط الجانبي.
    
    Args:
        stats: قاموس بالإحصائيات
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 إحصائيات سريعة")
    
    # عرض الإحصائيات
    stat_items = [
        ("📊 إجمالي التنبؤات", stats.get('total_predictions', 0)),
        ("✅ صحيحة", stats.get('correct_predictions', 0)),
        ("❌ خاطئة", stats.get('wrong_predictions', 0)),
        ("🎯 الدقة", f"{stats.get('accuracy', 0):.1%}" if stats.get('accuracy') else '0%')
    ]
    
    for label, value in stat_items:
        st.sidebar.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
            <span style="color: #AAAAAA; font-size: 0.8rem;">{label}</span>
            <span style="color: #FFFFFF; font-size: 0.8rem; font-weight: 500;">{value}</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 9. الشريط الجانبي الكامل (All-in-One)
# ============================================================

def render_sidebar(logo_path: Optional[str] = None,
                    model_info: Optional[Dict[str, Any]] = None,
                    stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    عرض الشريط الجانبي الكامل بجميع مكوناته.
    
    Args:
        logo_path: مسار ملف الشعار (اختياري)
        model_info: معلومات النموذج (اختياري)
        stats: إحصائيات سريعة (اختياري)
    
    Returns:
        قاموس بجميع مدخلات المستخدم والإعدادات
    """
    # 1. الشعار
    display_logo(logo_path)
    
    # 2. العنوان
    display_title(
        title="🧠 Brain Tumor MRI",
        subtitle="Classification System"
    )
    
    # 3. قائمة التنقل
    current_page = display_navigation_menu()
    
    # 4. رفع الملف
    uploaded_file = display_file_uploader()
    
    # 5. إعدادات المستخدم
    user_settings = display_user_settings()
    
    # 6. معلومات النموذج (إن وجدت)
    if model_info:
        display_model_info(model_info)
    
    # 7. إحصائيات سريعة (إن وجدت)
    if stats:
        display_quick_stats(stats)
    
    # 8. معلومات الاتصال
    display_contact_info()
    
    # إرجاع جميع البيانات
    return {
        'current_page': current_page,
        'uploaded_file': uploaded_file,
        'settings': user_settings
    }

# ============================================================
# 10. عرض الشريط الجانبي مع حالة التطبيق
# ============================================================

def render_sidebar_with_status(logo_path: Optional[str] = None,
                                model_info: Optional[Dict[str, Any]] = None,
                                stats: Optional[Dict[str, Any]] = None,
                                status_message: Optional[str] = None) -> Dict[str, Any]:
    """
    عرض الشريط الجانبي مع حالة التطبيق.
    
    Args:
        logo_path: مسار ملف الشعار (اختياري)
        model_info: معلومات النموذج (اختياري)
        stats: إحصائيات سريعة (اختياري)
        status_message: رسالة الحالة (اختياري)
    
    Returns:
        قاموس بجميع مدخلات المستخدم والإعدادات
    """
    # عرض الشريط الجانبي الأساسي
    result = render_sidebar(logo_path, model_info, stats)
    
    # عرض رسالة الحالة (إن وجدت)
    if status_message:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"""
        <div style="
            background-color: #1E3A1E;
            color: #4CAF50;
            padding: 0.75rem;
            border-radius: 8px;
            font-size: 0.8rem;
            text-align: center;
        ">
            ✅ {status_message}
        </div>
        """, unsafe_allow_html=True)
    
    return result

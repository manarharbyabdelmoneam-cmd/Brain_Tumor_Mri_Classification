# ============================================================
# app/app.py
# الملف الرئيسي لتطبيق Streamlit - Brain Tumor MRI Classifier
# ============================================================

# ============================================================
# 1. استيراد Streamlit أولاً (مهم جداً)
# ============================================================
import streamlit as st

# ============================================================
# 2. إعدادات الصفحة (يجب أن تكون أول شيء بعد استيراد streamlit)
# ============================================================
st.set_page_config(
    page_title="🧠 Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 3. باقي الاستيرادات (بعد set_page_config)
# ============================================================
import os
import sys
from pathlib import Path

# إضافة مسار المشروع إلى sys.path
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================
# 4. استيراد المكونات من المشروع
# ============================================================
from app.components.sidebar import render_sidebar
from app.components.uploader import render_uploader
from app.components.results import display_full_results, display_error_message
from app.components.metrics import display_metrics_dashboard
from app.components.charts import display_all_charts

from app.utils.model_loader import load_model, display_model_summary
from app.utils.predictor import predict_from_uploaded, display_prediction_results
from app.utils.image_processor import display_image_info
from app.utils.formatter import format_class_name, format_confidence

# ============================================================
# 5. إعدادات المشروع
# ============================================================

# مسارات النماذج
MODEL_PATH = "models_saved/keras_model.h5"
METADATA_PATH = "models_saved/metadata/model_config.json"
CLASS_NAMES_PATH = "models_saved/metadata/class_names.json"

# أسماء الفئات الافتراضية
DEFAULT_CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']

# حجم الصورة المستهدف
TARGET_SIZE = (224, 224)

# ============================================================
# 6. تحميل أسماء الفئات
# ============================================================

def load_class_names() -> list:
    """
    تحميل أسماء الفئات من ملف أو استخدام الافتراضية.
    
    Returns:
        قائمة بأسماء الفئات
    """
    try:
        if os.path.exists(CLASS_NAMES_PATH):
            import json
            with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"⚠️ تعذر تحميل أسماء الفئات: {e}")
    
    return DEFAULT_CLASS_NAMES

# ============================================================
# 7. تحميل النموذج
# ============================================================

@st.cache_resource
def get_model():
    """
    تحميل النموذج مع التخزين المؤقت.
    
    Returns:
        النموذج المحمّل أو None
    """
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        st.error(f"❌ ملف النموذج غير موجود: {MODEL_PATH}")
        st.info("💡 يرجى وضع ملف النموذج في المسار: models_saved/keras_model.h5")
        return None

# ============================================================
# 8. تهيئة حالة الجلسة
# ============================================================

def init_session_state():
    """
    تهيئة متغيرات حالة الجلسة.
    """
    if 'model' not in st.session_state:
        st.session_state.model = get_model()
    
    if 'class_names' not in st.session_state:
        st.session_state.class_names = load_class_names()
    
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None

# ============================================================
# 9. الصفحة الرئيسية (Home)
# ============================================================

def home_page():
    """
    عرض الصفحة الرئيسية للتطبيق.
    """
    st.markdown("""
    # 🧠 Brain Tumor MRI Classifier
    
    ### نظام ذكاء اصطناعي لتصنيف أورام المخ من صور الرنين المغناطيسي
    
    > ⚠️ **تنبيه:** هذا النظام هو **أداة مساعدة** للأطباء وليس بديلاً عن التشخيص الطبي البشري.
    > يجب دائمًا استشارة طبيب مختص قبل اتخاذ أي قرار علاجي.
    """)
    
    st.markdown("---")
    
    # عرض معلومات النموذج
    if st.session_state.model is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🧠 النموذج", "MobileNetV2")
        
        with col2:
            st.metric("📂 الفئات", len(st.session_state.class_names))
        
        with col3:
            st.metric("📐 حجم الإدخال", "224×224")
        
        with col4:
            try:
                params = st.session_state.model.count_params()
                st.metric("🔢 المعلمات", f"{params:,}")
            except:
                st.metric("🔢 المعلمات", "غير محدد")
    
    st.markdown("---")
    
    # قسم رفع الصورة
    st.markdown("## 📤 تحميل صورة للتحليل")
    
    # استخدام مكون رفع الصور
    image, file_path, metadata, controls = render_uploader(show_preview=True)
    
    if image is not None:
        st.session_state.uploaded_image = image
        st.session_state.uploaded_file_path = file_path
        
        # عرض معلومات الصورة
        with st.expander("📋 معلومات الصورة", expanded=False):
            display_image_info(image)
        
        # زر التحليل
        if st.button("🔍 تحليل الصورة", type="primary", use_container_width=True):
            with st.spinner("⏳ جاري تحليل الصورة..."):
                # استخدام النموذج للتنبؤ
                model = st.session_state.model
                class_names = st.session_state.class_names
                
                if model is not None:
                    from app.utils.predictor import predict_and_format
                    result = predict_and_format(
                        model=model,
                        image=image,
                        class_names=class_names,
                        target_size=TARGET_SIZE,
                        use_gradcam=True
                    )
                    
                    if result is not None:
                        st.session_state.prediction_result = result
                        st.rerun()
                else:
                    display_error_message("❌ النموذج غير محمّل. يرجى التحقق من وجود الملف.", "error")
    
    # عرض النتائج (إذا كانت موجودة)
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        st.markdown("## 📊 نتائج التحليل")
        
        result = st.session_state.prediction_result
        display_prediction_results(result)
        
        # زر مسح النتائج
        if st.button("🗑️ مسح النتائج وبدء تحليل جديد"):
            st.session_state.prediction_result = None
            st.session_state.uploaded_image = None
            st.rerun()
    
    # نصائح سريعة
    st.markdown("---")
    with st.expander("💡 نصائح سريعة", expanded=False):
        st.markdown("""
        - **جودة الصورة:** استخدم صور MRI عالية الجودة للحصول على نتائج أفضل.
        - **حجم الصورة:** الصورة ستُغير حجمها تلقائيًا إلى 224×224 بكسل.
        - **الفئات المدعومة:** Glioma, Meningioma, Pituitary, No Tumor.
        - **التفسير:** استخدم Grad-CAM لفهم المنطقة التي ركز عليها النموذج.
        - **الثقة:** النتائج التي تزيد عن 80% تعتبر موثوقة جدًا.
        """)

# ============================================================
# 10. صفحة تحليل الصورة
# ============================================================

def analyze_page():
    """
    عرض صفحة تحليل الصورة.
    """
    st.markdown("# 📊 تحليل الصورة")
    st.markdown("---")
    
    # استخدام مكون رفع الصور
    image, file_path, metadata, controls = render_uploader(show_preview=True)
    
    if image is not None:
        st.session_state.uploaded_image = image
        st.session_state.uploaded_file_path = file_path
        
        # عرض معلومات الصورة
        display_image_info(image)
        
        # زر التحليل
        if st.button("🔍 تحليل الصورة", type="primary", use_container_width=True):
            with st.spinner("⏳ جاري تحليل الصورة..."):
                model = st.session_state.model
                class_names = st.session_state.class_names
                
                if model is not None:
                    from app.utils.predictor import predict_and_format
                    result = predict_and_format(
                        model=model,
                        image=image,
                        class_names=class_names,
                        target_size=TARGET_SIZE,
                        use_gradcam=True
                    )
                    
                    if result is not None:
                        st.session_state.prediction_result = result
                        st.rerun()
                else:
                    display_error_message("❌ النموذج غير محمّل.", "error")
    
    # عرض النتائج
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        st.markdown("## 📊 النتائج")
        
        result = st.session_state.prediction_result
        display_prediction_results(result)
        
        if st.button("🗑️ مسح النتائج"):
            st.session_state.prediction_result = None
            st.rerun()
    else:
        st.info("💡 قم برفع صورة وتحليلها لعرض النتائج هنا.")

# ============================================================
# 11. دالة عرض الصفحة حسب التنقل
# ============================================================

def render_page(page: str):
    """
    عرض الصفحة المحددة.
    
    Args:
        page: اسم الصفحة
    """
    if page == 'home':
        home_page()
    elif page == 'analyze':
        analyze_page()
    else:
        home_page()

# ============================================================
# 12. تشغيل التطبيق
# ============================================================

def main():
    """
    تشغيل التطبيق الرئيسي.
    """
    # 1. تهيئة حالة الجلسة
    init_session_state()
    
    # 2. عرض الشريط الجانبي
    with st.sidebar:
        st.markdown("---")
        
        # عرض معلومات النموذج في الشريط الجانبي
        if st.session_state.model is not None:
            st.markdown("### 🤖 معلومات النموذج")
            st.caption(f"**الاسم:** MobileNetV2")
            st.caption(f"**الفئات:** {len(st.session_state.class_names)}")
            try:
                st.caption(f"**المعلمات:** {st.session_state.model.count_params():,}")
            except:
                pass
        
        st.markdown("---")
        
        # أزرار التنقل
        st.markdown("### 📌 التنقل")
        
        if st.button("🏠 الرئيسية", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        
        if st.button("📊 تحليل الصورة", use_container_width=True):
            st.session_state.page = 'analyze'
            st.rerun()
        
        # التحقق من وجود ملفات الصفحات قبل عرض الأزرار
        pages_dir = Path(__file__).parent / "pages"
        
        if (pages_dir / "2_Training_Analysis.py").exists():
            if st.button("📈 تحليل التدريب", use_container_width=True):
                st.switch_page("pages/2_Training_Analysis.py")
        
        if (pages_dir / "1_Model_Comparison.py").exists():
            if st.button("📊 مقارنة النماذج", use_container_width=True):
                st.switch_page("pages/1_Model_Comparison.py")
        
        if (pages_dir / "3_Documentation.py").exists():
            if st.button("📖 التوثيق", use_container_width=True):
                st.switch_page("pages/3_Documentation.py")
        
        st.markdown("---")
        
        # معلومات الاتصال
        st.markdown("""
        <div style="text-align: center; color: #666666; font-size: 0.7rem;">
            <p>🧠 Brain Tumor MRI Classifier</p>
            <p>v1.0.0</p>
            <p>📧 support@braintumor.ai</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 3. عرض الصفحة المحددة
    render_page(st.session_state.page)

# ============================================================
# 13. تشغيل التطبيق
# ============================================================

if __name__ == "__main__":
    main()

# ============================================================
# app/utils/model_loader.py
# تحميل النماذج - إدارة تحميل وتخزين نماذج TensorFlow/Keras
# ============================================================

import tensorflow as tf
import os
import json
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

# ============================================================
# 1. تحميل النموذج من ملف
# ============================================================

@st.cache_resource
def load_model(model_path: str) -> Optional[tf.keras.Model]:
    """
    تحميل نموذج Keras من ملف.
    
    Args:
        model_path: مسار ملف النموذج (.h5 أو .keras)
        
    Returns:
        النموذج المحمّل أو None في حالة الخطأ
    """
    try:
        # التحقق من وجود الملف
        if not os.path.exists(model_path):
            st.error(f"❌ الملف غير موجود: {model_path}")
            return None
        
        # تحميل النموذج
        model = tf.keras.models.load_model(model_path)
        
        # عرض معلومات النموذج
        st.success(f"✅ تم تحميل النموذج من: {model_path}")
        st.caption(f"📊 عدد المعلمات: {model.count_params():,}")
        
        return model
    
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return None

# ============================================================
# 2. تحميل النموذج مع معلومات إضافية
# ============================================================

def load_model_with_metadata(model_path: str,
                              metadata_path: Optional[str] = None) -> Tuple[Optional[tf.keras.Model],
                                                                             Optional[Dict[str, Any]]]:
    """
    تحميل النموذج مع معلوماته الوصفية.
    
    Args:
        model_path: مسار ملف النموذج
        metadata_path: مسار ملف المعلومات الوصفية (JSON)
        
    Returns:
        (النموذج, المعلومات الوصفية)
    """
    # 1. تحميل النموذج
    model = load_model(model_path)
    
    if model is None:
        return None, None
    
    # 2. تحميل المعلومات الوصفية
    metadata = None
    
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            st.warning(f"⚠️ تعذر تحميل المعلومات الوصفية: {e}")
    
    # 3. إذا لم يتم توفير معلومات وصفية، نحاول استخراجها من النموذج
    if metadata is None:
        metadata = extract_model_metadata(model)
    
    return model, metadata

# ============================================================
# 3. استخراج معلومات النموذج
# ============================================================

def extract_model_metadata(model: tf.keras.Model) -> Dict[str, Any]:
    """
    استخراج معلومات وصفية من النموذج.
    
    Args:
        model: نموذج Keras
        
    Returns:
        قاموس بالمعلومات الوصفية
    """
    metadata = {}
    
    # 1. هيكل النموذج
    metadata['input_shape'] = [layer.input_shape for layer in model.layers if hasattr(layer, 'input_shape')]
    metadata['output_shape'] = model.output_shape
    
    # 2. عدد المعلمات
    metadata['total_params'] = model.count_params()
    
    # 3. عدد الطبقات
    metadata['num_layers'] = len(model.layers)
    
    # 4. أسماء الطبقات
    metadata['layer_names'] = [layer.name for layer in model.layers]
    
    # 5. نوع النموذج
    metadata['model_type'] = type(model).__name__
    
    # 6. عدد الفئات (إذا كان التصنيف)
    try:
        if hasattr(model, 'output_shape') and len(model.output_shape) > 1:
            metadata['num_classes'] = model.output_shape[-1]
    except:
        pass
    
    return metadata

# ============================================================
# 4. تخزين النموذج
# ============================================================

def save_model(model: tf.keras.Model,
                model_path: str,
                metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    تخزين النموذج مع معلومات وصفية.
    
    Args:
        model: نموذج Keras
        model_path: مسار حفظ النموذج
        metadata: معلومات وصفية إضافية
        
    Returns:
        True إذا نجح الحفظ، False إذا فشل
    """
    try:
        # 1. إنشاء المجلدات إذا لزم الأمر
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 2. حفظ النموذج
        model.save(model_path)
        
        # 3. حفظ المعلومات الوصفية
        if metadata:
            metadata_path = model_path.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        return True
    
    except Exception as e:
        print(f"❌ خطأ في حفظ النموذج: {e}")
        return False

# ============================================================
# 5. تحميل نموذج من ملف مرفوع
# ============================================================

def load_model_from_uploaded(uploaded_file) -> Optional[tf.keras.Model]:
    """
    تحميل نموذج من ملف مرفوع في Streamlit.
    
    Args:
        uploaded_file: الملف المرفوع
        
    Returns:
        النموذج المحمّل أو None
    """
    try:
        # حفظ الملف مؤقتًا
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # تحميل النموذج
        model = tf.keras.models.load_model(tmp_path)
        
        # حذف الملف المؤقت
        try:
            os.remove(tmp_path)
        except:
            pass
        
        return model
    
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return None

# ============================================================
# 6. عرض ملخص النموذج
# ============================================================

def display_model_summary(model: tf.keras.Model) -> None:
    """
    عرض ملخص النموذج في واجهة Streamlit.
    
    Args:
        model: نموذج Keras
    """
    if model is None:
        st.warning("⚠️ لا يوجد نموذج لعرض ملخصه.")
        return
    
    # 1. المعلومات الأساسية
    metadata = extract_model_metadata(model)
    
    st.markdown("### 📊 معلومات النموذج")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📐 المدخلات", str(metadata.get('input_shape', 'غير محدد')))
    
    with col2:
        st.metric("📤 المخرجات", str(metadata.get('output_shape', 'غير محدد')))
    
    with col3:
        st.metric("🔢 المعلمات", f"{metadata.get('total_params', 0):,}")
    
    # 2. هيكل النموذج
    st.markdown("### 🏗️ هيكل النموذج")
    
    # عرض الطبقات في جدول
    layer_data = []
    for i, layer in enumerate(model.layers):
        layer_data.append({
            '#': i + 1,
            'الطبقة': layer.name,
            'النوع': layer.__class__.__name__,
            'المخرجات': str(layer.output_shape) if hasattr(layer, 'output_shape') else '-',
            'المعلمات': f"{layer.count_params():,}" if hasattr(layer, 'count_params') else '-'
        })
    
    st.dataframe(layer_data, use_container_width=True)
    
    # 3. عرض الصورة الكاملة (اختياري)
    if st.button("📋 عرض الملخص الكامل"):
        # استخدام TensorFlow لطباعة الملخص
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        
        st.code('\n'.join(summary_lines), language='text')

# ============================================================
# 7. اختبار النموذج بصورة عشوائية
# ============================================================

def test_model_with_random_image(model: tf.keras.Model,
                                  input_shape: Tuple[int, ...] = (224, 224, 3)) -> Dict[str, Any]:
    """
    اختبار النموذج بصورة عشوائية.
    
    Args:
        model: نموذج Keras
        input_shape: شكل الصورة المدخلة
        
    Returns:
        قاموس بنتائج الاختبار
    """
    # 1. توليد صورة عشوائية
    random_image = np.random.rand(1, *input_shape).astype(np.float32)
    
    # 2. التنبؤ
    start_time = tf.timestamp()
    predictions = model.predict(random_image, verbose=0)
    end_time = tf.timestamp()
    
    # 3. حساب وقت التنبؤ
    inference_time = float(end_time - start_time)
    
    return {
        'input_shape': input_shape,
        'output_shape': predictions.shape,
        'inference_time': inference_time,
        'predictions': predictions
    }

# ============================================================
# 8. تحميل النماذج المتعددة
# ============================================================

def load_multiple_models(model_paths: List[str]) -> Dict[str, tf.keras.Model]:
    """
    تحميل عدة نماذج من مسارات مختلفة.
    
    Args:
        model_paths: قائمة بمسارات النماذج
        
    Returns:
        قاموس بأسماء النماذج والنماذج المحملة
    """
    models = {}
    
    for path in model_paths:
        try:
            name = os.path.basename(path).replace('.h5', '').replace('.keras', '')
            model = load_model(path)
            if model is not None:
                models[name] = model
        except Exception as e:
            print(f"❌ خطأ في تحميل {path}: {e}")
    
    return models

# ============================================================
# 9. التحقق من صحة النموذج
# ============================================================

def validate_model(model: tf.keras.Model) -> Dict[str, Any]:
    """
    التحقق من صحة النموذج وتوافقه.
    
    Args:
        model: نموذج Keras
        
    Returns:
        قاموس بنتائج التحقق
    """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # 1. التحقق من وجود طبقات
    if len(model.layers) == 0:
        validation_result['is_valid'] = False
        validation_result['errors'].append("النموذج لا يحتوي على طبقات.")
    
    # 2. التحقق من المدخلات
    try:
        input_shape = model.input_shape
        if input_shape is None:
            validation_result['warnings'].append("لا يمكن تحديد شكل المدخلات.")
    except:
        validation_result['warnings'].append("تعذر الوصول إلى شكل المدخلات.")
    
    # 3. التحقق من المخرجات
    try:
        output_shape = model.output_shape
        if output_shape is None:
            validation_result['warnings'].append("لا يمكن تحديد شكل المخرجات.")
    except:
        validation_result['warnings'].append("تعذر الوصول إلى شكل المخرجات.")
    
    # 4. التحقق من وجود معلمات
    if model.count_params() == 0:
        validation_result['warnings'].append("النموذج لا يحتوي على معلمات قابلة للتدريب.")
    
    return validation_result

# ============================================================
# 10. واجهة تحميل النموذج في Streamlit
# ============================================================

def model_loader_ui(default_path: Optional[str] = None) -> Optional[tf.keras.Model]:
    """
    عرض واجهة تحميل النموذج في Streamlit.
    
    Args:
        default_path: المسار الافتراضي للنموذج
        
    Returns:
        النموذج المحمّل أو None
    """
    st.markdown("### 🤖 تحميل النموذج")
    
    # خيارات التحميل
    load_option = st.radio(
        "اختر طريقة التحميل:",
        ["تحميل من المسار الافتراضي", "رفع ملف النموذج", "اختيار المسار يدويًا"],
        index=0
    )
    
    model = None
    
    if load_option == "تحميل من المسار الافتراضي":
        if default_path and os.path.exists(default_path):
            model = load_model(default_path)
        else:
            st.warning("⚠️ لا يوجد مسار افتراضي صالح.")
    
    elif load_option == "رفع ملف النموذج":
        uploaded_file = st.file_uploader(
            "اختر ملف النموذج (.h5 أو .keras)",
            type=['h5', 'keras']
        )
        if uploaded_file:
            model = load_model_from_uploaded(uploaded_file)
    
    else:  # اختيار المسار يدويًا
        model_path = st.text_input("أدخل مسار النموذج:", value=default_path or "")
        if model_path and st.button("📂 تحميل النموذج"):
            model = load_model(model_path)
    
    # عرض معلومات النموذج إذا تم تحميله
    if model is not None:
        display_model_summary(model)
    
    return model

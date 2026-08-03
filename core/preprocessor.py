# ============================================================
# core/preprocessor.py
# معالجة الصور المسبقة - تجهيز الصور قبل التدريب والتنبؤ
# ============================================================

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any, Union
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# 1. تغيير حجم الصورة
# ============================================================

def resize_image(image: np.ndarray, 
                  target_size: Tuple[int, int] = (224, 224),
                  interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    تغيير حجم الصورة إلى الأبعاد المطلوبة.
    
    Args:
        image: الصورة كمصفوفة NumPy
        target_size: الأبعاد المستهدفة (width, height)
        interpolation: طريقة الاستيفاء
        
    Returns:
        الصورة بعد تغيير الحجم
    """
    return cv2.resize(image, target_size, interpolation=interpolation)

# ============================================================
# 2. تطبيع الصورة
# ============================================================

def normalize_image(image: np.ndarray, 
                     method: str = 'min_max') -> np.ndarray:
    """
    تطبيع قيم الصورة.
    
    Args:
        image: الصورة كمصفوفة NumPy
        method: طريقة التطبيع ('min_max', 'mean_std', 'global')
        
    Returns:
        الصورة بعد التطبيع
    """
    if method == 'min_max':
        # تطبيع إلى [0, 1]
        return image.astype(np.float32) / 255.0
    
    elif method == 'mean_std':
        # تطبيع باستخدام المتوسط والانحراف المعياري
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image.astype(np.float32) - mean) / std
        return image.astype(np.float32) / 255.0
    
    elif method == 'global':
        # تطبيع باستخدام قيم ثابتة (ImageNet)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = image.astype(np.float32) / 255.0
        
        # التأكد من الشكل
        if len(image_norm.shape) == 3 and image_norm.shape[2] == 3:
            return (image_norm - mean) / std
        else:
            return image_norm
    
    else:
        return image.astype(np.float32) / 255.0

# ============================================================
# 3. تحويل الصورة إلى تنسيق معين
# ============================================================

def convert_color_space(image: np.ndarray, 
                         to: str = 'RGB') -> np.ndarray:
    """
    تحويل الصورة بين مساحات الألوان.
    
    Args:
        image: الصورة كمصفوفة NumPy
        to: المساحة المستهدفة ('RGB', 'BGR', 'GRAY', 'LAB')
        
    Returns:
        الصورة بعد التحويل
    """
    if to == 'RGB':
        if len(image.shape) == 3 and image.shape[2] == 3:
            try:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                return image
        return image
    
    elif to == 'BGR':
        if len(image.shape) == 3 and image.shape[2] == 3:
            try:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except:
                return image
        return image
    
    elif to == 'GRAY':
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    elif to == 'LAB':
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return image
    
    else:
        return image

# ============================================================
# 4. تحسين جودة الصورة
# ============================================================

def enhance_image(image: np.ndarray,
                   brightness: float = 1.0,
                   contrast: float = 1.0,
                   sharpness: float = 1.0) -> np.ndarray:
    """
    تحسين جودة الصورة (السطوع، التباين، الحدة).
    
    Args:
        image: الصورة كمصفوفة NumPy
        brightness: عامل السطوع
        contrast: عامل التباين
        sharpness: عامل الحدة
        
    Returns:
        الصورة بعد التحسين
    """
    # 1. السطوع والتباين
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    # 2. الحدة
    if sharpness != 1.0:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * sharpness
        adjusted = cv2.filter2D(adjusted, -1, kernel)
    
    return adjusted

# ============================================================
# 5. إزالة التشويش
# ============================================================

def denoise_image(image: np.ndarray,
                   method: str = 'bilateral',
                   strength: float = 0.1) -> np.ndarray:
    """
    إزالة التشويش من الصورة.
    
    Args:
        image: الصورة كمصفوفة NumPy
        method: طريقة إزالة التشويش ('bilateral', 'gaussian', 'median')
        strength: قوة المعالجة
        
    Returns:
        الصورة بعد إزالة التشويش
    """
    if method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    elif method == 'gaussian':
        ksize = int(5 * strength + 3) if strength > 0 else 3
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    elif method == 'median':
        ksize = int(3 * strength + 3) if strength > 0 else 3
        if ksize % 2 == 0:
            ksize += 1
        return cv2.medianBlur(image, ksize)
    
    else:
        return image

# ============================================================
# 6. معالجة الصورة للتنبؤ (خطوة واحدة)
# ============================================================

def preprocess_for_prediction(image: np.ndarray,
                               target_size: Tuple[int, int] = (224, 224),
                               normalize: bool = True,
                               color_conversion: str = 'RGB') -> np.ndarray:
    """
    معالجة الصورة للتنبؤ بخطوة واحدة.
    
    Args:
        image: الصورة كمصفوفة NumPy
        target_size: الأبعاد المستهدفة
        normalize: تطبيع القيم
        color_conversion: تحويل الألوان
        
    Returns:
        الصورة جاهزة للتنبؤ (batch, height, width, channels)
    """
    # 1. تحويل الألوان
    if color_conversion:
        image = convert_color_space(image, color_conversion)
    
    # 2. تغيير الحجم
    resized = resize_image(image, target_size)
    
    # 3. تطبيع القيم
    if normalize:
        normalized = normalize_image(resized, 'min_max')
    else:
        normalized = resized.astype(np.float32)
    
    # 4. إضافة بعد الدفعة
    batched = np.expand_dims(normalized, axis=0)
    
    return batched

# ============================================================
# 7. معالجة الصورة للتدريب (دفعة واحدة)
# ============================================================

def preprocess_for_training(images: np.ndarray,
                             target_size: Tuple[int, int] = (224, 224),
                             normalize: bool = True) -> np.ndarray:
    """
    معالجة مجموعة من الصور للتدريب.
    
    Args:
        images: مصفوفة الصور
        target_size: الأبعاد المستهدفة
        normalize: تطبيع القيم
        
    Returns:
        الصور المعالجة
    """
    processed_images = []
    
    for image in images:
        # تغيير الحجم
        resized = resize_image(image, target_size)
        
        # تطبيع القيم
        if normalize:
            normalized = normalize_image(resized, 'min_max')
        else:
            normalized = resized.astype(np.float32)
        
        processed_images.append(normalized)
    
    return np.array(processed_images)

# ============================================================
# 8. استخراج الميزات من الصورة
# ============================================================

def extract_image_features(image: np.ndarray) -> Dict[str, Any]:
    """
    استخراج ميزات أساسية من الصورة.
    
    Args:
        image: الصورة كمصفوفة NumPy
        
    Returns:
        قاموس بالميزات المستخرجة
    """
    features = {}
    
    # 1. الأبعاد
    if len(image.shape) == 3:
        height, width, channels = image.shape
        features['height'] = height
        features['width'] = width
        features['channels'] = channels
    else:
        height, width = image.shape
        features['height'] = height
        features['width'] = width
        features['channels'] = 1
    
    # 2. الإحصائيات
    features['mean'] = float(np.mean(image))
    features['std'] = float(np.std(image))
    features['min'] = float(np.min(image))
    features['max'] = float(np.max(image))
    
    # 3. نسبة العرض إلى الارتفاع
    features['aspect_ratio'] = width / height if height > 0 else 0
    
    # 4. عدد البكسلات
    features['total_pixels'] = height * width
    
    return features

# ============================================================
# 9. عرض معلومات الصورة
# ============================================================

def display_image_info(image: np.ndarray) -> None:
    """
    عرض معلومات الصورة في واجهة Streamlit.
    
    Args:
        image: الصورة كمصفوفة NumPy
    """
    features = extract_image_features(image)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📐 الأبعاد", f"{features['width']} × {features['height']}")
    
    with col2:
        st.metric("🎨 القنوات", features['channels'])
    
    with col3:
        st.metric("📦 البكسلات", f"{features['total_pixels']:,}")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric("🌡️ المتوسط", f"{features['mean']:.2f}")
    
    with col5:
        st.metric("📊 الانحراف المعياري", f"{features['std']:.2f}")
    
    with col6:
        st.metric("📈 النسبة", f"{features['aspect_ratio']:.2f}")

# ============================================================
# 10. معالجة الصورة قبل العرض
# ============================================================

def prepare_image_for_display(image: np.ndarray,
                               normalize: bool = True) -> np.ndarray:
    """
    تجهيز الصورة للعرض في الواجهة.
    
    Args:
        image: الصورة كمصفوفة NumPy
        normalize: تطبيع القيم
        
    Returns:
        الصورة جاهزة للعرض
    """
    # 1. تحويل إلى 8-bit
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # 2. تحويل إلى RGB إذا كانت BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            pass
    
    return image

# ============================================================
# 11. معالجة الصور المرفوعة (دفعة واحدة)
# ============================================================

def process_uploaded_image(uploaded_file) -> Tuple[Optional[np.ndarray], Optional[str], Optional[Dict]]:
    """
    معالجة صورة مرفوعة من Streamlit.
    
    Args:
        uploaded_file: الملف المرفوع
        
    Returns:
        (الصورة، المسار، البيانات الوصفية)
    """
    if uploaded_file is None:
        return None, None, None
    
    try:
        # 1. قراءة الصورة
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            image_array = np.array(image)
        
        # 2. حفظ الملف مؤقتًا
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image.save(tmp_file.name)
            file_path = tmp_file.name
        
        # 3. البيانات الوصفية
        metadata = extract_image_features(image_array)
        metadata['file_name'] = uploaded_file.name
        metadata['file_size'] = uploaded_file.size
        metadata['file_path'] = file_path
        
        return image_array, file_path, metadata
    
    except Exception as e:
        st.error(f"❌ خطأ في معالجة الصورة: {e}")
        return None, None, None

# ============================================================
# 12. واجهة المعالجة في Streamlit
# ============================================================

def preprocessor_ui() -> None:
    """
    عرض واجهة معالجة الصور في Streamlit.
    """
    st.markdown("### 🛠️ معالجة الصورة")
    
    # 1. رفع الصورة
    uploaded_file = st.file_uploader(
        "اختر صورة للمعالجة",
        type=['jpg', 'jpeg', 'png'],
        key='preprocessor_uploader'
    )
    
    if uploaded_file is not None:
        # قراءة الصورة
        image, file_path, metadata = process_uploaded_image(uploaded_file)
        
        if image is not None:
            st.image(image, caption="الصورة الأصلية", use_container_width=True)
            
            # 2. خيارات المعالجة
            st.markdown("#### ⚙️ خيارات المعالجة")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_size = st.selectbox(
                    "حجم الصورة المستهدف:",
                    [(128, 128), (224, 224), (256, 256), (512, 512)],
                    index=1
                )
                
                normalize = st.checkbox("تطبيع القيم", value=True)
            
            with col2:
                color_mode = st.selectbox(
                    "نظام الألوان:",
                    ['RGB', 'BGR', 'GRAY', 'LAB'],
                    index=0
                )
                
                enhance = st.checkbox("تحسين الجودة", value=False)
            
            # 3. عرض الصورة المعالجة
            if st.button("🔄 معالجة الصورة", type="primary"):
                with st.spinner("⏳ جاري المعالجة..."):
                    # معالجة الصورة
                    processed = preprocess_for_prediction(
                        image, target_size, normalize, color_mode
                    )[0]
                    
                    # عرض النتائج
                    st.markdown("#### 📸 الصورة المعالجة")
                    st.image(processed, caption=f"الحجم: {target_size}", use_container_width=True)
                    
                    # عرض المعلومات
                    st.markdown("#### 📋 معلومات الصورة المعالجة")
                    display_image_info(processed)

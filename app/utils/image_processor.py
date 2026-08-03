# ============================================================
# app/utils/image_processor.py
# معالجة الصور - دوال لمعالجة وتحضير الصور للتنبؤ
# ============================================================

import numpy as np
import cv2
from PIL import Image
import streamlit as st
from typing import Optional, Tuple, List, Any, Dict
import io
import base64

# ============================================================
# 1. قراءة الصورة من مسار أو كائن
# ============================================================

def read_image(image_source: Union[str, np.ndarray, Image.Image, bytes]) -> Optional[np.ndarray]:
    """
    قراءة الصورة من مصادر مختلفة.
    
    Args:
        image_source: مصدر الصورة (مسار، مصفوفة، PIL Image، أو bytes)
        
    Returns:
        الصورة كمصفوفة NumPy (RGB) أو None في حالة الخطأ
    """
    try:
        # إذا كان المسار (str)
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
            if image is None:
                return None
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
        
        # إذا كانت مصفوفة NumPy
        elif isinstance(image_source, np.ndarray):
            if len(image_source.shape) == 3 and image_source.shape[2] == 3:
                return image_source
            return None
        
        # إذا كانت PIL Image
        elif isinstance(image_source, Image.Image):
            image_array = np.array(image_source)
            if image_source.mode != 'RGB':
                image_source = image_source.convert('RGB')
                image_array = np.array(image_source)
            return image_array
        
        # إذا كانت bytes
        elif isinstance(image_source, bytes):
            image = Image.open(io.BytesIO(image_source))
            image_array = np.array(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                image_array = np.array(image)
            return image_array
        
        else:
            return None
    
    except Exception as e:
        print(f"❌ خطأ في قراءة الصورة: {e}")
        return None

# ============================================================
# 2. تغيير حجم الصورة
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
# 3. تطبيع الصورة
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
        return (image_norm - mean) / std
    
    else:
        return image.astype(np.float32) / 255.0

# ============================================================
# 4. تحويل الصورة إلى تنسيق معين
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
            # تحقق إذا كانت BGR
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
# 5. تحسين جودة الصورة
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
# 6. إزالة التشويش
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
# 7. استخراج الميزات من الصورة
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
# 8. عرض معلومات الصورة
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
# 9. تحويل الصورة إلى Base64
# ============================================================

def image_to_base64(image: np.ndarray, 
                     format: str = 'PNG') -> str:
    """
    تحويل الصورة إلى Base64 للاستخدام في HTML.
    
    Args:
        image: الصورة كمصفوفة NumPy
        format: صيغة الصورة ('PNG', 'JPEG')
        
    Returns:
        النص المشفر Base64
    """
    # تحويل إلى PIL Image
    pil_image = Image.fromarray(np.uint8(image))
    
    # حفظ إلى BytesIO
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    
    # تشفير إلى Base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return image_base64

# ============================================================
# 10. تجهيز الصورة للتنبؤ (خطوة واحدة)
# ============================================================

def prepare_image_for_prediction(image: np.ndarray,
                                   target_size: Tuple[int, int] = (224, 224),
                                   normalize: bool = True) -> np.ndarray:
    """
    تجهيز الصورة للتنبؤ بخطوة واحدة.
    
    Args:
        image: الصورة كمصفوفة NumPy
        target_size: الأبعاد المستهدفة
        normalize: تطبيع القيم
        
    Returns:
        الصورة جاهزة للتنبؤ (batch, height, width, channels)
    """
    # 1. تغيير الحجم
    resized = resize_image(image, target_size)
    
    # 2. تطبيع القيم
    if normalize:
        normalized = normalize_image(resized, 'min_max')
    else:
        normalized = resized.astype(np.float32)
    
    # 3. إضافة بعد الدفعة
    batched = np.expand_dims(normalized, axis=0)
    
    return batched

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
        image = read_image(uploaded_file.getvalue())
        if image is None:
            return None, None, None
        
        # 2. حفظ الملف مؤقتًا
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            pil_image = Image.fromarray(np.uint8(image))
            pil_image.save(tmp_file.name)
            file_path = tmp_file.name
        
        # 3. البيانات الوصفية
        metadata = extract_image_features(image)
        metadata['file_name'] = uploaded_file.name
        metadata['file_size'] = uploaded_file.size
        metadata['file_path'] = file_path
        
        return image, file_path, metadata
    
    except Exception as e:
        st.error(f"❌ خطأ في معالجة الصورة: {e}")
        return None, None, None

# ============================================================
# 12. إنشاء مصغرات للصور (Thumbnails)
# ============================================================

def create_thumbnail(image: np.ndarray,
                      size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    إنشاء صورة مصغرة (Thumbnail).
    
    Args:
        image: الصورة الأصلية
        size: أبعاد المصغرة
        
    Returns:
        الصورة المصغرة
    """
    return resize_image(image, size)

# ============================================================
# 13. قص الصورة (Crop)
# ============================================================

def crop_image(image: np.ndarray,
                x: int, y: int,
                width: int, height: int) -> np.ndarray:
    """
    قص الصورة إلى منطقة محددة.
    
    Args:
        image: الصورة الأصلية
        x, y: إحداثيات البداية
        width, height: أبعاد المنطقة المقتطعة
        
    Returns:
        الصورة المقتطعة
    """
    h, w = image.shape[:2]
    
    # التأكد من أن الإحداثيات ضمن حدود الصورة
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    width = min(width, w - x)
    height = min(height, h - y)
    
    return image[y:y+height, x:x+width]

# ============================================================
# 14. دمج صورتين جنبًا إلى جنب
# ============================================================

def concatenate_images(image1: np.ndarray,
                        image2: np.ndarray,
                        axis: int = 1) -> np.ndarray:
    """
    دمج صورتين جنبًا إلى جنب (أفقي أو رأسي).
    
    Args:
        image1: الصورة الأولى
        image2: الصورة الثانية
        axis: محور الدمج (0: رأسي، 1: أفقي)
        
    Returns:
        الصورة المدمجة
    """
    # جعل الصور بنفس الحجم
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    if axis == 1:  # أفقي
        if h1 != h2:
            # تغيير حجم الصورة الثانية لتتناسب مع الأولى
            image2 = resize_image(image2, (w2, h1))
        return np.concatenate([image1, image2], axis=1)
    
    else:  # رأسي
        if w1 != w2:
            image2 = resize_image(image2, (w1, h2))
        return np.concatenate([image1, image2], axis=0)

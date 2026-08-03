Uploader · PY
# ============================================================
# app/components/uploader.py
# مكون رفع الصور - إدارة رفع ومعالجة الصور في واجهة Streamlit
# ============================================================
 
import streamlit as st
import os
import tempfile
from typing import Optional, Tuple, List, Any
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
 
# ============================================================
# 1. إعدادات رفع الملفات
# ============================================================
 
# الأنواع المسموحة
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']
 
# الحد الأقصى لحجم الملف (بالميجابايت)
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
 
# الأبعاد المطلوبة للنموذج
TARGET_SIZE = (224, 224)
 
# ============================================================
# 2. التحقق من صحة الملف
# ============================================================
 
def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    التحقق من صحة الملف المرفوع.
    
    Args:
        uploaded_file: الملف المرفوع من Streamlit
        
    Returns:
        (is_valid, message): نتيجة التحقق ورسالة الخطأ (إن وجدت)
    """
    # 1. التحقق من وجود الملف
    if uploaded_file is None:
        return False, "❌ لم يتم اختيار أي ملف."
    
    # 2. التحقق من الحجم
    uploaded_file.seek(0, os.SEEK_END)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)
    
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"❌ حجم الملف يتجاوز الحد المسموح ({MAX_FILE_SIZE_MB} MB)."
    
    if file_size == 0:
        return False, "❌ الملف فارغ."
    
    # 3. التحقق من الامتداد
    file_name = uploaded_file.name
    extension = file_name.split('.')[-1].lower()
    
    if extension not in ALLOWED_EXTENSIONS:
        return False, f"❌ صيغة الملف غير مسموحة. الصيغ المسموحة: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # 4. التحقق من نوع الملف (MIME type)
    if uploaded_file.type not in ALLOWED_MIME_TYPES:
        return False, f"❌ نوع الملف غير مدعوم. الأنواع المدعومة: {', '.join(ALLOWED_MIME_TYPES)}"
    
    return True, "✅ الملف صالح."
 
# ============================================================
# 3. قراءة الصورة ومعالجتها
# ============================================================
 
def load_image(uploaded_file) -> Optional[np.ndarray]:
    """
    قراءة الصورة من الملف المرفوع وتحويلها إلى مصفوفة NumPy.
    
    Args:
        uploaded_file: الملف المرفوع من Streamlit
        
    Returns:
        الصورة كمصفوفة NumPy (RGB) أو None في حالة الخطأ
    """
    try:
        # قراءة الصورة باستخدام PIL
        image = Image.open(uploaded_file)
        
        # تحويل إلى RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # تحويل إلى مصفوفة NumPy
        image_array = np.array(image)
        
        return image_array
    
    except Exception as e:
        st.error(f"❌ خطأ في قراءة الصورة: {e}")
        return None
 
# ============================================================
# 4. معالجة الصورة للتنبؤ
# ============================================================
 
def preprocess_image_for_prediction(image: np.ndarray, 
                                     target_size: Tuple[int, int] = TARGET_SIZE) -> Optional[np.ndarray]:
    """
    معالجة الصورة لتكون جاهزة للتنبؤ.
    
    Args:
        image: الصورة كمصفوفة NumPy
        target_size: الحجم المطلوب للصورة
        
    Returns:
        الصورة المعالجة جاهزة للتنبؤ أو None في حالة الخطأ
    """
    try:
        # 1. تغيير الحجم
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 2. تطبيع القيم إلى [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # 3. إضافة بعد الدفعة (Batch dimension)
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    except Exception as e:
        st.error(f"❌ خطأ في معالجة الصورة: {e}")
        return None
 
# ============================================================
# 5. حفظ الصورة المرفوعة مؤقتًا
# ============================================================
 
def save_uploaded_file(uploaded_file, 
                        save_dir: str = "uploads/temp") -> Optional[str]:
    """
    حفظ الملف المرفوع في مجلد مؤقت.
    
    Args:
        uploaded_file: الملف المرفوع من Streamlit
        save_dir: مجلد الحفظ المؤقت
        
    Returns:
        مسار الملف المحفوظ أو None في حالة الخطأ
    """
    try:
        # إنشاء المجلد إذا لم يكن موجودًا
        os.makedirs(save_dir, exist_ok=True)
        
        # إنشاء اسم ملف فريد
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(save_dir, file_name)
        
        # حفظ الملف
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    except Exception as e:
        st.error(f"❌ خطأ في حفظ الملف: {e}")
        return None
 
# ============================================================
# 6. حذف الملفات المؤقتة
# ============================================================
 
def cleanup_temp_files(temp_dir: str = "uploads/temp") -> None:
    """
    حذف جميع الملفات المؤقتة في المجلد.
    
    Args:
        temp_dir: مجلد الملفات المؤقتة
    """
    try:
        if os.path.exists(temp_dir):
            for file_name in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    except Exception as e:
        st.warning(f"⚠️ تعذر حذف الملفات المؤقتة: {e}")
 
# ============================================================
# 7. عرض معاينة الصورة
# ============================================================
 
def display_image_preview(image: np.ndarray, caption: Optional[str] = None) -> None:
    """
    عرض معاينة الصورة في الواجهة.
    
    Args:
        image: الصورة كمصفوفة NumPy
        caption: النص التوضيحي (اختياري)
    """
    if image is None:
        st.warning("⚠️ لا توجد صورة لعرضها.")
        return
    
    # تحويل إلى RGB إذا كانت BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        if isinstance(image[0,0,0], np.int64) or isinstance(image[0,0,0], np.uint8):
            # احتمالية أن تكون BGR
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                image_rgb = image
        else:
            image_rgb = image
    else:
        image_rgb = image
    
    # عرض الصورة
    st.image(image_rgb, caption=caption, use_column_width=True)
 
# ============================================================
# 8. مكون رفع الصورة الكامل
# ============================================================
 
def render_uploader(show_preview: bool = True,
                     show_validation: bool = True) -> Tuple[Optional[np.ndarray], 
                                                            Optional[str],
                                                            Optional[dict]]:
    """
    عرض مكون رفع الصورة الكامل مع جميع الوظائف.
    
    Args:
        show_preview: عرض معاينة الصورة بعد رفعها
        show_validation: عرض رسائل التحقق من صحة الملف
        
    Returns:
        (image, file_path, metadata): الصورة، مسار الملف، بيانات وصفية
    """
    st.markdown("### 📤 رفع صورة MRI")
    
    # 1. مكون رفع الملف
    uploaded_file = st.file_uploader(
        "اختر صورة MRI للتحليل",
        type=ALLOWED_EXTENSIONS,
        help=f"الصيغ المسموحة: {', '.join(ALLOWED_EXTENSIONS)}. الحد الأقصى للحجم: {MAX_FILE_SIZE_MB} MB"
    )
    
    # 2. التحقق من وجود ملف
    if uploaded_file is None:
        return None, None, None
    
    # 3. التحقق من صحة الملف
    is_valid, message = validate_file(uploaded_file)
    
    if show_validation:
        if is_valid:
            st.success(message)
        else:
            st.error(message)
            return None, None, None
    
    # 4. قراءة الصورة
    image = load_image(uploaded_file)
    
    if image is None:
        return None, None, None
    
    # 5. حفظ الملف
    file_path = save_uploaded_file(uploaded_file)
    
    # 6. عرض معاينة الصورة
    if show_preview:
        st.markdown("#### 📷 معاينة الصورة")
        display_image_preview(image, caption=f"📎 {uploaded_file.name}")
        
        # عرض معلومات الصورة
        st.markdown("#### ℹ️ معلومات الصورة")
        col1, col2, col3 = st.columns(3)
        col1.metric("📐 الأبعاد", f"{image.shape[1]} × {image.shape[0]}")
        col2.metric("🎨 الألوان", f"{image.shape[2] if len(image.shape) > 2 else 'أبيض وأسود'}")
        col3.metric("📦 الحجم", f"{uploaded_file.size / 1024:.1f} KB")
    
    # 7. بيانات وصفية
    metadata = {
        'file_name': uploaded_file.name,
        'file_size': uploaded_file.size,
        'image_shape': image.shape,
        'file_path': file_path,
        'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return image, file_path, metadata
 
# ============================================================
# 9. مكون رفع متعدد الصور
# ============================================================
 
def render_multi_uploader(max_files: int = 5) -> Tuple[List[np.ndarray], 
                                                         List[str], 
                                                         List[dict]]:
    """
    عرض مكون رفع متعدد الصور.
    
    Args:
        max_files: الحد الأقصى لعدد الصور
        
    Returns:
        (images, file_paths, metadata_list): قوائم الصور، المسارات، البيانات الوصفية
    """
    st.markdown("### 📤 رفع صور متعددة")
    
    uploaded_files = st.file_uploader(
        "اختر صور MRI للتحليل",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        help=f"يمكنك رفع حتى {max_files} صور. الصيغ المسموحة: {', '.join(ALLOWED_EXTENSIONS)}"
    )
    
    if not uploaded_files:
        return [], [], []
    
    # معالجة كل ملف
    images = []
    file_paths = []
    metadata_list = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        if i >= max_files:
            st.warning(f"⚠️ تم الوصول للحد الأقصى ({max_files} صور). تم تجاهل الباقي.")
            break
        
        # التحقق من صحة الملف
        is_valid, message = validate_file(uploaded_file)
        if not is_valid:
            st.warning(f"⚠️ الملف {uploaded_file.name}: {message}")
            continue
        
        # قراءة الصورة
        image = load_image(uploaded_file)
        if image is None:
            continue
        
        # حفظ الملف
        file_path = save_uploaded_file(uploaded_file)
        
        images.append(image)
        file_paths.append(file_path)
        metadata_list.append({
            'file_name': uploaded_file.name,
            'file_size': uploaded_file.size,
            'image_shape': image.shape,
            'file_path': file_path
        })
    
    return images, file_paths, metadata_list
 
# ============================================================
# 10. عرض أزرار التحكم (رفع، مسح، تحليل)
# ============================================================
 
def display_upload_controls() -> Tuple[bool, bool, bool]:
    """
    عرض أزرار التحكم في رفع الصور.
    
    Returns:
        (upload_clicked, clear_clicked, analyze_clicked)
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        upload_clicked = st.button("📤 رفع صورة", use_container_width=True)
    
    with col2:
        clear_clicked = st.button("🗑️ مسح الكل", use_container_width=True)
    
    with col3:
        analyze_clicked = st.button("🔍 تحليل", use_container_width=True, type="primary")
    
    return upload_clicked, clear_clicked, analyze_clicked
 
# ============================================================
# 11. عرض أزرار التحكم المتقدمة
# ============================================================
 
def display_advanced_controls() -> Dict[str, Any]:
    """
    عرض أزرار تحكم متقدمة (خيارات إضافية).
    
    Returns:
        قاموس بخيارات التحكم المتقدمة
    """
    with st.expander("⚙️ خيارات متقدمة", expanded=False):
        # خيارات المعالجة
        use_gradcam = st.checkbox("🔥 عرض Grad-CAM", value=True)
        show_confidence = st.checkbox("📊 عرض نسبة الثقة", value=True)
        
        # خيارات الحفظ
        save_results = st.checkbox("💾 حفظ النتائج", value=False)
        
        # خيارات العرض
        show_all_probabilities = st.checkbox("📈 عرض احتمالات جميع الفئات", value=True)
        
        return {
            'use_gradcam': use_gradcam,
            'show_confidence': show_confidence,
            'save_results': save_results,
            'show_all_probabilities': show_all_probabilities
        }
 
# ============================================================
# 12. مكون رفع الصورة الكامل مع التحكم
# ============================================================
 
def render_full_upload_section(show_controls: bool = True,
                                show_advanced: bool = True) -> Tuple[Optional[np.ndarray],
                                                                      Optional[str],
                                                                      Optional[dict],
                                                                      Dict[str, Any]]:
    """
    عرض قسم رفع الصورة الكامل مع جميع خيارات التحكم.
    
    Args:
        show_controls: عرض أزرار التحكم
        show_advanced: عرض الخيارات المتقدمة
        
    Returns:
        (image, file_path, metadata, controls): الصورة، المسار، البيانات، خيارات التحكم
    """
    # 1. مكون رفع الصورة الأساسي
    image, file_path, metadata = render_uploader(show_preview=True, show_validation=True)
    
    # 2. أزرار التحكم
    controls = {}
    
    if show_controls and image is not None:
        upload_clicked, clear_clicked, analyze_clicked = display_upload_controls()
        controls = {
            'upload_clicked': upload_clicked,
            'clear_clicked': clear_clicked,
            'analyze_clicked': analyze_clicked
        }
    
    # 3. خيارات متقدمة
    if show_advanced:
        advanced_options = display_advanced_controls()
        controls.update(advanced_options)
    
    return image, file_path, metadata, controls
 
# ============================================================
# 13. معاينة الصورة قبل المعالجة
# ============================================================
 
def preview_uploaded_image(image: np.ndarray, 
                            target_size: Tuple[int, int] = TARGET_SIZE) -> None:
    """
    عرض معاينة الصورة قبل وبعد المعالجة.
    
    Args:
        image: الصورة الأصلية
        target_size: الحجم المستهدف
    """
    if image is None:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔹 الصورة الأصلية")
        display_image_preview(image, caption=f"الأبعاد: {image.shape[1]} × {image.shape[0]}")
    
    with col2:
        st.markdown("#### 🔸 الصورة المعالجة")
        
        # معالجة الصورة
        processed = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        display_image_preview(processed, caption=f"الأبعاد: {target_size[0]} × {target_size[1]}")
 

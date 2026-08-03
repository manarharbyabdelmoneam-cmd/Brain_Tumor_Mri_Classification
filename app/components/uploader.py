# ============================================================
# app/components/uploader.py
# مكون رفع الصور - إدارة رفع ومعالجة الصور في واجهة Streamlit
# ============================================================

import streamlit as st
import os
import tempfile
from typing import Optional, Tuple, List, Any, Dict
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

# ============================================================
# 1. إعدادات رفع الملفات
# ============================================================

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

TARGET_SIZE = (224, 224)

# ============================================================
# 2. التحقق من صحة الملف
# ============================================================

def validate_file(uploaded_file) -> Tuple[bool, str]:
    if uploaded_file is None:
        return False, "❌ لم يتم اختيار أي ملف."

    uploaded_file.seek(0, os.SEEK_END)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)

    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"❌ حجم الملف يتجاوز الحد المسموح ({MAX_FILE_SIZE_MB} MB)."

    if file_size == 0:
        return False, "❌ الملف فارغ."

    extension = uploaded_file.name.split('.')[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False, f"❌ صيغة الملف غير مسموحة. الصيغ المسموحة: {', '.join(ALLOWED_EXTENSIONS)}"

    if uploaded_file.type not in ALLOWED_MIME_TYPES:
        return False, f"❌ نوع الملف غير مدعوم. الأنواع المدعومة: {', '.join(ALLOWED_MIME_TYPES)}"

    return True, "✅ الملف صالح."

# ============================================================
# 3. قراءة الصورة ومعالجتها
# ============================================================

def load_image(uploaded_file) -> Optional[np.ndarray]:
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        st.error(f"❌ خطأ في قراءة الصورة: {e}")
        return None

# ============================================================
# 4. معالجة الصورة للتنبؤ
# ============================================================

def preprocess_image_for_prediction(image: np.ndarray,
                                     target_size: Tuple[int, int] = TARGET_SIZE) -> Optional[np.ndarray]:
    try:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)
    except Exception as e:
        st.error(f"❌ خطأ في معالجة الصورة: {e}")
        return None

# ============================================================
# 5. حفظ الصورة المرفوعة مؤقتًا
# ============================================================

def save_uploaded_file(uploaded_file,
                        save_dir: str = "uploads/temp") -> Optional[str]:
    try:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(save_dir, file_name)
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
    if image is None:
        st.warning("⚠️ لا توجد صورة لعرضها.")
        return

    # تحويل numpy array إلى PIL Image لضمان التوافق مع st.image
    try:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = image
        st.image(pil_image, caption=caption, width=700)
    except Exception as e:
        st.error(f"❌ خطأ في عرض الصورة: {e}")

# ============================================================
# 8. مكون رفع الصورة الكامل
# ============================================================

def render_uploader(show_preview: bool = True,
                     show_validation: bool = True) -> Tuple[Optional[np.ndarray],
                                                            Optional[str],
                                                            Optional[dict],
                                                            Optional[dict]]:
    st.markdown("### 📤 رفع صورة MRI")

    uploaded_file = st.file_uploader(
        "اختر صورة MRI للتحليل",
        type=ALLOWED_EXTENSIONS,
        help=f"الصيغ المسموحة: {', '.join(ALLOWED_EXTENSIONS)}. الحد الأقصى للحجم: {MAX_FILE_SIZE_MB} MB"
    )

    if uploaded_file is None:
        return None, None, None, None

    is_valid, message = validate_file(uploaded_file)

    if show_validation:
        if is_valid:
            st.success(message)
        else:
            st.error(message)
            return None, None, None, None

    image = load_image(uploaded_file)
    if image is None:
        return None, None, None, None

    file_path = save_uploaded_file(uploaded_file)

    if show_preview:
        st.markdown("#### 📷 معاينة الصورة")
        display_image_preview(image, caption=f"📎 {uploaded_file.name}")

        st.markdown("#### ℹ️ معلومات الصورة")
        col1, col2, col3 = st.columns(3)
        col1.metric("📐 الأبعاد", f"{image.shape[1]} × {image.shape[0]}")
        col2.metric("🎨 الألوان", f"{image.shape[2] if len(image.shape) > 2 else 'أبيض وأسود'}")
        col3.metric("📦 الحجم", f"{uploaded_file.size / 1024:.1f} KB")

    metadata = {
        'file_name': uploaded_file.name,
        'file_size': uploaded_file.size,
        'image_shape': image.shape,
        'file_path': file_path,
        'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    controls = {
        'upload_clicked': False,
        'clear_clicked': False,
        'analyze_clicked': False
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        # ✅ عرض الصورة بعرض ثابت للتوافق مع جميع الـ Streamlit versions
        if st.button("📤 رفع صورة", width=700):
            controls['upload_clicked'] = True

    with col2:
        if st.button("🗑️ مسح الكل", width=700):
            controls['clear_clicked'] = True

    with col3:
        if st.button("🔍 تحليل", width=700, type="primary"):
            controls['analyze_clicked'] = True

    return image, file_path, metadata, controls

# ============================================================
# 9. مكون رفع متعدد الصور
# ============================================================

def render_multi_uploader(max_files: int = 5) -> Tuple[List[np.ndarray],
                                                         List[str],
                                                         List[dict]]:
    st.markdown("### 📤 رفع صور متعددة")

    uploaded_files = st.file_uploader(
        "اختر صور MRI للتحليل",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        help=f"يمكنك رفع حتى {max_files} صور. الصيغ المسموحة: {', '.join(ALLOWED_EXTENSIONS)}"
    )

    if not uploaded_files:
        return [], [], []

    images, file_paths, metadata_list = [], [], []

    for i, uploaded_file in enumerate(uploaded_files):
        if i >= max_files:
            st.warning(f"⚠️ تم الوصول للحد الأقصى ({max_files} صور). تم تجاهل الباقي.")
            break

        is_valid, message = validate_file(uploaded_file)
        if not is_valid:
            st.warning(f"⚠️ الملف {uploaded_file.name}: {message}")
            continue

        image = load_image(uploaded_file)
        if image is None:
            continue

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
# 10. عرض أزرار التحكم
# ============================================================

def display_upload_controls() -> Tuple[bool, bool, bool]:
    col1, col2, col3 = st.columns(3)

    with col1:
        # ✅ عرض الصورة بعرض ثابت للتوافق مع جميع الـ Streamlit versions
        upload_clicked = st.button("📤 رفع صورة", width=700)

    with col2:
        clear_clicked = st.button("🗑️ مسح الكل", width=700)

    with col3:
        analyze_clicked = st.button("🔍 تحليل", width=700, type="primary")

    return upload_clicked, clear_clicked, analyze_clicked

# ============================================================
# 11. عرض أزرار التحكم المتقدمة
# ============================================================

def display_advanced_controls() -> Dict[str, Any]:
    with st.expander("⚙️ خيارات متقدمة", expanded=False):
        use_gradcam = st.checkbox("🔥 عرض Grad-CAM", value=True)
        show_confidence = st.checkbox("📊 عرض نسبة الثقة", value=True)
        save_results = st.checkbox("💾 حفظ النتائج", value=False)
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
    image, file_path, metadata, controls = render_uploader(show_preview=True, show_validation=True)

    if show_controls and image is not None:
        upload_clicked, clear_clicked, analyze_clicked = display_upload_controls()
        controls = {
            'upload_clicked': upload_clicked,
            'clear_clicked': clear_clicked,
            'analyze_clicked': analyze_clicked
        }

    if show_advanced:
        advanced_options = display_advanced_controls()
        if controls:
            controls.update(advanced_options)

    return image, file_path, metadata, controls

# ============================================================
# 13. معاينة الصورة قبل وبعد المعالجة
# ============================================================

def preview_uploaded_image(image: np.ndarray,
                            target_size: Tuple[int, int] = TARGET_SIZE) -> None:
    if image is None:
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔹 الصورة الأصلية")
        display_image_preview(image, caption=f"الأبعاد: {image.shape[1]} × {image.shape[0]}")

    with col2:
        st.markdown("#### 🔸 الصورة المعالجة")
        processed = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        display_image_preview(processed, caption=f"الأبعاد: {target_size[0]} × {target_size[1]}")

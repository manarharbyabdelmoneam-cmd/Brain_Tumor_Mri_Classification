# ============================================================
# app/utils/predictor.py
# منطق التنبؤ - إدارة التنبؤات ومعالجة الصور للتصنيف
# ============================================================

import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, List, Tuple, Union
import streamlit as st
import pandas as pd
from datetime import datetime

from app.utils.image_processor import prepare_image_for_prediction, read_image
from app.utils.gradcam import generate_gradcam_for_image, get_last_conv_layer
from app.utils.formatter import format_class_name, format_confidence, format_prediction_results

# ============================================================
# 1. التنبؤ بصورة واحدة
# ============================================================

def predict_single_image(model: tf.keras.Model,
                          image: np.ndarray,
                          class_names: List[str],
                          target_size: Tuple[int, int] = (224, 224),
                          use_gradcam: bool = True) -> Dict[str, Any]:

    # ✅ التأكد من أن class_names قائمة وليست dict
    if isinstance(class_names, dict):
        class_names = list(class_names.values())
    else:
        class_names = list(class_names)

    # 1. معالجة الصورة للتنبؤ
    processed_image = prepare_image_for_prediction(image, target_size, normalize=True)

    # 2. التنبؤ
    predictions = model.predict(processed_image, verbose=0)

    # ✅ تحويل الـ index إلى Python int عادي لتجنب KeyError مع numpy.int64
    predicted_class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class_idx])

    # ✅ التحقق من أن الـ index في نطاق القائمة
    if predicted_class_idx >= len(class_names):
        raise ValueError(
            f"predicted_class_idx={predicted_class_idx} خارج نطاق class_names (حجمها {len(class_names)}). "
            f"تأكد من تطابق عدد فئات النموذج مع class_names."
        )

    predicted_class = class_names[predicted_class_idx]

    # 3. توليد Grad-CAM
    gradcam_image = None
    heatmap = None

    if use_gradcam:
        try:
            original_uint8 = np.uint8(image * 255) if image.max() <= 1 else np.uint8(image)
            result = generate_gradcam_for_image(model, original_uint8, class_names)
            gradcam_image = result['gradcam_image']
            heatmap = result['heatmap']
        except Exception as e:
            print(f"⚠️ تعذر توليد Grad-CAM: {e}")

    # 4. بناء النتائج
    result = {
        'predictions': predictions[0],
        'predicted_class': predicted_class,
        'predicted_class_idx': predicted_class_idx,
        'confidence': confidence,
        'processed_image': processed_image,
        'gradcam_image': gradcam_image,
        'heatmap': heatmap,
        'class_names': class_names
    }

    return result

# ============================================================
# 2. التنبؤ مع تنسيق النتائج
# ============================================================

def predict_and_format(model: tf.keras.Model,
                        image: np.ndarray,
                        class_names: List[str],
                        target_size: Tuple[int, int] = (224, 224),
                        use_gradcam: bool = True) -> Optional[Dict[str, Any]]:

    try:
        # ✅ التأكد من أن class_names قائمة
        if isinstance(class_names, dict):
            class_names = list(class_names.values())
        else:
            class_names = list(class_names)

        result = predict_single_image(model, image, class_names, target_size, use_gradcam)

        formatted = {
            'predicted_class': format_class_name(result['predicted_class']),
            'confidence': format_confidence(result['confidence']),
            'confidence_value': result['confidence'],
            'predictions': result['predictions'],
            'class_names': result['class_names'],
            'predicted_class_idx': result['predicted_class_idx'],
            'gradcam_image': result['gradcam_image'],
            'heatmap': result['heatmap'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # احتمالات كل فئة مرتبة تنازليًا
        prob_dict = {}
        for i, name in enumerate(class_names):
            prob_dict[name] = float(result['predictions'][i])

        formatted['probabilities'] = dict(
            sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return formatted

    except Exception as e:
        st.error(f"❌ خطأ في التنبؤ: {e}")
        return None

# ============================================================
# 3. التنبؤ بعدة صور دفعة واحدة
# ============================================================

def predict_batch(model: tf.keras.Model,
                   images: List[np.ndarray],
                   class_names: List[str],
                   target_size: Tuple[int, int] = (224, 224)) -> List[Dict[str, Any]]:

    if isinstance(class_names, dict):
        class_names = list(class_names.values())
    else:
        class_names = list(class_names)

    results = []

    batch_images = []
    for image in images:
        processed = prepare_image_for_prediction(image, target_size, normalize=True)
        batch_images.append(processed)

    batch = np.vstack(batch_images)
    predictions = model.predict(batch, verbose=0)

    for i, pred in enumerate(predictions):
        predicted_class_idx = int(np.argmax(pred))
        confidence = float(pred[predicted_class_idx])
        predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else "unknown"

        results.append({
            'index': i,
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'predictions': pred,
            'class_names': class_names
        })

    return results

# ============================================================
# 4. التنبؤ من ملف مرفوع
# ============================================================

def predict_from_uploaded(model: tf.keras.Model,
                           uploaded_file,
                           class_names: List[str],
                           target_size: Tuple[int, int] = (224, 224),
                           use_gradcam: bool = True) -> Optional[Dict[str, Any]]:
    try:
        image = read_image(uploaded_file.getvalue())

        if image is None:
            st.error("❌ تعذر قراءة الصورة.")
            return None

        result = predict_and_format(model, image, class_names, target_size, use_gradcam)

        if result is not None:
            result['file_name'] = uploaded_file.name
            result['file_size'] = uploaded_file.size

        return result

    except Exception as e:
        st.error(f"❌ خطأ في التنبؤ: {e}")
        return None

# ============================================================
# 5. عرض نتائج التنبؤ في Streamlit
# ============================================================

def display_prediction_results(result: Dict[str, Any]) -> None:
    if result is None:
        st.warning("⚠️ لا توجد نتائج لعرضها.")
        return

    st.markdown("### 🧠 نتائج التنبؤ")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, #1E1E1E, #2A2A2A);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 2px solid #4A90D9;
        ">
            <div style="color: #AAAAAA; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;">
                الفئة المتوقعة
            </div>
            <div style="color: #4A90D9; font-size: 2rem; font-weight: 700; margin: 0.5rem 0;">
                {result['predicted_class']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        confidence_color = (
            "#4CAF50" if result['confidence_value'] >= 0.80
            else "#FFD93D" if result['confidence_value'] >= 0.60
            else "#FF6B6B"
        )

        st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, #1E1E1E, #2A2A2A);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 2px solid {confidence_color};
        ">
            <div style="color: #AAAAAA; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;">
                نسبة الثقة
            </div>
            <div style="color: {confidence_color}; font-size: 2rem; font-weight: 700; margin: 0.5rem 0;">
                {result['confidence']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 احتمالات جميع الفئات")

    class_names = result['class_names']
    probabilities = result['predictions']
    sorted_indices = np.argsort(probabilities)[::-1]

    # ✅ استخراج اسم الفئة الأصلي بدون emoji للمقارنة
    raw_predicted = result['predicted_class'].replace('🧬 ', '').replace('✅ ', '').strip()

    for idx in sorted_indices:
        idx = int(idx)
        class_name = class_names[idx]
        prob = float(probabilities[idx])
        is_predicted = (class_name == raw_predicted)
        color = "#4A90D9" if is_predicted else "#333333"

        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                <span style="color: {'white' if is_predicted else '#AAAAAA'}; font-weight: {'bold' if is_predicted else 'normal'};">
                    {class_name}{' 👈' if is_predicted else ''}
                </span>
                <span style="color: #4A90D9; font-weight: 600;">{prob:.1%}</span>
            </div>
            <div style="background-color: #333333; border-radius: 8px; height: 16px; overflow: hidden;">
                <div style="
                    width: {prob * 100:.1f}%;
                    height: 100%;
                    background: linear-gradient(90deg, {color}, {color}CC);
                    border-radius: 8px;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if result.get('gradcam_image') is not None:
        st.markdown("### 🔥 Grad-CAM (المنطقة المؤثرة)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📷 الصورة الأصلية")
            if result.get('original_image') is not None:
                st.image(result['original_image'], use_container_width=True)
            else:
                st.info("لا توجد صورة أصلية لعرضها.")

        with col2:
            st.markdown("#### 🔥 Grad-CAM")
            st.image(result['gradcam_image'], use_container_width=True)
            st.caption("المناطق الحمراء: المناطق التي ركز عليها النموذج")

    with st.expander("📋 معلومات إضافية", expanded=False):
        st.markdown(f"""
        - **⏱️ وقت التنبؤ:** {result.get('inference_time', 'غير محدد')}
        - **📅 التاريخ:** {result.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}
        - **📁 اسم الملف:** {result.get('file_name', 'غير محدد')}
        - **📦 حجم الملف:** {result.get('file_size', 'غير محدد')}
        """)

# ============================================================
# 6. واجهة التنبؤ في Streamlit
# ============================================================

def predictor_ui(model: tf.keras.Model,
                  class_names: List[str],
                  target_size: Tuple[int, int] = (224, 224)) -> None:

    st.markdown("### 🔍 التنبؤ")

    uploaded_file = st.file_uploader(
        "اختر صورة MRI للتحليل",
        type=['jpg', 'jpeg', 'png'],
        help="الصيغ المسموحة: JPG, JPEG, PNG"
    )

    col1, col2 = st.columns(2)

    with col1:
        use_gradcam = st.checkbox("🔥 عرض Grad-CAM", value=True)

    with col2:
        if st.button("🔍 تحليل الصورة", type="primary", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner("⏳ جاري تحليل الصورة..."):
                    result = predict_from_uploaded(
                        model, uploaded_file, class_names, target_size, use_gradcam
                    )
                    if result is not None:
                        st.session_state['prediction_result'] = result
                        st.rerun()
            else:
                st.warning("⚠️ يرجى اختيار صورة أولاً.")

    if 'prediction_result' in st.session_state:
        result = st.session_state['prediction_result']
        display_prediction_results(result)

        if st.button("🗑️ مسح النتائج"):
            del st.session_state['prediction_result']
            st.rerun()

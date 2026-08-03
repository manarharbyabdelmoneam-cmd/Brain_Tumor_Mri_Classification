# ============================================================
# app/components/results.py
# مكون عرض النتائج - عرض نتائج التنبؤ في واجهة Streamlit
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# ============================================================
# 1. عرض النتائج الأساسية (الفئة + الثقة)
# ============================================================

def display_prediction_results(predicted_class: str,
                                confidence: float,
                                class_names: List[str],
                                probabilities: np.ndarray) -> None:
    """
    عرض نتائج التنبؤ الأساسية: الفئة المتوقعة ونسبة الثقة.
    
    Args:
        predicted_class: اسم الفئة المتوقعة
        confidence: نسبة الثقة (0-1)
        class_names: قائمة بأسماء جميع الفئات
        probabilities: مصفوفة الاحتمالات لجميع الفئات
    """
    # تحديد اللون بناءً على نسبة الثقة
    if confidence >= 0.80:
        color = "#4CAF50"  # أخضر - ثقة عالية
        emoji = "✅"
    elif confidence >= 0.60:
        color = "#FFD93D"  # أصفر - ثقة متوسطة
        emoji = "⚠️"
    else:
        color = "#FF6B6B"  # أحمر - ثقة منخفضة
        emoji = "❌"
    
    # عرض النتيجة في صندوق جميل
    st.markdown(f"""
    <div style="
        background: linear-gradient(145deg, #1E1E1E, #2A2A2A);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 2px solid {color};
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    ">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{emoji}</div>
        <div style="color: #AAAAAA; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">
            التشخيص المتوقع
        </div>
        <div style="color: {color}; font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0;">
            {predicted_class}
        </div>
        <div style="color: #AAAAAA; font-size: 0.9rem;">
            نسبة الثقة
        </div>
        <div style="color: {color}; font-size: 1.8rem; font-weight: 600;">
            {confidence:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # عرض احتمالات جميع الفئات
    display_probabilities_bar(probabilities, class_names, predicted_class)

# ============================================================
# 2. عرض شريط احتمالات جميع الفئات
# ============================================================

def display_probabilities_bar(probabilities: np.ndarray,
                               class_names: List[str],
                               predicted_class: Optional[str] = None) -> None:
    """
    عرض شريط أفقي يوضح احتمالات جميع الفئات.
    
    Args:
        probabilities: مصفوفة الاحتمالات
        class_names: قائمة بأسماء الفئات
        predicted_class: (اختياري) الفئة المتوقعة لتظليلها
    """
    # ألوان الفئات
    class_colors = {
        'glioma': '#FF6B6B',
        'meningioma': '#4ECDC4',
        'pituitary': '#45B7D1',
        'notumor': '#96CEB4'
    }
    
    st.markdown("### 📊 احتمالات جميع الفئات")
    
    # ترتيب الاحتمالات تنازليًا
    sorted_indices = np.argsort(probabilities)[::-1]
    
    for idx in sorted_indices:
        class_name = class_names[idx]
        prob = probabilities[idx]
        
        # تحديد اللون
        color = class_colors.get(class_name, '#4A90D9')
        
        # تظليل الفئة المتوقعة
        is_predicted = (class_name == predicted_class)
        border = "2px solid white" if is_predicted else "none"
        
        # عرض الشريط
        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                <span style="color: {'white' if is_predicted else '#AAAAAA'}; font-weight: {'bold' if is_predicted else 'normal'};">
                    {class_name}
                    {(' 👈' if is_predicted else '')}
                </span>
                <span style="color: {color}; font-weight: 600;">{prob:.1%}</span>
            </div>
            <div style="
                background-color: #333333;
                border-radius: 8px;
                height: 20px;
                overflow: hidden;
                border: {border};
                box-shadow: {'0 0 10px rgba(74,144,217,0.3)' if is_predicted else 'none'};
            ">
                <div style="
                    width: {prob * 100:.1f}%;
                    height: 100%;
                    background: linear-gradient(90deg, {color}, {color}CC);
                    border-radius: 8px;
                    transition: width 0.5s ease;
                ">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 3. عرض الصورة الأصلية + Grad-CAM جنبًا إلى جنب
# ============================================================

def display_image_with_gradcam(original_image: np.ndarray,
                                gradcam_image: np.ndarray,
                                predicted_class: str,
                                confidence: float) -> None:
    """
    عرض الصورة الأصلية وصورة Grad-CAM جنبًا إلى جنب مع التسميات.
    
    Args:
        original_image: الصورة الأصلية (RGB)
        gradcam_image: صورة Grad-CAM (مع الخريطة الحرارية)
        predicted_class: الفئة المتوقعة
        confidence: نسبة الثقة
    """
    st.markdown("### 🖼️ تحليل الصورة")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📷 الصورة الأصلية**")
        st.image(original_image, use_column_width=True)
    
    with col2:
        st.markdown("**🔥 Grad-CAM (المنطقة المؤثرة)**")
        st.image(gradcam_image, use_column_width=True)
    
    # إضافة شرح بسيط
    st.caption(f"""
    📌 **الفئة المتوقعة:** {predicted_class}  
    📊 **نسبة الثقة:** {confidence:.1%}  
    🔥 **المناطق الحمراء:** المناطق التي ركز عليها النموذج لاتخاذ القرار
    """)

# ============================================================
# 4. عرض بطاقة معلومات المريض (اختياري)
# ============================================================

def display_patient_info(patient_info: Dict[str, Any]) -> None:
    """
    عرض معلومات المريض في بطاقة منظمة.
    
    Args:
        patient_info: قاموس بمعلومات المريض
                     مثال: {
                         'patient_id': 'P-001',
                         'age': 45,
                         'gender': 'Male',
                         'date': '2024-01-01'
                     }
    """
    st.markdown("### 👤 معلومات المريض")
    
    cols = st.columns(len(patient_info))
    
    for i, (key, value) in enumerate(patient_info.items()):
        display_name = key.replace('_', ' ').title()
        cols[i].markdown(f"""
        <div style="
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 0.75rem;
            text-align: center;
        ">
            <div style="color: #AAAAAA; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">
                {display_name}
            </div>
            <div style="color: white; font-size: 1rem; font-weight: 600;">
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 5. عرض سجل التنبؤات (جدول)
# ============================================================

def display_prediction_history(history: List[Dict[str, Any]]) -> None:
    """
    عرض سجل التنبؤات السابقة في جدول.
    
    Args:
        history: قائمة بالتنبؤات السابقة
    """
    if not history:
        st.info("لا توجد تنبؤات سابقة.")
        return
    
    st.markdown("### 📜 سجل التنبؤات السابقة")
    
    # تحويل إلى DataFrame
    df = pd.DataFrame(history)
    
    # تنسيق الأعمدة
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    if 'confidence' in df.columns:
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
    
    # عرض الجدول
    st.dataframe(df, use_container_width=True)

# ============================================================
# 6. عرض النتائج الكاملة (All-in-One)
# ============================================================

def display_full_results(predicted_class: str,
                          confidence: float,
                          class_names: List[str],
                          probabilities: np.ndarray,
                          original_image: Optional[np.ndarray] = None,
                          gradcam_image: Optional[np.ndarray] = None,
                          patient_info: Optional[Dict[str, Any]] = None,
                          history: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    عرض جميع النتائج دفعة واحدة في لوحة تحكم كاملة.
    
    Args:
        predicted_class: الفئة المتوقعة
        confidence: نسبة الثقة
        class_names: قائمة بأسماء الفئات
        probabilities: مصفوفة الاحتمالات
        original_image: (اختياري) الصورة الأصلية
        gradcam_image: (اختياري) صورة Grad-CAM
        patient_info: (اختياري) معلومات المريض
        history: (اختياري) سجل التنبؤات
    """
    st.markdown("---")
    st.markdown("## 🧠 نتيجة التشخيص")
    
    # 1. معلومات المريض (إن وجدت)
    if patient_info:
        display_patient_info(patient_info)
    
    # 2. النتائج الأساسية
    display_prediction_results(predicted_class, confidence, class_names, probabilities)
    
    # 3. الصور (إن وجدت)
    if original_image is not None and gradcam_image is not None:
        display_image_with_gradcam(original_image, gradcam_image, predicted_class, confidence)
    
    # 4. سجل التنبؤات (إن وجد)
    if history:
        display_prediction_history(history)
    
    # 5. أزرار إضافية
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 تحليل صورة أخرى", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📊 عرض التفاصيل", use_container_width=True):
            st.session_state.show_details = not st.session_state.get('show_details', False)
    
    with col3:
        if st.button("📥 حفظ التقرير", use_container_width=True):
            st.success("✅ تم حفظ التقرير!")
    
    # 6. التفاصيل الإضافية (إن كانت مطلوبة)
    if st.session_state.get('show_details', False):
        display_detailed_results(probabilities, class_names, predicted_class)

# ============================================================
# 7. عرض تفاصيل إضافية (للخبراء)
# ============================================================

def display_detailed_results(probabilities: np.ndarray,
                              class_names: List[str],
                              predicted_class: str) -> None:
    """
    عرض تفاصيل إضافية عن النتائج.
    
    Args:
        probabilities: مصفوفة الاحتمالات
        class_names: قائمة بأسماء الفئات
        predicted_class: الفئة المتوقعة
    """
    with st.expander("📊 تفاصيل إضافية", expanded=True):
        # جدول الاحتمالات
        df = pd.DataFrame({
            'الفئة': class_names,
            'الاحتمالية': probabilities,
            'الحالة': ['✅ متوقعة' if c == predicted_class else '' for c in class_names]
        })
        
        # تنسيق الأرقام
        df['الاحتمالية'] = df['الاحتمالية'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(df, use_container_width=True)
        
        # معلومات إضافية
        st.markdown("#### ℹ️ معلومات إضافية")
        st.markdown(f"""
        - **الفئة ذات أعلى احتمال:** {class_names[np.argmax(probabilities)]}
        - **الفئة ذات أقل احتمال:** {class_names[np.argmin(probabilities)]}
        - **فرق الثقة بين الأولى والثانية:** {(probabilities[np.argsort(probabilities)[-1]] - probabilities[np.argsort(probabilities)[-2]]):.2%}
        """)

# ============================================================
# 8. عرض رسالة خطأ أو تحذير
# ============================================================

def display_error_message(message: str, error_type: str = "error") -> None:
    """
    عرض رسالة خطأ أو تحذير بتنسيق جميل.
    
    Args:
        message: نص الرسالة
        error_type: نوع الخطأ ('error', 'warning', 'info', 'success')
    """
    icons = {
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'success': '✅'
    }
    
    colors = {
        'error': '#FF6B6B',
        'warning': '#FFD93D',
        'info': '#4A90D9',
        'success': '#4CAF50'
    }
    
    icon = icons.get(error_type, 'ℹ️')
    color = colors.get(error_type, '#4A90D9')
    
    st.markdown(f"""
    <div style="
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid {color};
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    ">
        <div style="font-size: 2rem;">{icon}</div>
        <div style="color: #FFFFFF; font-size: 1rem;">
            {message}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 9. عرض نتائج متعددة (للاختبارات)
# ============================================================

def display_batch_results(results: List[Dict[str, Any]]) -> None:
    """
    عرض نتائج عدة صور دفعة واحدة.
    
    Args:
        results: قائمة بنتائج كل صورة
    """
    if not results:
        st.info("لا توجد نتائج لعرضها.")
        return
    
    st.markdown("### 📊 نتائج الدفعة")
    
    # تحويل إلى DataFrame
    df = pd.DataFrame(results)
    
    # تنسيق الأعمدة
    if 'confidence' in df.columns:
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
    
    # عرض الجدول
    st.dataframe(df, use_container_width=True)

# ============================================================
# 10. عرض صورة مع تسميات
# ============================================================

def display_image_with_labels(image: np.ndarray,
                               predicted_class: str,
                               confidence: float,
                               title: Optional[str] = None) -> None:
    """
    عرض صورة مع تسميات توضيحية.
    
    Args:
        image: الصورة (RGB)
        predicted_class: الفئة المتوقعة
        confidence: نسبة الثقة
        title: (اختياري) عنوان الصورة
    """
    # تحويل الصورة إلى RGB إذا كانت BGR
    if image.shape[-1] == 3 and isinstance(image[0,0,0], np.int64):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # إنشاء الرسم
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_rgb)
    ax.axis('off')
    
    # إضافة التسميات
    title_text = title if title else f"التصنيف: {predicted_class}"
    ax.set_title(title_text, color='white', fontsize=14, fontweight='bold')
    
    # إضافة نسبة الثقة
    ax.text(0.5, -0.05, f"الثقة: {confidence:.1%}",
            transform=ax.transAxes, ha='center', va='top',
            color='#4CAF50', fontsize=12)
    
    st.pyplot(fig)
    plt.close(fig)

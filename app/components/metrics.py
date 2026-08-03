# ============================================================
# app/components/metrics.py
# مكون عرض المقاييس - عرض إحصائيات الأداء في واجهة Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. عرض المقاييس الأساسية في صف واحد
# ============================================================

def display_metrics_row(metrics: Dict[str, float]) -> None:
    """
    عرض المقاييس في صف واحد (4 أعمدة).
    
    Args:
        metrics: قاموس يحتوي على المقياس وقيمته
                 مثال: {'accuracy': 0.95, 'precision': 0.94, 'recall': 0.93, 'f1_score': 0.94}
    """
    cols = st.columns(len(metrics))
    
    # ألوان وإيموجي لكل مقياس
    metric_icons = {
        'accuracy': ('🎯', '#4A90D9'),
        'precision': ('🎯', '#4CAF50'),
        'recall': ('📊', '#FFD93D'),
        'f1_score': ('⚖️', '#9B59B6'),
        'loss': ('📉', '#FF6B6B'),
        'val_loss': ('📉', '#E67E22'),
        'val_accuracy': ('🎯', '#1ABC9C')
    }
    
    for i, (metric_name, value) in enumerate(metrics.items()):
        icon, color = metric_icons.get(metric_name, ('📌', '#AAAAAA'))
        display_metric_card(cols[i], metric_name, value, icon, color)

# ============================================================
# 2. عرض بطاقة مقياس فردية
# ============================================================

def display_metric_card(column, metric_name: str, value: float, icon: str = '📌', color: str = '#4A90D9') -> None:
    """
    عرض بطاقة مقياس فردية في عمود Streamlit.
    
    Args:
        column: عمود Streamlit (st.columns)
        metric_name: اسم المقياس
        value: قيمة المقياس
        icon: إيموجي المقياس
        color: لون المقياس
    """
    # تنسيق اسم المقياس للعرض
    display_name = metric_name.replace('_', ' ').title()
    
    # تنسيق القيمة
    if isinstance(value, float):
        if abs(value) < 1:
            formatted_value = f"{value:.2%}"
        else:
            formatted_value = f"{value:.2f}"
    else:
        formatted_value = str(value)
    
    # عرض البطاقة باستخدام HTML
    column.markdown(f"""
    <div style="
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 1rem 0.5rem;
        text-align: center;
        border-left: 4px solid {color};
        margin: 0.25rem 0;
    ">
        <div style="font-size: 2rem;">{icon}</div>
        <div style="color: #AAAAAA; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;">
            {display_name}
        </div>
        <div style="color: {color}; font-size: 1.8rem; font-weight: 700;">
            {formatted_value}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 3. عرض تقرير التصنيف كامل (جدول)
# ============================================================

def display_classification_report(y_true: np.ndarray, 
                                   y_pred: np.ndarray, 
                                   class_names: List[str]) -> None:
    """
    عرض تقرير التصنيف في شكل جدول جميل.
    
    Args:
        y_true: القيم الحقيقية
        y_pred: القيم المتوقعة
        class_names: قائمة بأسماء الفئات
    """
    # حساب التقرير
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # تحويل إلى DataFrame للتنسيق
    df = pd.DataFrame(report).transpose()
    
    # تنسيق الأرقام
    for col in df.columns:
        if col not in ['support', '']:
            df[col] = df[col].apply(lambda x: f"{x:.2%}" if isinstance(x, float) else x)
    
    # عرض الجدول
    st.subheader("📊 تقرير التصنيف")
    st.dataframe(df, use_container_width=True)

# ============================================================
# 4. عرض مصفوفة الارتباك (بتنسيق جميل)
# ============================================================

def display_confusion_matrix(y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              class_names: List[str]) -> None:
    """
    عرض مصفوفة الارتباك بتنسيق Streamlit.
    
    Args:
        y_true: القيم الحقيقية
        y_pred: القيم المتوقعة
        class_names: قائمة بأسماء الفئات
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # تحويل إلى DataFrame
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # عرض البيانات
    st.subheader("📋 مصفوفة الارتباك")
    
    # استخدام المكون المخصص للعرض
    st.dataframe(df_cm.style.background_gradient(cmap='Blues'), use_container_width=True)

# ============================================================
# 5. عرض ملخص الأداء (بطاقات + جدول)
# ============================================================

def display_performance_summary(y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 class_names: List[str]) -> None:
    """
    عرض ملخص شامل لأداء النموذج.
    
    Args:
        y_true: القيم الحقيقية
        y_pred: القيم المتوقعة
        class_names: قائمة بأسماء الفئات
    """
    # حساب المقاييس الأساسية
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    # عرض البطاقات
    st.subheader("📈 ملخص الأداء")
    display_metrics_row(metrics)
    
    # عرض مصفوفة الارتباك
    display_confusion_matrix(y_true, y_pred, class_names)
    
    # عرض تقرير التصنيف
    display_classification_report(y_true, y_pred, class_names)

# ============================================================
# 6. عرض مقارنة بين نموذجين (بطاقات)
# ============================================================

def display_model_comparison(models_results: Dict[str, Dict[str, float]]) -> None:
    """
    عرض مقارنة بين نموذجين أو أكثر.
    
    Args:
        models_results: قاموس باسم النموذج وقاموس المقاييس
                        مثال: {
                            'MobileNetV2': {'accuracy': 0.95, 'precision': 0.94},
                            'ResNet50': {'accuracy': 0.93, 'precision': 0.92}
                        }
    """
    st.subheader("⚖️ مقارنة النماذج")
    
    # تحويل البيانات إلى DataFrame
    df = pd.DataFrame(models_results).transpose()
    
    # تنسيق الأرقام
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].apply(lambda x: f"{x:.2%}")
    
    # عرض الجدول
    st.dataframe(df, use_container_width=True)

# ============================================================
# 7. عرض سجل التنبؤات الأخيرة
# ============================================================

def display_prediction_log(predictions: List[Dict[str, Any]]) -> None:
    """
    عرض آخر التنبؤات التي تمت.
    
    Args:
        predictions: قائمة بالتنبؤات
                    مثال: [
                        {
                            'image_name': 'image1.jpg',
                            'predicted_class': 'glioma',
                            'confidence': 0.95,
                            'timestamp': '2024-01-01 12:00:00'
                        }
                    ]
    """
    if not predictions:
        st.info("لا توجد تنبؤات مسجلة حتى الآن.")
        return
    
    # تحويل إلى DataFrame
    df = pd.DataFrame(predictions)
    
    # تنسيق الأعمدة
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    if 'confidence' in df.columns:
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
    
    # عرض البيانات
    st.subheader("📜 سجل التنبؤات الأخيرة")
    st.dataframe(df, use_container_width=True)

# ============================================================
# 8. عرض مقاييس متقدمة (للخبراء)
# ============================================================

def display_advanced_metrics(y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              class_names: List[str]) -> None:
    """
    عرض مقاييس متقدمة مثل Sensitivity, Specificity, AUC.
    
    Args:
        y_true: القيم الحقيقية
        y_pred: القيم المتوقعة
        class_names: قائمة بأسماء الفئات
    """
    from sklearn.metrics import roc_auc_score, confusion_matrix
    
    st.subheader("🧠 مقاييس متقدمة")
    
    # حساب المقاييس لكل فئة
    metrics_data = []
    n_classes = len(class_names)
    
    # مصفوفة الارتباك
    cm = confusion_matrix(y_true, y_pred)
    
    for i, class_name in enumerate(class_names):
        # True Positives, False Positives, True Negatives, False Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # حساب المقاييس
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics_data.append({
            'Class': class_name,
            'Sensitivity (Recall)': f"{sensitivity:.2%}",
            'Specificity': f"{specificity:.2%}",
            'Precision': f"{precision:.2%}",
            'F1-Score': f"{f1:.2%}",
            'Support': int(tp + fn)
        })
    
    # عرض الجدول
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True)

# ============================================================
# 9. عرض حالة النموذج (معلومات عامة)
# ============================================================

def display_model_info(model_name: str, 
                        input_shape: Tuple[int, ...], 
                        num_params: int,
                        num_classes: int) -> None:
    """
    عرض معلومات عامة عن النموذج.
    
    Args:
        model_name: اسم النموذج
        input_shape: شكل المدخلات
        num_params: عدد المعلمات
        num_classes: عدد الفئات
    """
    cols = st.columns(4)
    
    info_items = [
        (cols[0], '📌 اسم النموذج', model_name),
        (cols[1], '📐 شكل الإدخال', str(input_shape)),
        (cols[2], '🔢 عدد المعلمات', f"{num_params:,}"),
        (cols[3], '📂 عدد الفئات', str(num_classes))
    ]
    
    for col, icon, label, value in info_items:
        col.markdown(f"""
        <div style="
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 0.75rem;
            text-align: center;
        ">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div style="color: #AAAAAA; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">
                {label}
            </div>
            <div style="color: white; font-size: 0.9rem; font-weight: 600;">
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 10. عرض جميع المقاييس دفعة واحدة (Dashboard)
# ============================================================

def display_metrics_dashboard(y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               class_names: List[str],
                               history: Optional[Dict] = None,
                               model_info: Optional[Dict] = None) -> None:
    """
    عرض لوحة تحكم كاملة بجميع المقاييس.
    
    Args:
        y_true: القيم الحقيقية
        y_pred: القيم المتوقعة
        class_names: قائمة بأسماء الفئات
        history: (اختياري) سجل التدريب
        model_info: (اختياري) معلومات النموذج
    """
    # 1. معلومات النموذج
    if model_info:
        display_model_info(
            model_info.get('name', 'Unknown'),
            model_info.get('input_shape', (224, 224, 3)),
            model_info.get('num_params', 0),
            model_info.get('num_classes', len(class_names))
        )
    
    # 2. المقاييس الأساسية
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    display_metrics_row(metrics)
    
    # 3. مصفوفة الارتباك وتقرير التصنيف
    col1, col2 = st.columns(2)
    
    with col1:
        display_confusion_matrix(y_true, y_pred, class_names)
    
    with col2:
        display_classification_report(y_true, y_pred, class_names)
    
    # 4. مقاييس متقدمة
    display_advanced_metrics(y_true, y_pred, class_names)
    
    # 5. منحنيات التدريب (إذا كانت متاحة)
    if history:
        from app.components.charts import plot_training_curves
        plot_training_curves(history)

# ============================================================
# app/pages/2_Training_Analysis.py
# صفحة تحليل التدريب - عرض منحنيات التدريب ومصفوفة الارتباك
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# ============================================================
# 1. إعدادات الصفحة
# ============================================================

st.set_page_config(
    page_title="📈 تحليل التدريب - Brain Tumor MRI Classifier",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# 2. بيانات التدريب الوهمية
# ============================================================

def get_default_training_history() -> Dict[str, List[float]]:
    """
    الحصول على بيانات تدريب افتراضية.
    
    Returns:
        قاموس يحتوي على سجل التدريب
    """
    epochs = 25
    
    # توليد بيانات محاكاة
    np.random.seed(42)
    
    # الدقة - تزيد مع الوقت مع بعض التذبذب
    accuracy = 0.65 + np.cumsum(np.random.normal(0.015, 0.02, epochs))
    accuracy = np.clip(accuracy, 0.65, 0.98)
    
    val_accuracy = accuracy + np.random.normal(0.01, 0.015, epochs)
    val_accuracy = np.clip(val_accuracy, 0.60, 0.97)
    
    # الخسارة - تنخفض مع الوقت
    loss = 1.2 - np.cumsum(np.random.normal(0.025, 0.02, epochs))
    loss = np.clip(loss, 0.05, 1.2)
    
    val_loss = loss + np.random.normal(0.02, 0.015, epochs)
    val_loss = np.clip(val_loss, 0.08, 1.25)
    
    # الـ Learning Rate - ينخفض تدريجيًا
    lr = 1e-4 * (0.8 ** np.arange(epochs))
    
    return {
        'accuracy': list(accuracy),
        'val_accuracy': list(val_accuracy),
        'loss': list(loss),
        'val_loss': list(val_loss),
        'lr': list(lr),
        'epochs': list(range(1, epochs + 1))
    }

# ============================================================
# 3. عرض منحنيات التدريب (Accuracy & Loss)
# ============================================================

def display_training_curves(history: Dict[str, List[float]]) -> None:
    """
    عرض منحنيات التدريب (الدقة والخسارة).
    
    Args:
        history: قاموس يحتوي على سجل التدريب
    """
    st.markdown("## 📈 منحنيات التدريب")
    
    # استخدام Plotly للرسوم البيانية التفاعلية
    fig = make_subplots(rows=1, cols=2, 
                         subplot_titles=('الدقة', 'الخسارة'))
    
    # 1. منحنى الدقة
    epochs = history.get('epochs', list(range(1, len(history.get('accuracy', [])) + 1)))
    
    if 'accuracy' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['accuracy'],
                       mode='lines+markers',
                       name='Training',
                       line=dict(color='#4A90D9', width=2),
                       marker=dict(size=6)),
            row=1, col=1
        )
    
    if 'val_accuracy' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_accuracy'],
                       mode='lines+markers',
                       name='Validation',
                       line=dict(color='#4CAF50', width=2, dash='dash'),
                       marker=dict(size=6)),
            row=1, col=1
        )
    
    # 2. منحنى الخسارة
    if 'loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['loss'],
                       mode='lines+markers',
                       name='Training',
                       line=dict(color='#FF6B6B', width=2),
                       marker=dict(size=6),
                       showlegend=False),
            row=1, col=2
        )
    
    if 'val_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'],
                       mode='lines+markers',
                       name='Validation',
                       line=dict(color='#FFD93D', width=2, dash='dash'),
                       marker=dict(size=6),
                       showlegend=False),
            row=1, col=2
        )
    
    # تنسيق الرسم
    fig.update_layout(
        height=450,
        showlegend=True,
        template='plotly_dark',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    fig.update_yaxes(title_text='الدقة', row=1, col=1)
    fig.update_yaxes(title_text='الخسارة', row=1, col=2)
    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 4. عرض منحنى Learning Rate
# ============================================================

def display_learning_rate_curve(history: Dict[str, List[float]]) -> None:
    """
    عرض منحنى Learning Rate.
    
    Args:
        history: قاموس يحتوي على سجل التدريب
    """
    if 'lr' not in history:
        return
    
    st.markdown("### 📉 Learning Rate")
    
    epochs = history.get('epochs', list(range(1, len(history['lr']) + 1)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['lr'],
        mode='lines+markers',
        name='Learning Rate',
        line=dict(color='#9B59B6', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        height=300,
        template='plotly_dark',
        xaxis_title='Epoch',
        yaxis_title='Learning Rate',
        yaxis_type='log'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 5. عرض مصفوفة الارتباك
# ============================================================

def display_confusion_matrix(confusion_matrix: Optional[np.ndarray] = None,
                               class_names: List[str] = None) -> None:
    """
    عرض مصفوفة الارتباك.
    
    Args:
        confusion_matrix: مصفوفة الارتباك
        class_names: قائمة بأسماء الفئات
    """
    st.markdown("## 📊 مصفوفة الارتباك")
    
    # بيانات افتراضية إذا لم يتم توفيرها
    if confusion_matrix is None:
        confusion_matrix = np.array([
            [135, 5, 3, 2],
            [4, 130, 6, 1],
            [3, 4, 140, 3],
            [2, 1, 4, 138]
        ])
    
    if class_names is None:
        class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
    
    # رسم مصفوفة الارتباك
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar_kws={'label': 'عدد الصور'})
    
    ax.set_xlabel('الفئة المتوقعة', fontsize=11)
    ax.set_ylabel('الفئة الحقيقية', fontsize=11)
    ax.set_title('مصفوفة الارتباك', fontsize=13)
    
    # تنسيق الألوان
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    st.pyplot(fig)
    plt.close(fig)
    
    # عرض المقاييس المستخلصة من المصفوفة
    display_confusion_matrix_metrics(confusion_matrix, class_names)

# ============================================================
# 6. عرض المقاييس المستخلصة من مصفوفة الارتباك
# ============================================================

def display_confusion_matrix_metrics(cm: np.ndarray, class_names: List[str]) -> None:
    """
    عرض المقاييس المستخلصة من مصفوفة الارتباك.
    
    Args:
        cm: مصفوفة الارتباك
        class_names: قائمة بأسماء الفئات
    """
    st.markdown("### 📊 مقاييس لكل فئة")
    
    metrics_data = []
    
    for i, class_name in enumerate(class_names):
        # True Positives, False Positives, True Negatives, False Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # حساب المقاييس
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / cm.sum()
        
        metrics_data.append({
            'الفئة': class_name,
            'الدقة (Precision)': f"{precision:.2%}",
            'الاستدعاء (Recall)': f"{recall:.2%}",
            'F1-Score': f"{f1:.2%}",
            'الدقة الكلية': f"{accuracy:.2%}",
            'الدعم (Support)': int(tp + fn)
        })
    
    # عرض الجدول
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True)

# ============================================================
# 7. عرض المقاييس العامة
# ============================================================

def display_general_metrics(history: Dict[str, List[float]]) -> None:
    """
    عرض المقاييس العامة للتدريب.
    
    Args:
        history: قاموس يحتوي على سجل التدريب
    """
    st.markdown("## 🎯 المقاييس العامة")
    
    # استخراج أفضل القيم
    best_accuracy = max(history.get('val_accuracy', [0]))
    best_loss = min(history.get('val_loss', [float('inf')]))
    final_accuracy = history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0
    final_loss = history.get('val_loss', [0])[-1] if history.get('val_loss') else 0
    
    # عرض البطاقات
    cols = st.columns(4)
    
    metrics = [
        (cols[0], '🏆 أفضل دقة تحقق', f"{best_accuracy:.2%}", '#4CAF50'),
        (cols[1], '📉 أفضل خسارة', f"{best_loss:.4f}", '#FF6B6B'),
        (cols[2], '🎯 الدقة النهائية', f"{final_accuracy:.2%}", '#4A90D9'),
        (cols[3], '📊 الخسارة النهائية', f"{final_loss:.4f}", '#FFD93D')
    ]
    
    for col, label, value, color in metrics:
        col.markdown(f"""
        <div style="
            background-color: #1E1E1E;
            border-radius: 12px;
            padding: 1rem 0.5rem;
            text-align: center;
            border-top: 3px solid {color};
        ">
            <div style="color: #AAAAAA; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;">
                {label}
            </div>
            <div style="color: {color}; font-size: 1.8rem; font-weight: 700;">
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # عرض معلومات إضافية
    st.markdown("---")
    st.markdown("### 📋 معلومات إضافية")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📊 عدد الحلقات (Epochs)", len(history.get('epochs', [])))
        st.metric("⏱️ وقت التدريب (تقديري)", "~45 دقيقة")
    
    with col2:
        st.metric("📈 نوع النموذج", "MobileNetV2")
        st.metric("📦 حجم النموذج", "14.2 MB")

# ============================================================
# 8. عرض ملفات التدريب المحفوظة
# ============================================================

def display_saved_training_files() -> None:
    """
    عرض ملفات التدريب المحفوظة.
    """
    st.markdown("## 📂 ملفات التدريب المحفوظة")
    
    # مسار ملفات التدريب
    reports_dir = "reports"
    
    if os.path.exists(reports_dir):
        files = os.listdir(reports_dir)
        
        if files:
            # تصفية الملفات ذات الصلة
            training_files = [f for f in files if 'training' in f.lower() or 'history' in f.lower()]
            
            if training_files:
                for file in training_files:
                    file_path = os.path.join(reports_dir, file)
                    file_size = os.path.getsize(file_path)
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"📄 {file}")
                    with col2:
                        st.write(f"{file_size / 1024:.1f} KB")
                    with col3:
                        if st.button(f"📥 تحميل", key=file):
                            with open(file_path, 'rb') as f:
                                st.download_button(
                                    label="تحميل",
                                    data=f,
                                    file_name=file
                                )
            else:
                st.info("لا توجد ملفات تدريب محفوظة.")
        else:
            st.info("مجلد التقارير فارغ.")
    else:
        st.warning("⚠️ مجلد التقارير غير موجود.")

# ============================================================
# 9. تحميل تاريخ التدريب من ملف
# ============================================================

def load_training_history(file_path: str) -> Optional[Dict[str, List[float]]]:
    """
    تحميل تاريخ التدريب من ملف JSON.
    
    Args:
        file_path: مسار الملف
        
    Returns:
        قاموس سجل التدريب أو None في حالة الخطأ
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الملف: {e}")
        return None

# ============================================================
# 10. الصفحة الرئيسية لتحليل التدريب
# ============================================================

def main() -> None:
    """
    الصفحة الرئيسية لتحليل التدريب.
    """
    # العنوان
    st.markdown("""
    # 📈 تحليل التدريب
    ### عرض منحنيات التدريب ومصفوفة الارتباك وأداء النموذج
    """)
    
    st.markdown("---")
    
    # الحصول على بيانات التدريب
    history = get_default_training_history()
    
    # 1. المقاييس العامة
    display_general_metrics(history)
    
    st.markdown("---")
    
    # 2. منحنيات التدريب
    display_training_curves(history)
    
    st.markdown("---")
    
    # 3. Learning Rate
    display_learning_rate_curve(history)
    
    st.markdown("---")
    
    # 4. مصفوفة الارتباك
    display_confusion_matrix()
    
    st.markdown("---")
    
    # 5. ملفات التدريب المحفوظة
    display_saved_training_files()
    
    st.markdown("---")
    
    # 6. خيارات التحميل
    st.markdown("### 📤 تحميل تاريخ التدريب")
    
    uploaded_file = st.file_uploader(
        "اختر ملف JSON لتاريخ التدريب",
        type=['json'],
        help="ملف JSON يحتوي على سجل التدريب (accuracy, loss, val_accuracy, val_loss)"
    )
    
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            st.success("✅ تم تحميل الملف بنجاح!")
            
            if st.button("📊 عرض البيانات المحملة"):
                display_training_curves(data)
                if 'lr' in data:
                    display_learning_rate_curve(data)
        except Exception as e:
            st.error(f"❌ خطأ في قراءة الملف: {e}")

# ============================================================
# 11. تشغيل الصفحة
# ============================================================

if __name__ == "__main__":
    main()

# ============================================================
# app/components/charts.py
# مكون الرسوم البيانية - عرض المخططات في واجهة Streamlit
# ============================================================

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

# ============================================================
# 1. إعدادات التنسيق العامة
# ============================================================

# ثيم داكن للرسومات (يتناسب مع واجهة Streamlit)
plt.style.use('dark_background')

# الألوان المخصصة للمشروع
COLORS = {
    'primary': '#4A90D9',
    'secondary': '#357ABD',
    'success': '#4CAF50',
    'warning': '#FFD93D',
    'danger': '#FF6B6B',
    'purple': '#9B59B6',
    'orange': '#E67E22',
    'cyan': '#1ABC9C',
    'gray': '#95A5A6',
    'dark': '#2C3E50'
}

# ألوان الفئات (لكل ورم لون مختلف)
CLASS_COLORS = {
    'glioma': '#FF6B6B',
    'meningioma': '#4ECDC4',
    'pituitary': '#45B7D1',
    'notumor': '#96CEB4'
}

# ============================================================
# 2. شريط التقدم (نسبة الثقة)
# ============================================================

def plot_confidence_bar(confidence: float, class_name: str, width: int = 400) -> None:
    """
    رسم شريط تقدم يعرض نسبة الثقة للفئة المتوقعة.
    
    Args:
        confidence: نسبة الثقة (0-1)
        class_name: اسم الفئة المتوقعة
        width: عرض الشريط بالبكسل
    """
    # تحديد اللون بناءً على نسبة الثقة
    if confidence >= 0.80:
        color = COLORS['success']
    elif confidence >= 0.60:
        color = COLORS['warning']
    else:
        color = COLORS['danger']
    
    fig, ax = plt.subplots(figsize=(width/100, 0.5))
    
    # إخفاء المحاور
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # رسم الشريط الخلفي
    ax.barh(0, 1, height=0.3, color='#333333', alpha=0.5)
    
    # رسم الشريط الأمامي (نسبة الثقة)
    ax.barh(0, confidence, height=0.3, color=color, alpha=0.8)
    
    # إضافة النص على الشريط
    ax.text(confidence / 2, 0, f'{confidence:.1%}', 
            ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    
    # إضافة اسم الفئة
    ax.text(1.02, 0, class_name, va='center', color='white', fontsize=10)
    
    # عرض الرسم في Streamlit
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 3. عرض احتمالات جميع الفئات (Bar Chart)
# ============================================================

def plot_class_probabilities(probabilities: np.ndarray, class_names: List[str]) -> None:
    """
    رسم شريطي يوضح احتمالات جميع الفئات.
    
    Args:
        probabilities: مصفوفة الاحتمالات (حجمها = عدد الفئات)
        class_names: قائمة بأسماء الفئات
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # ترتيب البيانات تنازليًا
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    # ألوان كل فئة
    colors = [CLASS_COLORS.get(name, COLORS['primary']) for name in sorted_names]
    
    # رسم الأشرطة
    bars = ax.barh(sorted_names, sorted_probs, color=colors, alpha=0.8)
    
    # إضافة الأرقام على الأشرطة
    for bar, prob in zip(bars, sorted_probs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', va='center', fontsize=9, color='white')
    
    # تنسيق المحاور
    ax.set_xlim(0, 1)
    ax.set_xlabel('الاحتمالية', fontsize=10)
    ax.set_title('احتمالات جميع الفئات', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.tick_params(colors='#AAAAAA')
    
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 4. عرض مصفوفة الارتباك (Confusion Matrix)
# ============================================================

def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str]) -> None:
    """
    رسم مصفوفة الارتباك.
    
    Args:
        confusion_matrix: مصفوفة الارتباك (2D)
        class_names: قائمة بأسماء الفئات
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # رسم الخريطة الحرارية
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

# ============================================================
# 5. عرض منحنيات التدريب (Training Curves)
# ============================================================

def plot_training_curves(history: Dict[str, List[float]]) -> None:
    """
    رسم منحنيات التدريب (الدقة والخسارة).
    
    Args:
        history: قاموس يحتوي على سجل التدريب
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 5.1. منحنى الدقة
    ax1 = axes[0]
    ax1.plot(history.get('accuracy', []), label='Training', color=COLORS['primary'], linewidth=2)
    ax1.plot(history.get('val_accuracy', []), label='Validation', color=COLORS['success'], linewidth=2)
    ax1.set_title('دقة النموذج', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='#AAAAAA')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.title.set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#444444')
    ax1.spines['bottom'].set_color('#444444')
    
    # 5.2. منحنى الخسارة
    ax2 = axes[1]
    ax2.plot(history.get('loss', []), label='Training', color=COLORS['primary'], linewidth=2)
    ax2.plot(history.get('val_loss', []), label='Validation', color=COLORS['danger'], linewidth=2)
    ax2.set_title('خسارة النموذج', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='#AAAAAA')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.title.set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#444444')
    ax2.spines['bottom'].set_color('#444444')
    
    # عرض الرسم
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 6. عرض توزيع الفئات (باي-شارت)
# ============================================================

def plot_class_distribution_pie(class_counts: Dict[str, int]) -> None:
    """
    رسم دائري يوضح توزيع الفئات.
    
    Args:
        class_counts: قاموس باسم الفئة وعدد الصور
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())
    colors = [CLASS_COLORS.get(label, COLORS['primary']) for label in labels]
    
    # رسم المخطط الدائري
    wedges, texts, autotexts = ax.pie(sizes, 
                                       labels=labels, 
                                       colors=colors,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       explode=[0.02] * len(labels))
    
    # تنسيق النصوص
    for text in texts:
        text.set_color('white')
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    ax.set_title('توزيع الفئات في البيانات', fontsize=13, color='white')
    
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 7. عرض توزيع الفئات (شريطي)
# ============================================================

def plot_class_distribution_bar(class_counts: Dict[str, int]) -> None:
    """
    رسم شريطي يوضح توزيع الفئات.
    
    Args:
        class_counts: قاموس باسم الفئة وعدد الصور
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [CLASS_COLORS.get(label, COLORS['primary']) for label in labels]
    
    # رسم الأشرطة
    bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # إضافة الأرقام فوق الأشرطة
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', color='white', fontsize=10)
    
    ax.set_xlabel('الفئة', fontsize=11)
    ax.set_ylabel('عدد الصور', fontsize=11)
    ax.set_title('توزيع الفئات في البيانات', fontsize=13)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 8. عرض منحنيات ROC (Receiver Operating Characteristic)
# ============================================================

def plot_roc_curves(fpr: Dict[str, np.ndarray], 
                     tpr: Dict[str, np.ndarray],
                     auc: Dict[str, float]) -> None:
    """
    رسم منحنيات ROC لكل فئة.
    
    Args:
        fpr: قاموس بقيم False Positive Rate لكل فئة
        tpr: قاموس بقيم True Positive Rate لكل فئة
        auc: قاموس بقيم AUC لكل فئة
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # رسم منحنى لكل فئة
    for class_name in fpr.keys():
        ax.plot(fpr[class_name], tpr[class_name],
                label=f'{class_name} (AUC = {auc[class_name]:.3f})',
                linewidth=2)
    
    # خط المرجع (Random Classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('منحنيات ROC', fontsize=13)
    ax.legend(loc='lower right')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 9. عرض مقارنة بين نموذجين (جانبًا إلى جنب)
# ============================================================

def plot_model_comparison(model_metrics: Dict[str, Dict[str, float]]) -> None:
    """
    رسم مقارنة بين نموذجين أو أكثر.
    
    Args:
        model_metrics: قاموس باسم النموذج وقيم المقاييس
                       مثال: {
                           'MobileNetV2': {'accuracy': 0.95, 'precision': 0.94},
                           'ResNet50': {'accuracy': 0.93, 'precision': 0.92}
                       }
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # تجهيز البيانات
    models = list(model_metrics.keys())
    metrics = list(model_metrics[models[0]].keys())
    
    x = np.arange(len(models))
    width = 0.2
    
    # رسم كل مقياس بلون مختلف
    for i, metric in enumerate(metrics):
        values = [model_metrics[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    # تنسيق المحاور
    ax.set_xlabel('النموذج', fontsize=11)
    ax.set_ylabel('القيمة', fontsize=11)
    ax.set_title('مقارنة أداء النماذج', fontsize=13)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 10. عرض رسوم بيانية بتنسيق جميل (All-in-One)
# ============================================================

def display_all_charts(probabilities: np.ndarray, 
                        class_names: List[str],
                        confidence: float,
                        predicted_class: str) -> None:
    """
    عرض جميع الرسوم البيانية المهمة دفعة واحدة.
    
    Args:
        probabilities: مصفوفة الاحتمالات
        class_names: قائمة بأسماء الفئات
        confidence: نسبة الثقة
        predicted_class: الفئة المتوقعة
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 نسبة الثقة")
        plot_confidence_bar(confidence, predicted_class)
        
        st.subheader("📈 احتمالات جميع الفئات")
        plot_class_probabilities(probabilities, class_names)
    
    with col2:
        st.subheader("📋 توزيع الفئات")
        # هنا يمكن إضافة رسم توزيع الفئات إذا كانت البيانات متاحة
        st.info("يمكن إضافة رسم توزيع الفئات هنا")

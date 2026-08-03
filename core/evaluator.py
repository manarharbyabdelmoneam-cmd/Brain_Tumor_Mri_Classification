# ============================================================
# core/evaluator.py
# تقييم النموذج - حساب المقاييس وعرض النتائج
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, auc)
import tensorflow as tf
import json
import os
from datetime import datetime

# ============================================================
# 1. حساب المقاييس الأساسية
# ============================================================

def calculate_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       average: str = 'weighted') -> Dict[str, float]:
    """
    حساب المقاييس الأساسية للتصنيف.
    
    Args:
        y_true: القيم الحقيقية
        y_pred: القيم المتوقعة
        average: طريقة حساب المتوسط ('weighted', 'macro', 'micro')
        
    Returns:
        قاموس بالمقاييس
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics

# ============================================================
# 2. حساب المقاييس لكل فئة
# ============================================================

def calculate_per_class_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 class_names: List[str]) -> pd.DataFrame:
    """
    حساب المقاييس لكل فئة على حدة.
    
    Args:
        y_true: القيم الحقيقية
        y_pred: القيم المتوقعة
        class_names: قائمة بأسماء الفئات
        
    Returns:
        DataFrame بالمقاييس لكل فئة
    """
    # حساب مصفوفة الارتباك
    cm = confusion_matrix(y_true, y_pred)
    
    metrics_list = []
    
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
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_list.append({
            'الفئة': class_name,
            'الدقة (Precision)': precision,
            'الاستدعاء (Recall)': recall,
            'F1-Score': f1,
            'الخصوصية (Specificity)': specificity,
            'الدعم (Support)': int(tp + fn)
        })
    
    return pd.DataFrame(metrics_list)

# ============================================================
# 3. حساب منحنى ROC
# ============================================================

def calculate_roc_curves(y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          class_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    حساب منحنيات ROC لكل فئة.
    
    Args:
        y_true: القيم الحقيقية (one-hot)
        y_pred_proba: الاحتمالات المتوقعة
        class_names: قائمة بأسماء الفئات
        
    Returns:
        قاموس بمنحنيات ROC لكل فئة
    """
    roc_data = {}
    n_classes = len(class_names)
    
    for i, class_name in enumerate(class_names):
        # تحويل إلى binary
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = y_pred_proba[:, i]
        
        # حساب ROC
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_binary)
        roc_auc = auc(fpr, tpr)
        
        roc_data[class_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    return roc_data

# ============================================================
# 4. رسم مصفوفة الارتباك
# ============================================================

def plot_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           class_names: List[str],
                           title: str = "مصفوفة الارتباك",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    رسم مصفوفة الارتباك.
    
    Args:
        y_true: القيم الحقيقية
        y_pred: القيم المتوقعة
        class_names: قائمة بأسماء الفئات
        title: عنوان الرسم
        save_path: مسار حفظ الرسم (اختياري)
        
    Returns:
        كائن Figure
    """
    # حساب مصفوفة الارتباك
    cm = confusion_matrix(y_true, y_pred)
    
    # إنشاء الرسم
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar_kws={'label': 'عدد الصور'})
    
    ax.set_xlabel('الفئة المتوقعة', fontsize=11)
    ax.set_ylabel('الفئة الحقيقية', fontsize=11)
    ax.set_title(title, fontsize=13)
    
    # تنسيق الألوان
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    plt.tight_layout()
    
    # حفظ الرسم
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

# ============================================================
# 5. رسم منحنيات ROC
# ============================================================

def plot_roc_curves(roc_data: Dict[str, Dict[str, Any]],
                     title: str = "منحنيات ROC",
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    رسم منحنيات ROC لجميع الفئات.
    
    Args:
        roc_data: قاموس بمنحنيات ROC
        title: عنوان الرسم
        save_path: مسار حفظ الرسم (اختياري)
        
    Returns:
        كائن Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ألوان مختلفة لكل فئة
    colors = ['#4A90D9', '#4CAF50', '#FFD93D', '#FF6B6B', '#9B59B6', '#1ABC9C']
    
    # رسم منحنيات ROC
    for i, (class_name, data) in enumerate(roc_data.items()):
        color = colors[i % len(colors)]
        ax.plot(data['fpr'], data['tpr'],
                label=f'{class_name} (AUC = {data["auc"]:.3f})',
                color=color, linewidth=2)
    
    # خط المرجع
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='lower right')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    
    plt.tight_layout()
    
    # حفظ الرسم
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

# ============================================================
# 6. عرض التقرير الكامل
# ============================================================

def display_evaluation_report(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None,
                               class_names: Optional[List[str]] = None,
                               save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    عرض تقرير التقييم الكامل.
    
    Args:
        y_true: القيم الحقيقية
        y_pred: القيم المتوقعة
        y_pred_proba: الاحتمالات المتوقعة (اختياري)
        class_names: قائمة بأسماء الفئات (اختياري)
        save_dir: مجلد حفظ التقارير (اختياري)
        
    Returns:
        قاموس بجميع النتائج
    """
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(np.unique(y_true)))]
    
    # 1. المقاييس الأساسية
    metrics = calculate_metrics(y_true, y_pred)
    
    # 2. المقاييس لكل فئة
    per_class_metrics = calculate_per_class_metrics(y_true, y_pred, class_names)
    
    # 3. مصفوفة الارتباك
    cm = confusion_matrix(y_true, y_pred)
    
    # 4. تقرير التصنيف
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 5. منحنيات ROC (إذا كانت الاحتمالات متوفرة)
    roc_data = None
    if y_pred_proba is not None:
        roc_data = calculate_roc_curves(y_true, y_pred_proba, class_names)
    
    # 6. حفظ التقارير
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # حفظ مصفوفة الارتباك
        plot_confusion_matrix(y_true, y_pred, class_names, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        
        # حفظ منحنيات ROC
        if roc_data:
            plot_roc_curves(roc_data, save_path=os.path.join(save_dir, 'roc_curves.png'))
        
        # حفظ المقاييس كـ CSV
        per_class_metrics.to_csv(os.path.join(save_dir, 'per_class_metrics.csv'), index=False)
        
        # حفظ التقرير كـ JSON
        with open(os.path.join(save_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'per_class_metrics': per_class_metrics.to_dict('records'),
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
    
    return {
        'metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_data': roc_data
    }

# ============================================================
# 7. تقييم النموذج في Streamlit
# ============================================================

def display_evaluation_ui(evaluation_results: Dict[str, Any],
                           class_names: List[str]) -> None:
    """
    عرض نتائج التقييم في Streamlit.
    
    Args:
        evaluation_results: نتائج التقييم
        class_names: قائمة بأسماء الفئات
    """
    st.markdown("## 📊 نتائج التقييم")
    
    # 1. المقاييس الأساسية
    metrics = evaluation_results.get('metrics', {})
    
    if metrics:
        st.markdown("### 🎯 المقاييس الأساسية")
        cols = st.columns(4)
        
        metric_names = {
            'accuracy': 'الدقة',
            'precision': 'الاحكام',
            'recall': 'الاستدعاء',
            'f1_score': 'F1-Score'
        }
        
        colors = {
            'accuracy': '#4A90D9',
            'precision': '#4CAF50',
            'recall': '#FFD93D',
            'f1_score': '#9B59B6'
        }
        
        for i, (key, value) in enumerate(metrics.items()):
            if key in metric_names:
                cols[i].markdown(f"""
                <div style="
                    background-color: #1E1E1E;
                    border-radius: 12px;
                    padding: 1rem;
                    text-align: center;
                    border-top: 3px solid {colors.get(key, '#4A90D9')};
                ">
                    <div style="color: #AAAAAA; font-size: 0.75rem; text-transform: uppercase;">
                        {metric_names.get(key, key)}
                    </div>
                    <div style="color: white; font-size: 1.8rem; font-weight: 700;">
                        {value:.2%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # 2. مصفوفة الارتباك
    cm = evaluation_results.get('confusion_matrix')
    if cm is not None:
        st.markdown("### 📋 مصفوفة الارتباك")
        fig = plot_confusion_matrix(
            np.array([]), np.array([]), class_names
        )  # سيتم رسمها بشكل منفصل
        st.pyplot(fig)
        plt.close(fig)
    
    # 3. المقاييس لكل فئة
    per_class = evaluation_results.get('per_class_metrics')
    if per_class is not None:
        st.markdown("### 📊 المقاييس لكل فئة")
        
        # تنسيق الأرقام
        df_display = per_class.copy()
        for col in ['الدقة (Precision)', 'الاستدعاء (Recall)', 'F1-Score', 'الخصوصية (Specificity)']:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(df_display, use_container_width=True)
    
    # 4. منحنيات ROC
    roc_data = evaluation_results.get('roc_data')
    if roc_data:
        st.markdown("### 📈 منحنيات ROC")
        fig = plot_roc_curves(roc_data)
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# 8. تقييم النموذج على بيانات الاختبار
# ============================================================

def evaluate_on_test_set(model: tf.keras.Model,
                          test_generator,
                          class_names: List[str],
                          save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    تقييم النموذج على بيانات الاختبار.
    
    Args:
        model: النموذج المدرب
        test_generator: مولد بيانات الاختبار
        class_names: قائمة بأسماء الفئات
        save_dir: مجلد حفظ التقارير (اختياري)
        
    Returns:
        نتائج التقييم
    """
    # 1. الحصول على التصنيفات
    y_true = test_generator.classes
    
    # 2. التنبؤ
    y_pred_proba = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 3. التقييم
    results = display_evaluation_report(
        y_true, y_pred, y_pred_proba, class_names, save_dir
    )
    
    # 4. إضافة معلومات إضافية
    results['num_samples'] = len(y_true)
    results['num_classes'] = len(class_names)
    
    return results

# ============================================================
# 9. مقارنة بين نموذجين
# ============================================================

def compare_models(model1: tf.keras.Model,
                    model2: tf.keras.Model,
                    test_generator,
                    class_names: List[str]) -> Dict[str, Any]:
    """
    مقارنة أداء نموذجين على نفس البيانات.
    
    Args:
        model1: النموذج الأول
        model2: النموذج الثاني
        test_generator: مولد بيانات الاختبار
        class_names: قائمة بأسماء الفئات
        
    Returns:
        نتائج المقارنة
    """
    # 1. تقييم النموذج الأول
    results1 = evaluate_on_test_set(model1, test_generator, class_names)
    
    # 2. تقييم النموذج الثاني
    results2 = evaluate_on_test_set(model2, test_generator, class_names)
    
    # 3. المقارنة
    comparison = {
        'model1': results1['metrics'],
        'model2': results2['metrics']
    }
    
    # 4. حساب الفرق
    diff = {}
    for key in results1['metrics']:
        diff[key] = results2['metrics'][key] - results1['metrics'][key]
    
    comparison['difference'] = diff
    
    return comparison

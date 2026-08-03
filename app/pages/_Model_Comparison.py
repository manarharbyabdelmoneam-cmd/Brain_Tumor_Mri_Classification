# ============================================================
# app/pages/1_Model_Comparison.py
# صفحة مقارنة النماذج - مقارنة أداء نماذج مختلفة
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. إعدادات الصفحة
# ============================================================

st.set_page_config(
    page_title="📊 مقارنة النماذج - Brain Tumor MRI Classifier",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 2. البيانات الوهمية للنماذج (للمقارنة)
# ============================================================

def get_default_model_data() -> Dict[str, Dict[str, float]]:
    """
    الحصول على بيانات النماذج الافتراضية للمقارنة.
    
    Returns:
        قاموس يحتوي على بيانات النماذج
    """
    return {
        'MobileNetV2': {
            'accuracy': 0.952,
            'precision': 0.948,
            'recall': 0.944,
            'f1_score': 0.946,
            'training_time': 45,
            'model_size_mb': 14.2,
            'inference_time': 0.08
        },
        'ResNet50': {
            'accuracy': 0.961,
            'precision': 0.958,
            'recall': 0.954,
            'f1_score': 0.956,
            'training_time': 85,
            'model_size_mb': 98.5,
            'inference_time': 0.15
        },
        'EfficientNetB0': {
            'accuracy': 0.958,
            'precision': 0.955,
            'recall': 0.952,
            'f1_score': 0.953,
            'training_time': 65,
            'model_size_mb': 29.4,
            'inference_time': 0.11
        },
        'VGG16': {
            'accuracy': 0.943,
            'precision': 0.940,
            'recall': 0.938,
            'f1_score': 0.939,
            'training_time': 120,
            'model_size_mb': 528.0,
            'inference_time': 0.25
        },
        'Custom CNN': {
            'accuracy': 0.925,
            'precision': 0.922,
            'recall': 0.918,
            'f1_score': 0.920,
            'training_time': 30,
            'model_size_mb': 8.5,
            'inference_time': 0.05
        }
    }

# ============================================================
# 3. عرض مقارنة النماذج (جدول)
# ============================================================

def display_model_comparison_table(model_data: Dict[str, Dict[str, float]]) -> None:
    """
    عرض مقارنة النماذج في جدول.
    
    Args:
        model_data: قاموس بيانات النماذج
    """
    st.markdown("## 📊 مقارنة أداء النماذج")
    
    # تحويل البيانات إلى DataFrame
    df = pd.DataFrame(model_data).transpose()
    
    # تنسيق الأعمدة
    if 'accuracy' in df.columns:
        df['accuracy'] = df['accuracy'].apply(lambda x: f"{x:.2%}")
    if 'precision' in df.columns:
        df['precision'] = df['precision'].apply(lambda x: f"{x:.2%}")
    if 'recall' in df.columns:
        df['recall'] = df['recall'].apply(lambda x: f"{x:.2%}")
    if 'f1_score' in df.columns:
        df['f1_score'] = df['f1_score'].apply(lambda x: f"{x:.2%}")
    if 'training_time' in df.columns:
        df['training_time'] = df['training_time'].apply(lambda x: f"{x:.0f} دقيقة")
    if 'model_size_mb' in df.columns:
        df['model_size_mb'] = df['model_size_mb'].apply(lambda x: f"{x:.1f} MB")
    if 'inference_time' in df.columns:
        df['inference_time'] = df['inference_time'].apply(lambda x: f"{x*1000:.0f} ms")
    
    # عرض الجدول
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "accuracy": "الدقة",
            "precision": "الاحكام",
            "recall": "الاستدعاء",
            "f1_score": "F1-Score",
            "training_time": "وقت التدريب",
            "model_size_mb": "حجم النموذج",
            "inference_time": "وقت التنبؤ"
        }
    )

# ============================================================
# 4. عرض مقارنة النماذج (بار تشارت)
# ============================================================

def display_model_comparison_chart(model_data: Dict[str, Dict[str, float]], 
                                    metric: str = 'accuracy') -> None:
    """
    عرض مقارنة النماذج في رسم بياني شريطي.
    
    Args:
        model_data: قاموس بيانات النماذج
        metric: المقياس المراد عرضه ('accuracy', 'precision', 'recall', 'f1_score')
    """
    if metric not in ['accuracy', 'precision', 'recall', 'f1_score']:
        st.error(f"❌ المقياس '{metric}' غير مدعوم.")
        return
    
    # تجهيز البيانات
    models = list(model_data.keys())
    values = [model_data[model].get(metric, 0) for model in models]
    
    # ألوان مختلفة لكل نموذج
    colors = ['#4A90D9', '#4CAF50', '#FFD93D', '#FF6B6B', '#9B59B6']
    
    # إنشاء الرسم البياني
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8, edgecolor='white', linewidth=1)
    
    # إضافة الأرقام فوق الأشرطة
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2%}', ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
    
    # تنسيق المحاور
    metric_names = {
        'accuracy': 'الدقة',
        'precision': 'الاحكام',
        'recall': 'الاستدعاء',
        'f1_score': 'F1-Score'
    }
    
    ax.set_xlabel('النموذج', fontsize=12)
    ax.set_ylabel(metric_names.get(metric, metric.capitalize()), fontsize=12)
    ax.set_title(f'مقارنة النماذج حسب {metric_names.get(metric, metric.capitalize())}', fontsize=14)
    ax.set_ylim(0.8, 1.0)
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
# 5. عرض مقارنة النماذج (رادار تشارت)
# ============================================================

def display_model_radar_chart(model_data: Dict[str, Dict[str, float]]) -> None:
    """
    عرض مقارنة النماذج في رسم بياني رادار (العنكبوت).
    
    Args:
        model_data: قاموس بيانات النماذج
    """
    st.markdown("### 🕸️ مقارنة شاملة (رادار)")
    
    # تجهيز البيانات
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    models = list(model_data.keys())
    
    # اختيار أفضل 5 نماذج للعرض
    if len(models) > 5:
        models = models[:5]
    
    # إنشاء الرسم البياني
    fig = go.Figure()
    
    for model in models:
        values = [model_data[model].get(metric, 0) for metric in metrics]
        values.append(values[0])  # إغلاق الدائرة
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            name=model,
            fill='toself',
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.8, 1.0],
                tickformat='.0%'
            )
        ),
        showlegend=True,
        height=500,
        template='plotly_dark',
        title='مقارنة النماذج (رادار)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6. عرض مقارنة الحجم والسرعة
# ============================================================

def display_model_efficiency_chart(model_data: Dict[str, Dict[str, float]]) -> None:
    """
    عرض مقارنة كفاءة النماذج (الحجم مقابل الدقة).
    
    Args:
        model_data: قاموس بيانات النماذج
    """
    st.markdown("### ⚡ كفاءة النماذج (الحجم مقابل الدقة)")
    
    # تجهيز البيانات
    models = list(model_data.keys())
    accuracies = [model_data[model].get('accuracy', 0) for model in models]
    sizes = [model_data[model].get('model_size_mb', 0) for model in models]
    times = [model_data[model].get('inference_time', 0) for model in models]
    
    # إنشاء الرسم البياني
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # رسم النقاط
    scatter = ax.scatter(sizes, accuracies, s=[t * 500 for t in times], 
                         c=range(len(models)), cmap='viridis', alpha=0.7)
    
    # إضافة تسميات النماذج
    for i, model in enumerate(models):
        ax.annotate(model, (sizes[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points',
                   color='white', fontsize=9)
    
    ax.set_xlabel('حجم النموذج (MB)', fontsize=12)
    ax.set_ylabel('الدقة', fontsize=12)
    ax.set_title('كفاءة النماذج: الحجم مقابل الدقة (حجم الدائرة = وقت التنبؤ)', fontsize=14)
    ax.set_xlim(0, max(sizes) * 1.2)
    ax.set_ylim(0.88, 0.98)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    
    # إضافة شريط الألوان
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ترتيب النموذج', color='white')
    cbar.ax.yaxis.set_tick_params(colors='white')
    
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 7. عرض المقاييس التفصيلية لكل نموذج
# ============================================================

def display_model_details(model_data: Dict[str, Dict[str, float]], 
                           selected_model: str) -> None:
    """
    عرض تفاصيل نموذج معين.
    
    Args:
        model_data: قاموس بيانات النماذج
        selected_model: اسم النموذج المختار
    """
    if selected_model not in model_data:
        st.error(f"❌ النموذج '{selected_model}' غير موجود.")
        return
    
    data = model_data[selected_model]
    
    st.markdown(f"### 📋 تفاصيل النموذج: {selected_model}")
    
    # عرض المقاييس في صف
    cols = st.columns(5)
    
    metrics = [
        ('🎯 الدقة', data.get('accuracy', 0), '#4A90D9'),
        ('🎯 الاحكام', data.get('precision', 0), '#4CAF50'),
        ('📊 الاستدعاء', data.get('recall', 0), '#FFD93D'),
        ('⚖️ F1-Score', data.get('f1_score', 0), '#9B59B6'),
        ('⏱️ وقت التنبؤ', f"{data.get('inference_time', 0)*1000:.0f} ms", '#FF6B6B')
    ]
    
    for col, (icon, value, color) in zip(cols, metrics):
        col.markdown(f"""
        <div style="
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 1rem 0.5rem;
            text-align: center;
            border-top: 3px solid {color};
        ">
            <div style="color: #AAAAAA; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">
                {icon} {col}
            </div>
            <div style="color: {color}; font-size: 1.5rem; font-weight: 700;">
                {f"{value:.2%}" if isinstance(value, float) else value}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 8. الصفحة الرئيسية للمقارنة
# ============================================================

def main() -> None:
    """
    الصفحة الرئيسية لمقارنة النماذج.
    """
    # العنوان
    st.markdown("""
    # 📊 مقارنة النماذج
    ### تحليل ومقارنة أداء نماذج التعلم العميق المختلفة
    """)
    
    st.markdown("---")
    
    # الحصول على بيانات النماذج
    model_data = get_default_model_data()
    
    # عرض جدول المقارنة
    display_model_comparison_table(model_data)
    
    st.markdown("---")
    
    # اختيار المقياس للعرض
    st.markdown("### 📈 عرض المقارنة")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_metric = st.selectbox(
            "اختر المقياس",
            options=['accuracy', 'precision', 'recall', 'f1_score'],
            format_func=lambda x: {
                'accuracy': 'الدقة',
                'precision': 'الاحكام',
                'recall': 'الاستدعاء',
                'f1_score': 'F1-Score'
            }.get(x, x.capitalize())
        )
    
    with col2:
        # عرض الرسم البياني للمقياس المختار
        display_model_comparison_chart(model_data, selected_metric)
    
    st.markdown("---")
    
    # عرض الرادار
    display_model_radar_chart(model_data)
    
    st.markdown("---")
    
    # عرض كفاءة النماذج
    display_model_efficiency_chart(model_data)
    
    st.markdown("---")
    
    # عرض تفاصيل نموذج محدد
    st.markdown("### 🔍 تفاصيل نموذج محدد")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_model = st.selectbox(
            "اختر النموذج",
            options=list(model_data.keys())
        )
    
    with col2:
        display_model_details(model_data, selected_model)
    
    st.markdown("---")
    
    # التوصيات
    st.markdown("""
    ## 💡 توصيات اختيار النموذج
    
    | الاحتياج | النموذج الموصى به | السبب |
    |----------|-------------------|-------|
    | **أعلى دقة** | ResNet50 | دقة 96.1% |
    | **أسرع تنبؤ** | Custom CNN | 0.05 ms |
    | **أصغر حجم** | Custom CNN | 8.5 MB |
    | **أفضل توازن** | EfficientNetB0 | دقة 95.8%، حجم 29.4 MB |
    | **الاستخدام العام** | MobileNetV2 | توازن جيد بين الدقة والسرعة |
    """)

# ============================================================
# 9. تشغيل الصفحة
# ============================================================

if __name__ == "__main__":
    main()

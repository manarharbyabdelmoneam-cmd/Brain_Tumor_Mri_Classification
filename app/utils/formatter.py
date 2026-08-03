# ============================================================
# app/utils/formatter.py
# دوال تنسيق المخرجات - تنسيق الأرقام والنصوص والنتائج للعرض
# ============================================================

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Any
from datetime import datetime
import json
import re

# ============================================================
# 1. تنسيق الأرقام
# ============================================================

def format_number(value: Union[int, float, str], 
                   decimals: int = 2,
                   use_thousands_separator: bool = True) -> str:
    """
    تنسيق الأرقام مع فواصل الآلاف ومنازل عشرية.
    
    Args:
        value: القيمة المراد تنسيقها
        decimals: عدد المنازل العشرية
        use_thousands_separator: استخدام فواصل الآلاف
        
    Returns:
        النص المنسق
    """
    try:
        # تحويل إلى رقم
        num = float(value)
        
        # تنسيق المنازل العشرية
        if decimals == 0:
            formatted = f"{num:.0f}"
        else:
            formatted = f"{num:.{decimals}f}"
        
        # إضافة فواصل الآلاف
        if use_thousands_separator and decimals >= 0:
            parts = formatted.split('.')
            parts[0] = re.sub(r'(\d)(?=(\d{3})+(?!\d))', r'\1,', parts[0])
            formatted = '.'.join(parts) if len(parts) > 1 else parts[0]
        
        return formatted
    
    except (ValueError, TypeError):
        return str(value)

# ============================================================
# 2. تنسيق النسبة المئوية
# ============================================================

def format_percentage(value: Union[int, float], 
                       decimals: int = 1,
                       include_symbol: bool = True) -> str:
    """
    تنسيق النسبة المئوية.
    
    Args:
        value: القيمة (0-1 أو 0-100)
        decimals: عدد المنازل العشرية
        include_symbol: إدراج علامة %
        
    Returns:
        النص المنسق
    """
    try:
        num = float(value)
        
        # تحويل من 0-1 إلى 0-100 إذا لزم الأمر
        if 0 <= num <= 1:
            num *= 100
        
        formatted = f"{num:.{decimals}f}"
        
        if include_symbol:
            formatted += "%"
        
        return formatted
    
    except (ValueError, TypeError):
        return str(value)

# ============================================================
# 3. تنسيق الوقت
# ============================================================

def format_time(seconds: Union[int, float]) -> str:
    """
    تنسيق الوقت بالثواني إلى صيغة مقروءة.
    
    Args:
        seconds: عدد الثواني
        
    Returns:
        النص المنسق (مثال: "2:30:45" أو "45s")
    """
    try:
        seconds = float(seconds)
        
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"
    
    except (ValueError, TypeError):
        return str(seconds)

# ============================================================
# 4. تنسيق التاريخ والوقت
# ============================================================

def format_datetime(dt: Optional[datetime] = None,
                     format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    تنسيق التاريخ والوقت.
    
    Args:
        dt: كائن datetime (إذا كان None، يستخدم الوقت الحالي)
        format_str: صيغة التنسيق
        
    Returns:
        النص المنسق
    """
    if dt is None:
        dt = datetime.now()
    
    return dt.strftime(format_str)

# ============================================================
# 5. تنسيق حجم الملف
# ============================================================

def format_file_size(size_bytes: int) -> str:
    """
    تنسيق حجم الملف إلى صيغة مقروءة.
    
    Args:
        size_bytes: الحجم بالبايت
        
    Returns:
        النص المنسق (مثال: "2.5 MB")
    """
    try:
        size_bytes = int(size_bytes)
        
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / 1024 ** 2:.1f} MB"
        else:
            return f"{size_bytes / 1024 ** 3:.1f} GB"
    
    except (ValueError, TypeError):
        return str(size_bytes)

# ============================================================
# 6. تنسيق اسم الفئة
# ============================================================

def format_class_name(class_name: str, 
                       as_title: bool = True,
                       include_emoji: bool = True) -> str:
    """
    تنسيق اسم الفئة مع إيموجي مناسب.
    
    Args:
        class_name: اسم الفئة
        as_title: تحويل إلى عنوان (الحروف الكبيرة)
        include_emoji: إضافة إيموجي
        
    Returns:
        النص المنسق
    """
    # تحويل إلى عنوان
    if as_title:
        class_name = class_name.title()
    
    # إضافة إيموجي
    if include_emoji:
        emojis = {
            'glioma': '🧬',
            'meningioma': '🧬',
            'pituitary': '🧬',
            'notumor': '✅',
            'no_tumor': '✅',
            'no tumor': '✅'
        }
        
        emoji = emojis.get(class_name.lower(), '🧠')
        return f"{emoji} {class_name}"
    
    return class_name

# ============================================================
# 7. تنسيق نسبة الثقة
# ============================================================

def format_confidence(confidence: float, 
                       include_emoji: bool = True) -> str:
    """
    تنسيق نسبة الثقة مع إيموجي مناسب.
    
    Args:
        confidence: نسبة الثقة (0-1)
        include_emoji: إضافة إيموجي
        
    Returns:
        النص المنسق
    """
    try:
        conf = float(confidence)
        
        # تحديد الإيموجي
        if include_emoji:
            if conf >= 0.80:
                emoji = "✅"
            elif conf >= 0.60:
                emoji = "⚠️"
            else:
                emoji = "❌"
        else:
            emoji = ""
        
        # تنسيق النسبة
        percentage = format_percentage(conf, decimals=1)
        
        return f"{emoji} {percentage}" if include_emoji else percentage
    
    except (ValueError, TypeError):
        return str(confidence)

# ============================================================
# 8. تنسيق القاموس إلى نص
# ============================================================

def format_dict(data: Dict[Any, Any], 
                 indent: int = 2,
                 sort_keys: bool = False) -> str:
    """
    تنسيق القاموس إلى نص JSON منسق.
    
    Args:
        data: القاموس المراد تنسيقه
        indent: عدد المسافات للتنسيق
        sort_keys: ترتيب المفاتيح أبجديًا
        
    Returns:
        النص المنسق
    """
    try:
        return json.dumps(data, indent=indent, sort_keys=sort_keys, default=str)
    except Exception:
        return str(data)

# ============================================================
# 9. تنسيق النتائج للتقرير
# ============================================================

def format_prediction_results(predicted_class: str,
                               confidence: float,
                               probabilities: np.ndarray,
                               class_names: List[str]) -> Dict[str, Any]:
    """
    تنسيق نتائج التنبؤ في قاموس منظم.
    
    Args:
        predicted_class: الفئة المتوقعة
        confidence: نسبة الثقة
        probabilities: مصفوفة الاحتمالات
        class_names: قائمة بأسماء الفئات
        
    Returns:
        قاموس النتائج المنسقة
    """
    # تنسيق الاحتمالات
    prob_dict = {}
    for i, name in enumerate(class_names):
        prob_dict[name] = float(probabilities[i])
    
    # ترتيب الاحتمالات تنازليًا
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'predicted_class': format_class_name(predicted_class),
        'confidence': format_confidence(confidence, include_emoji=False),
        'confidence_value': float(confidence),
        'probabilities': sorted_probs,
        'timestamp': format_datetime()
    }

# ============================================================
# 10. تنسيق DataFrame للعرض
# ============================================================

def format_dataframe(df: pd.DataFrame,
                      format_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    تنسيق DataFrame للعرض.
    
    Args:
        df: DataFrame المراد تنسيقه
        format_dict: قاموس بتنسيق كل عمود
        
    Returns:
        DataFrame المنسقة
    """
    if format_dict is None:
        format_dict = {}
    
    df_copy = df.copy()
    
    for col, fmt in format_dict.items():
        if col in df_copy.columns:
            if fmt == 'percentage':
                df_copy[col] = df_copy[col].apply(lambda x: format_percentage(x))
            elif fmt == 'number':
                df_copy[col] = df_copy[col].apply(lambda x: format_number(x))
            elif fmt == 'datetime':
                df_copy[col] = df_copy[col].apply(lambda x: format_datetime(pd.to_datetime(x)))
            elif fmt == 'filesize':
                df_copy[col] = df_copy[col].apply(lambda x: format_file_size(int(x)))
    
    return df_copy

# ============================================================
# 11. تنسيق النص حسب الطول
# ============================================================

def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    اختصار النص إذا كان أطول من الحد المحدد.
    
    Args:
        text: النص المراد اختصاره
        max_length: الحد الأقصى للطول
        suffix: النص المضاف في النهاية
        
    Returns:
        النص المختصر
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# ============================================================
# 12. تنسيق الرسائل
# ============================================================

def format_message(message: str, 
                    message_type: str = 'info',
                    include_emoji: bool = True) -> str:
    """
    تنسيق الرسائل مع إيموجي مناسب.
    
    Args:
        message: نص الرسالة
        message_type: نوع الرسالة ('info', 'success', 'warning', 'error')
        include_emoji: إضافة إيموجي
        
    Returns:
        النص المنسق
    """
    emojis = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    }
    
    if include_emoji and message_type in emojis:
        return f"{emojis[message_type]} {message}"
    
    return message

# ============================================================
# 13. تنسيق التقرير النهائي
# ============================================================

def format_final_report(predictions: List[Dict[str, Any]],
                         total_time: float,
                         model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    تنسيق التقرير النهائي.
    
    Args:
        predictions: قائمة بالتنبؤات
        total_time: الوقت الإجمالي
        model_info: معلومات النموذج
        
    Returns:
        قاموس التقرير المنسق
    """
    # حساب الإحصائيات
    total = len(predictions)
    
    if total > 0:
        avg_confidence = np.mean([p.get('confidence_value', 0) for p in predictions])
        class_counts = {}
        for p in predictions:
            cls = p.get('predicted_class', 'Unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
    else:
        avg_confidence = 0
        class_counts = {}
    
    # بناء التقرير
    report = {
        'summary': {
            'total_predictions': total,
            'average_confidence': format_percentage(avg_confidence),
            'total_time': format_time(total_time),
            'timestamp': format_datetime()
        },
        'class_distribution': class_counts,
        'predictions': predictions
    }
    
    if model_info:
        report['model_info'] = {
            'name': model_info.get('name', 'Unknown'),
            'version': model_info.get('version', '1.0.0'),
            'num_classes': model_info.get('num_classes', 0)
        }
    
    return report

# ============================================================
# 14. تنسيق عنوان الصفحة
# ============================================================

def format_page_title(title: str, 
                       subtitle: Optional[str] = None) -> str:
    """
    تنسيق عنوان الصفحة.
    
    Args:
        title: العنوان الرئيسي
        subtitle: العنوان الفرعي
        
    Returns:
        النص المنسق
    """
    formatted = f"# {title}"
    
    if subtitle:
        formatted += f"\n### {subtitle}"
    
    return formatted

# ============================================================
# 15. تنسيق معلومات النموذج
# ============================================================

def format_model_info(model_info: Dict[str, Any]) -> Dict[str, str]:
    """
    تنسيق معلومات النموذج للعرض.
    
    Args:
        model_info: قاموس معلومات النموذج
        
    Returns:
        قاموس المعلومات المنسقة
    """
    formatted = {}
    
    # تنسيق كل حقل
    for key, value in model_info.items():
        if key in ['accuracy', 'precision', 'recall', 'f1_score']:
            formatted[key] = format_percentage(value)
        elif key in ['num_params', 'total_params']:
            formatted[key] = format_number(value)
        elif key in ['training_time']:
            formatted[key] = format_time(value)
        elif key in ['model_size_mb', 'size_mb']:
            formatted[key] = format_file_size(int(value * 1024 * 1024))
        else:
            formatted[key] = str(value)
    
    return formatted

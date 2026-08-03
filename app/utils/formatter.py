# ============================================================
# app/utils/formatter.py
# دوال تنسيق المخرجات
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
    try:
        num = float(value)
        formatted = f"{num:.{decimals}f}" if decimals > 0 else f"{num:.0f}"
        if use_thousands_separator:
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
    try:
        num = float(value)
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
    try:
        seconds = float(seconds)
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h}h {m}m {s}s"
    except (ValueError, TypeError):
        return str(seconds)

# ============================================================
# 4. تنسيق التاريخ والوقت
# ============================================================

def format_datetime(dt: Optional[datetime] = None,
                     format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)

# ============================================================
# 5. تنسيق حجم الملف
# ============================================================

def format_file_size(size_bytes: int) -> str:
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

def format_class_name(class_name: Any,
                       as_title: bool = True,
                       include_emoji: bool = True) -> str:
    """
    تنسيق اسم الفئة مع إيموجي مناسب.
    """
    # ✅ معالجة حالة dict أو أي نوع غير string
    if isinstance(class_name, dict):
        # لو dict، خذ أول قيمة أو المفتاح 'name'
        class_name = class_name.get('name', class_name.get('class', str(list(class_name.values())[0])))
    elif not isinstance(class_name, str):
        class_name = str(class_name)

    # تنظيف النص
    class_name = class_name.strip()

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
            'no tumor': '✅',
            'notumor': '✅',
        }
        emoji = emojis.get(class_name.lower().replace(' ', ''), '🧠')
        return f"{emoji} {class_name}"

    return class_name

# ============================================================
# 7. تنسيق نسبة الثقة
# ============================================================

def format_confidence(confidence: Any, include_emoji: bool = True) -> str:
    try:
        conf = float(confidence)
        if include_emoji:
            emoji = "✅" if conf >= 0.80 else "⚠️" if conf >= 0.60 else "❌"
        else:
            emoji = ""
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
    try:
        return json.dumps(data, indent=indent, sort_keys=sort_keys, default=str)
    except Exception:
        return str(data)

# ============================================================
# 9. تنسيق النتائج للتقرير
# ============================================================

def format_prediction_results(predicted_class: Any,
                               confidence: float,
                               probabilities: np.ndarray,
                               class_names: List[str]) -> Dict[str, Any]:
    # ✅ التأكد من أن class_names قائمة
    if isinstance(class_names, dict):
        class_names = list(class_names.values())
    else:
        class_names = list(class_names)

    prob_dict = {}
    for i, name in enumerate(class_names):
        prob_dict[str(name)] = float(probabilities[i])

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
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# ============================================================
# 12. تنسيق الرسائل
# ============================================================

def format_message(message: str,
                    message_type: str = 'info',
                    include_emoji: bool = True) -> str:
    emojis = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
    if include_emoji and message_type in emojis:
        return f"{emojis[message_type]} {message}"
    return message

# ============================================================
# 13. تنسيق التقرير النهائي
# ============================================================

def format_final_report(predictions: List[Dict[str, Any]],
                         total_time: float,
                         model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

def format_page_title(title: str, subtitle: Optional[str] = None) -> str:
    formatted = f"# {title}"
    if subtitle:
        formatted += f"\n### {subtitle}"
    return formatted

# ============================================================
# 15. تنسيق معلومات النموذج
# ============================================================

def format_model_info(model_info: Dict[str, Any]) -> Dict[str, str]:
    formatted = {}
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

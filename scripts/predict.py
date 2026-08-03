# ============================================================
# scripts/predict.py
# التنبؤ من سطر الأوامر - تشغيل التنبؤ على صورة واحدة أو مجلد
# ============================================================

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import cv2

# إضافة مسار المشروع إلى sys.path
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.model_loader import load_model
from app.utils.image_processor import read_image, preprocess_for_prediction
from app.utils.formatter import format_class_name, format_confidence

# ============================================================
# 1. إعدادات التنبؤ
# ============================================================

def get_config():
    """
    الحصول على إعدادات التنبؤ من argparse.
    
    Returns:
        قاموس الإعدادات
    """
    parser = argparse.ArgumentParser(description='التنبؤ على صور أورام المخ')
    
    # مسار النموذج والبيانات
    parser.add_argument('--model_path', type=str, default='models_saved/keras_model.h5',
                        help='مسار ملف النموذج')
    parser.add_argument('--image_path', type=str, required=True,
                        help='مسار الصورة أو مجلد الصور')
    parser.add_argument('--class_names_path', type=str, default='models_saved/metadata/class_names.json',
                        help='مسار ملف أسماء الفئات')
    
    # إعدادات المعالجة
    parser.add_argument('--target_size', type=str, default='224,224',
                        help='حجم الصورة المستهدف (width,height)')
    parser.add_argument('--batch_mode', action='store_true',
                        help='تشغيل الوضع الدفعي (معالجة مجلد)')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                        help='مجلد حفظ النتائج')
    parser.add_argument('--save_output', action='store_true',
                        help='حفظ النتائج في ملف')
    parser.add_argument('--no_console', action='store_true',
                        help='عدم عرض النتائج في الكونسول')
    
    return parser.parse_args()

# ============================================================
# 2. تحميل أسماء الفئات
# ============================================================

def load_class_names(class_names_path: str) -> list:
    """
    تحميل أسماء الفئات من ملف JSON.
    
    Args:
        class_names_path: مسار ملف أسماء الفئات
        
    Returns:
        قائمة بأسماء الفئات
    """
    try:
        with open(class_names_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'class_names' in data:
                return data['class_names']
            elif isinstance(data, list):
                return data
    except Exception as e:
        print(f"⚠️ تعذر تحميل أسماء الفئات: {e}")
    
    # أسماء الفئات الافتراضية
    return ['glioma', 'meningioma', 'pituitary', 'notumor']

# ============================================================
# 3. التنبؤ على صورة واحدة
# ============================================================

def predict_single_image(model: tf.keras.Model,
                          image_path: str,
                          class_names: list,
                          target_size: tuple) -> dict:
    """
    التنبؤ على صورة واحدة.
    
    Args:
        model: النموذج المدرب
        image_path: مسار الصورة
        class_names: قائمة بأسماء الفئات
        target_size: الأبعاد المستهدفة
        
    Returns:
        قاموس بنتائج التنبؤ
    """
    # 1. قراءة الصورة
    image = read_image(image_path)
    
    if image is None:
        raise ValueError(f"تعذر قراءة الصورة: {image_path}")
    
    # 2. معالجة الصورة
    processed = preprocess_for_prediction(image, target_size, normalize=True)
    
    # 3. التنبؤ
    predictions = model.predict(processed, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = class_names[predicted_idx]
    
    # 4. بناء النتائج
    results = {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'predicted_class': predicted_class,
        'predicted_class_idx': int(predicted_idx),
        'confidence': float(confidence),
        'probabilities': {name: float(pred) for name, pred in zip(class_names, predictions[0])}
    }
    
    return results

# ============================================================
# 4. التنبؤ على مجلد (وضع دفعة)
# ============================================================

def predict_batch(model: tf.keras.Model,
                   folder_path: str,
                   class_names: list,
                   target_size: tuple,
                   extensions: list = ['.jpg', '.jpeg', '.png']) -> list:
    """
    التنبؤ على جميع الصور في مجلد.
    
    Args:
        model: النموذج المدرب
        folder_path: مسار المجلد
        class_names: قائمة بأسماء الفئات
        target_size: الأبعاد المستهدفة
        extensions: قائمة بامتدادات الصور المسموحة
        
    Returns:
        قائمة بنتائج التنبؤ
    """
    # 1. الحصول على قائمة الصور
    images = []
    for ext in extensions:
        images.extend(Path(folder_path).glob(f'*{ext}'))
        images.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    
    if not images:
        print(f"⚠️ لم يتم العثور على صور في: {folder_path}")
        return []
    
    print(f"📊 تم العثور على {len(images)} صورة.")
    
    # 2. التنبؤ على كل صورة
    results = []
    for i, img_path in enumerate(images):
        try:
            print(f"   [{i+1}/{len(images)}] معالجة: {img_path.name}")
            
            result = predict_single_image(
                model, str(img_path), class_names, target_size
            )
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ خطأ في {img_path.name}: {e}")
            continue
    
    return results

# ============================================================
# 5. عرض النتائج
# ============================================================

def print_results(results: dict) -> None:
    """
    عرض نتائج التنبؤ في الكونسول.
    
    Args:
        results: قاموس النتائج
    """
    print("\n" + "="*60)
    print("🧠 نتائج التنبؤ")
    print("="*60)
    
    print(f"📁 الصورة: {results['image_name']}")
    print(f"📂 المسار: {results['image_path']}")
    print("-"*40)
    print(f"🏷️ الفئة المتوقعة: {format_class_name(results['predicted_class'])}")
    print(f"📊 نسبة الثقة: {format_confidence(results['confidence'])}")
    print("-"*40)
    
    # عرض الاحتمالات
    print("📊 احتمالات جميع الفئات:")
    for name, prob in results['probabilities'].items():
        bar = "█" * int(prob * 50)
        print(f"   {name:12s}: {bar} {prob:.2%}")
    
    print("="*60)

def print_batch_results(results: list) -> None:
    """
    عرض نتائج التنبؤ الدفعي في الكونسول.
    
    Args:
        results: قائمة بنتائج التنبؤ
    """
    print("\n" + "="*60)
    print("🧠 نتائج التنبؤ الدفعي")
    print("="*60)
    
    print(f"📊 عدد الصور: {len(results)}")
    print("-"*60)
    
    # عرض النتائج في جدول
    print(f"{'#':>3} | {'الصورة':<30} | {'الفئة':<15} | {'الثقة':<8}")
    print("-"*60)
    
    for i, result in enumerate(results):
        name = result['image_name'][:30]
        cls = result['predicted_class'][:15]
        conf = f"{result['confidence']:.1%}"
        print(f"{i+1:>3} | {name:<30} | {cls:<15} | {conf:<8}")
    
    print("="*60)

# ============================================================
# 6. حفظ النتائج
# ============================================================

def save_results(results: list, output_dir: str) -> None:
    """
    حفظ نتائج التنبؤ في ملف JSON.
    
    Args:
        results: قائمة بنتائج التنبؤ
        output_dir: مجلد الحفظ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'predictions_{timestamp}.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ تم حفظ النتائج في: {output_path}")

# ============================================================
# 7. التشغيل الرئيسي
# ============================================================

def main():
    """
    تشغيل عملية التنبؤ.
    """
    print("="*60)
    print("🧠 Brain Tumor MRI Classifier - Prediction")
    print("="*60)
    
    # 1. الحصول على الإعدادات
    args = get_config()
    
    # 2. تحويل target_size إلى tuple
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # 3. إنشاء مجلد المخرجات
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. تحميل أسماء الفئات
    class_names = load_class_names(args.class_names_path)
    print(f"📊 أسماء الفئات: {class_names}")
    
    # 5. تحميل النموذج
    print(f"📂 تحميل النموذج من: {args.model_path}")
    model = load_model(args.model_path)
    
    if model is None:
        print("❌ فشل تحميل النموذج.")
        sys.exit(1)
    
    print(f"✅ تم تحميل النموذج بنجاح.")
    
    # 6. التأكد من وجود الصورة/المجلد
    if not os.path.exists(args.image_path):
        print(f"❌ المسار غير موجود: {args.image_path}")
        sys.exit(1)
    
    # 7. التنبؤ
    all_results = []
    
    if args.batch_mode:
        # وضع دفعة (مجلد)
        print(f"📂 وضع الدفعة: {args.image_path}")
        all_results = predict_batch(
            model, args.image_path, class_names, target_size
        )
        
        if not args.no_console:
            print_batch_results(all_results)
    
    else:
        # وضع صورة واحدة
        print(f"🖼️ وضع الصورة الواحدة: {args.image_path}")
        try:
            result = predict_single_image(
                model, args.image_path, class_names, target_size
            )
            all_results = [result]
            
            if not args.no_console:
                print_results(result)
                
        except Exception as e:
            print(f"❌ خطأ في التنبؤ: {e}")
            sys.exit(1)
    
    # 8. حفظ النتائج
    if args.save_output and all_results:
        save_results(all_results, args.output_dir)
    
    print("\n✅ انتهى التنبؤ بنجاح!")

# ============================================================
# 8. تشغيل السكريبت
# ============================================================

if __name__ == "__main__":
    main()

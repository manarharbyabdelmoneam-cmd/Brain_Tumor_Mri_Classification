# ============================================================
# scripts/evaluate.py
# تقييم النموذج - تشغيل تقييم النموذج على بيانات الاختبار
# ============================================================

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# إضافة مسار المشروع إلى sys.path
sys.path.append(str(Path(__file__).parent.parent))

from core.evaluator import evaluate_on_test_set, display_evaluation_report
from core.data_loader import load_data_from_directory, load_data_with_augmentation
from app.utils.model_loader import load_model, load_model_with_metadata

# ============================================================
# 1. إعدادات التقييم
# ============================================================

def get_config():
    """
    الحصول على إعدادات التقييم من argparse.
    
    Returns:
        قاموس الإعدادات
    """
    parser = argparse.ArgumentParser(description='تقييم نموذج تصنيف أورام المخ')
    
    # مسارات البيانات والنموذج
    parser.add_argument('--model_path', type=str, default='models_saved/keras_model.h5',
                        help='مسار ملف النموذج')
    parser.add_argument('--data_dir', type=str, default='data/raw/Testing',
                        help='مسار بيانات الاختبار')
    parser.add_argument('--class_names_path', type=str, default='models_saved/metadata/class_names.json',
                        help='مسار ملف أسماء الفئات')
    parser.add_argument('--save_dir', type=str, default='reports/metrics',
                        help='مجلد حفظ التقارير')
    
    # إعدادات التقييم
    parser.add_argument('--batch_size', type=int, default=32,
                        help='حجم الدفعة')
    parser.add_argument('--target_size', type=str, default='224,224',
                        help='حجم الصورة المستهدف (width,height)')
    parser.add_argument('--no_gradcam', action='store_true',
                        help='تعطيل توليد Grad-CAM')
    
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
# 3. التقييم الرئيسي
# ============================================================

def main():
    """
    تشغيل عملية التقييم.
    """
    print("="*60)
    print("🧠 Brain Tumor MRI Classifier - Evaluation")
    print("="*60)
    
    # 1. الحصول على الإعدادات
    args = get_config()
    
    # 2. تحويل target_size إلى tuple
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # 3. إنشاء مجلد الحفظ
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 4. تحميل أسماء الفئات
    class_names = load_class_names(args.class_names_path)
    print(f"📊 أسماء الفئات: {class_names}")
    
    # 5. تحميل النموذج
    print(f"📂 تحميل النموذج من: {args.model_path}")
    model, metadata = load_model_with_metadata(args.model_path)
    
    if model is None:
        print("❌ فشل تحميل النموذج.")
        sys.exit(1)
    
    print(f"✅ تم تحميل النموذج بنجاح.")
    print(f"📊 عدد المعلمات: {model.count_params():,}")
    
    # 6. تحميل بيانات الاختبار
    print(f"📂 تحميل بيانات الاختبار من: {args.data_dir}")
    
    if not os.path.exists(args.data_dir):
        print(f"❌ المسار غير موجود: {args.data_dir}")
        sys.exit(1)
    
    # استخدام DataGenerator لتحميل البيانات
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        args.data_dir,
        target_size=target_size,
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ تم تحميل {test_generator.samples} صورة.")
    
    # 7. تقييم النموذج
    print("\n📊 بدء التقييم...")
    print("-"*40)
    
    results = evaluate_on_test_set(
        model=model,
        test_generator=test_generator,
        class_names=class_names,
        save_dir=args.save_dir
    )
    
    # 8. عرض النتائج
    print("\n📊 نتائج التقييم:")
    print("-"*40)
    
    metrics = results.get('metrics', {})
    print(f"   - الدقة (Accuracy): {metrics.get('accuracy', 0):.4f}")
    print(f"   - الاحكام (Precision): {metrics.get('precision', 0):.4f}")
    print(f"   - الاستدعاء (Recall): {metrics.get('recall', 0):.4f}")
    print(f"   - F1-Score: {metrics.get('f1_score', 0):.4f}")
    
    # 9. عرض المقاييس لكل فئة
    per_class = results.get('per_class_metrics')
    if per_class is not None:
        print("\n📊 المقاييس لكل فئة:")
        print("-"*40)
        
        for _, row in per_class.iterrows():
            class_name = row['الفئة']
            precision = row['الدقة (Precision)']
            recall = row['الاستدعاء (Recall)']
            f1 = row['F1-Score']
            print(f"   - {class_name}:")
            print(f"       Precision: {precision:.4f}")
            print(f"       Recall: {recall:.4f}")
            print(f"       F1-Score: {f1:.4f}")
    
    # 10. حفظ النتائج
    print("\n💾 حفظ النتائج...")
    
    # حفظ كم JSON
    results_path = os.path.join(args.save_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        # تحويل البيانات إلى JSON
        json_data = {
            'metrics': results.get('metrics', {}),
            'confusion_matrix': results.get('confusion_matrix', []).tolist() if isinstance(results.get('confusion_matrix'), np.ndarray) else results.get('confusion_matrix', []),
            'num_samples': results.get('num_samples', 0),
            'num_classes': results.get('num_classes', 0),
            'timestamp': datetime.now().isoformat()
        }
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ تم حفظ النتائج في: {results_path}")
    
    # 11. إنشاء تقرير HTML
    try:
        from scripts.generate_report import generate_html_report
        report_path = os.path.join(args.save_dir, 'performance_report.html')
        generate_html_report(results, class_names, report_path)
        print(f"✅ تم إنشاء التقرير في: {report_path}")
    except Exception as e:
        print(f"⚠️ تعذر إنشاء التقرير HTML: {e}")
    
    # 12. عرض مصفوفة الارتباك
    cm = results.get('confusion_matrix')
    if cm is not None:
        print("\n📊 مصفوفة الارتباك:")
        print("-"*40)
        print("         " + " ".join([f"{name:>10}" for name in class_names]))
        for i, row in enumerate(cm):
            print(f"{class_names[i]:>10} " + " ".join([f"{val:>10}" for val in row]))
    
    print("\n" + "="*60)
    print("✅ انتهى التقييم بنجاح!")
    print("="*60)

# ============================================================
# 4. تشغيل السكريبت
# ============================================================

if __name__ == "__main__":
    main()

# ============================================================
# scripts/train.py
# تدريب النموذج - تشغيل عملية التدريب من سطر الأوامر
# ============================================================

import os
import sys
import json
import argparse
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import numpy as np

# إضافة مسار المشروع إلى sys.path
sys.path.append(str(Path(__file__).parent.parent))

from core.trainer import train_model, compile_model, TrainingConfig
from core.data_loader import load_data_from_directory, load_data_with_augmentation
from core.preprocessor import preprocess_for_training
from core.evaluator import evaluate_on_test_set
from app.utils.model_loader import save_model, extract_model_metadata
from app.utils.formatter import format_percentage, format_time

# ============================================================
# 1. بناء النموذج
# ============================================================

def build_model(input_shape: tuple = (224, 224, 3),
                num_classes: int = 4,
                dropout_rate: float = 0.3,
                dense_units: int = 128) -> tf.keras.Model:
    """
    بناء نموذج MobileNetV2 للتصنيف.
    
    Args:
        input_shape: شكل الصورة المدخلة
        num_classes: عدد الفئات
        dropout_rate: نسبة الـ Dropout
        dense_units: عدد الخلايا في الطبقة الكثيفة
        
    Returns:
        نموذج Keras
    """
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models
    
    # 1. Base Model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # تجميد طبقات Base Model
    base_model.trainable = False
    
    # 2. إضافة طبقات مخصصة
    inputs = base_model.input
    x = base_model.output
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# ============================================================
# 2. إعدادات التدريب من argparse
# ============================================================

def get_config():
    """
    الحصول على إعدادات التدريب من argparse.
    
    Returns:
        قاموس الإعدادات
    """
    parser = argparse.ArgumentParser(description='تدريب نموذج تصنيف أورام المخ')
    
    # مسارات البيانات
    parser.add_argument('--data_dir', type=str, default='data/raw/Training',
                        help='مسار بيانات التدريب')
    parser.add_argument('--test_dir', type=str, default='data/raw/Testing',
                        help='مسار بيانات الاختبار')
    parser.add_argument('--save_dir', type=str, default='models_saved',
                        help='مجلد حفظ النموذج')
    
    # إعدادات النموذج
    parser.add_argument('--input_shape', type=str, default='224,224,3',
                        help='شكل المدخلات (height,width,channels)')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='عدد الفئات')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='نسبة الـ Dropout')
    parser.add_argument('--dense_units', type=int, default=128,
                        help='عدد الخلايا في الطبقة الكثيفة')
    
    # إعدادات التدريب
    parser.add_argument('--epochs', type=int, default=25,
                        help='عدد الحلقات')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='حجم الدفعة')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='معدل التعلم')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='نسبة بيانات التحقق')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='عدد الحلقات قبل الإيقاف المبكر')
    parser.add_argument('--reduce_lr_patience', type=int, default=3,
                        help='عدد الحلقات قبل تخفيض LR')
    parser.add_argument('--reduce_lr_factor', type=float, default=0.2,
                        help='عامل تخفيض LR')
    
    # Augmentation
    parser.add_argument('--augmentation', action='store_true',
                        help='تفعيل Data Augmentation')
    
    # Fine-tuning
    parser.add_argument('--fine_tune', action='store_true',
                        help='تفعيل Fine-tuning بعد التدريب')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                        help='عدد حلقات Fine-tuning')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5,
                        help='معدل التعلم في Fine-tuning')
    
    # أسماء الفئات
    parser.add_argument('--class_names_path', type=str, default='models_saved/metadata/class_names.json',
                        help='مسار ملف أسماء الفئات')
    
    return parser.parse_args()

# ============================================================
# 3. تحميل أسماء الفئات
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
# 4. تشغيل التدريب
# ============================================================

def main():
    """
    تشغيل عملية التدريب.
    """
    print("="*60)
    print("🧠 Brain Tumor MRI Classifier - Training")
    print("="*60)
    
    # 1. الحصول على الإعدادات
    args = get_config()
    
    # 2. تحويل input_shape إلى tuple
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # 3. إنشاء مجلد الحفظ
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'metadata'), exist_ok=True)
    
    # 4. تحميل أسماء الفئات
    class_names = load_class_names(args.class_names_path)
    print(f"📊 أسماء الفئات: {class_names}")
    
    # 5. بناء النموذج
    print("🏗️ بناء النموذج...")
    model = build_model(
        input_shape=input_shape,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        dense_units=args.dense_units
    )
    
    print(f"✅ تم بناء النموذج بنجاح.")
    print(f"📊 عدد المعلمات: {model.count_params():,}")
    
    # 6. تحميل البيانات
    print(f"📂 تحميل البيانات من: {args.data_dir}")
    
    if not os.path.exists(args.data_dir):
        print(f"❌ المسار غير موجود: {args.data_dir}")
        sys.exit(1)
    
    if args.augmentation:
        datagen, train_generator, val_generator, test_generator = load_data_with_augmentation(
            data_dir=args.data_dir,
            target_size=input_shape[:2],
            batch_size=args.batch_size,
            validation_split=args.validation_split
        )
    else:
        datagen, train_generator, val_generator, test_generator = load_data_from_directory(
            data_dir=args.data_dir,
            target_size=input_shape[:2],
            batch_size=args.batch_size,
            validation_split=args.validation_split
        )
    
    print(f"✅ تم تحميل البيانات:")
    print(f"   - التدريب: {train_generator.samples} صورة")
    print(f"   - التحقق: {val_generator.samples} صورة")
    
    # 7. إعدادات التدريب
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_factor=args.reduce_lr_factor
    )
    
    # 8. التدريب
    print("\n🚀 بدء التدريب...")
    print("-"*40)
    
    start_time = datetime.now()
    
    model, history = train_model(
        model=model,
        train_data=train_generator,
        validation_data=val_generator,
        config=config,
        model_name='brain_tumor_model',
        save_dir=args.save_dir
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"✅ انتهى التدريب في {format_time(training_time)}")
    
    # 9. تقييم النموذج على بيانات الاختبار (إذا كانت موجودة)
    if test_generator is not None and test_generator.samples > 0:
        print("\n📊 تقييم النموذج على بيانات الاختبار...")
        
        test_results = model.evaluate(test_generator, verbose=0)
        
        # عرض نتائج الاختبار
        metric_names = ['loss', 'accuracy']
        for i, value in enumerate(test_results):
            if i < len(metric_names):
                print(f"   - {metric_names[i]}: {value:.4f}")
    
    # 10. حفظ معلومات النموذج
    print("\n💾 حفظ معلومات النموذج...")
    
    metadata = extract_model_metadata(model)
    metadata['class_names'] = class_names
    metadata['training_config'] = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': args.validation_split,
        'augmentation': args.augmentation,
        'fine_tune': args.fine_tune
    }
    metadata['training_time_seconds'] = training_time
    metadata['training_date'] = datetime.now().isoformat()
    
    # حفظ معلومات التدريب
    metadata_path = os.path.join(args.save_dir, 'metadata', 'training_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"✅ تم حفظ معلومات التدريب في: {metadata_path}")
    
    # 11. حفظ أسماء الفئات
    class_names_path = os.path.join(args.save_dir, 'metadata', 'class_names.json')
    with open(class_names_path, 'w', encoding='utf-8') as f:
        json.dump({
            'class_names': class_names,
            'num_classes': len(class_names),
            'last_updated': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ تم حفظ أسماء الفئات في: {class_names_path}")
    
    # 12. Fine-tuning (إذا كان مطلوبًا)
    if args.fine_tune:
        print("\n🔧 بدء Fine-tuning...")
        print("-"*40)
        
        # فتح الطبقات العلوية
        base_model = model.layers[0]
        base_model.trainable = True
        
        # تجميد الطبقات السفلية
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # إعادة تجميع النموذج
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # إعدادات Fine-tuning
        fine_tune_config = TrainingConfig(
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            learning_rate=args.fine_tune_lr,
            validation_split=args.validation_split,
            early_stopping_patience=3,
            reduce_lr_patience=2,
            reduce_lr_factor=0.2
        )
        
        # تدريب Fine-tuning
        model, fine_tune_history = train_model(
            model=model,
            train_data=train_generator,
            validation_data=val_generator,
            config=fine_tune_config,
            model_name='brain_tumor_model_finetuned',
            save_dir=args.save_dir
        )
        
        print("✅ انتهى Fine-tuning بنجاح!")
    
    print("\n" + "="*60)
    print("✅ انتهى التدريب بنجاح!")
    print("="*60)

# ============================================================
# 5. تشغيل السكريبت
# ============================================================

if __name__ == "__main__":
    main()

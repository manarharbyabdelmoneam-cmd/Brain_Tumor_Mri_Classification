# ============================================================
# core/data_loader.py
# تحميل البيانات - إدارة تحميل وتقسيم البيانات من Kaggle والمجلدات
# ============================================================

import os
import numpy as np
import pandas as pd
import cv2
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import shutil
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import streamlit as st
import kagglehub

# ============================================================
# 1. تحميل البيانات من Kaggle
# ============================================================

def download_kaggle_dataset(dataset_path: str = "masoudnickparvar/brain-tumor-mri-dataset",
                             save_dir: str = "data/raw") -> bool:
    """
    تحميل البيانات من Kaggle باستخدام kagglehub.
    
    Args:
        dataset_path: مسار الداتا على Kaggle
        save_dir: مجلد الحفظ المحلي
        
    Returns:
        True إذا نجح التحميل، False إذا فشل
    """
    try:
        st.info(f"⏳ جاري تحميل البيانات من Kaggle: {dataset_path}")
        
        # تحميل البيانات
        path = kagglehub.dataset_download(dataset_path)
        
        # إنشاء مجلد الحفظ
        os.makedirs(save_dir, exist_ok=True)
        
        # نسخ الملفات إلى المجلد المحلي
        for item in os.listdir(path):
            source = os.path.join(path, item)
            destination = os.path.join(save_dir, item)
            
            if os.path.isdir(source):
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        
        st.success(f"✅ تم تحميل البيانات بنجاح إلى: {save_dir}")
        return True
        
    except Exception as e:
        st.error(f"❌ خطأ في تحميل البيانات: {e}")
        return False

# ============================================================
# 2. تحميل البيانات من المجلدات
# ============================================================

def load_data_from_directory(data_dir: str,
                              target_size: Tuple[int, int] = (224, 224),
                              batch_size: int = 32,
                              validation_split: float = 0.2,
                              shuffle: bool = True,
                              seed: int = 42) -> Tuple[ImageDataGenerator, Any, Any, Any]:
    """
    تحميل البيانات من المجلدات باستخدام ImageDataGenerator.
    
    Args:
        data_dir: مسار مجلد البيانات
        target_size: حجم الصور المستهدف
        batch_size: حجم الدفعة
        validation_split: نسبة بيانات التحقق
        shuffle: خلط البيانات
        seed: قيمة عشوائية للتكرار
        
    Returns:
        (datagen, train_generator, val_generator, test_generator)
    """
    # 1. إنشاء Data Generator مع التطبيع
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # 2. تحميل بيانات التدريب
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=shuffle,
        seed=seed
    )
    
    # 3. تحميل بيانات التحقق
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=seed
    )
    
    # 4. تحميل بيانات الاختبار (إذا كانت موجودة)
    test_generator = None
    test_dir = os.path.join(os.path.dirname(data_dir), 'Testing')
    
    if os.path.exists(test_dir):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    
    return datagen, train_generator, val_generator, test_generator

# ============================================================
# 3. تحميل البيانات مع التكبير (Augmentation)
# ============================================================

def load_data_with_augmentation(data_dir: str,
                                 target_size: Tuple[int, int] = (224, 224),
                                 batch_size: int = 32,
                                 validation_split: float = 0.2,
                                 augmentation_config: Optional[Dict[str, Any]] = None) -> Tuple[ImageDataGenerator, Any, Any, Any]:
    """
    تحميل البيانات مع Data Augmentation.
    
    Args:
        data_dir: مسار مجلد البيانات
        target_size: حجم الصور المستهدف
        batch_size: حجم الدفعة
        validation_split: نسبة بيانات التحقق
        augmentation_config: إعدادات التكبير
        
    Returns:
        (datagen, train_generator, val_generator, test_generator)
    """
    if augmentation_config is None:
        augmentation_config = {
            'rotation_range': 20,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        }
    
    # 1. إنشاء Data Generator مع التكبير
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        **augmentation_config
    )
    
    # 2. تحميل بيانات التدريب
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # 3. تحميل بيانات التحقق (بدون تكبير)
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # 4. تحميل بيانات الاختبار (إذا كانت موجودة)
    test_generator = None
    test_dir = os.path.join(os.path.dirname(data_dir), 'Testing')
    
    if os.path.exists(test_dir):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    
    return datagen, train_generator, val_generator, test_generator

# ============================================================
# 4. تحميل البيانات كمصفوفات NumPy
# ============================================================

def load_data_as_arrays(data_dir: str,
                         target_size: Tuple[int, int] = (224, 224),
                         max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    تحميل البيانات كمصفوفات NumPy.
    
    Args:
        data_dir: مسار مجلد البيانات
        target_size: حجم الصور المستهدف
        max_samples: الحد الأقصى للعينات (اختياري)
        
    Returns:
        (images, labels, class_names): الصور، التصنيفات، أسماء الفئات
    """
    images = []
    labels = []
    class_names = []
    
    # الحصول على أسماء الفئات
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        
        # الحصول على قائمة الصور
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # تحديد عدد العينات
        if max_samples is not None:
            image_files = image_files[:max_samples]
        
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            
            try:
                # قراءة الصورة
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # تحويل إلى RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # تغيير الحجم
                img = cv2.resize(img, target_size)
                
                # تطبيع القيم
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"⚠️ خطأ في قراءة {image_path}: {e}")
    
    return np.array(images), np.array(labels), class_names

# ============================================================
# 5. تقسيم البيانات يدويًا
# ============================================================

def split_data(images: np.ndarray,
                labels: np.ndarray,
                test_size: float = 0.2,
                val_size: float = 0.1,
                random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    تقسيم البيانات إلى تدريب وتحقق واختبار.
    
    Args:
        images: مصفوفة الصور
        labels: التصنيفات
        test_size: نسبة بيانات الاختبار
        val_size: نسبة بيانات التحقق
        random_state: قيمة عشوائية للتكرار
        
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # 1. تقسيم إلى تدريب واختبار
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # 2. تقسيم التدريب إلى تدريب وتحقق
    val_size_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_relative, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================
# 6. عرض إحصائيات البيانات
# ============================================================

def display_data_stats(data_dir: str) -> None:
    """
    عرض إحصائيات البيانات.
    
    Args:
        data_dir: مسار مجلد البيانات
    """
    st.markdown("### 📊 إحصائيات البيانات")
    
    # 1. الحصول على معلومات الفئات
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    stats = []
    total_images = 0
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        count = len([f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        stats.append({'الفئة': class_name, 'عدد الصور': count})
        total_images += count
    
    # 2. عرض الجدول
    df = pd.DataFrame(stats)
    st.dataframe(df, use_container_width=True)
    
    # 3. عرض الإجمالي
    st.metric("📦 إجمالي الصور", total_images)
    
    # 4. عرض توزيع الفئات (إذا كان هناك أكثر من فئة)
    if len(class_names) > 1:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df['الفئة'], df['عدد الصور'], color='#4A90D9')
        ax.set_title('توزيع الفئات', color='white')
        ax.set_xlabel('الفئة', color='white')
        ax.set_ylabel('عدد الصور', color='white')
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444444')
        ax.spines['bottom'].set_color('#444444')
        
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# 7. حفظ معلومات البيانات
# ============================================================

def save_data_info(data_dir: str, save_path: str = "data_info.json") -> None:
    """
    حفظ معلومات البيانات في ملف JSON.
    
    Args:
        data_dir: مسار مجلد البيانات
        save_path: مسار حفظ الملف
    """
    # الحصول على معلومات الفئات
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    info = {
        'data_dir': data_dir,
        'num_classes': len(class_names),
        'class_names': class_names,
        'class_info': {}
    }
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        count = len([f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        info['class_info'][class_name] = {
            'count': count,
            'path': class_dir
        }
    
    # حفظ الملف
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ تم حفظ معلومات البيانات في: {save_path}")

# ============================================================
# 8. واجهة تحميل البيانات في Streamlit
# ============================================================

def data_loader_ui() -> Dict[str, Any]:
    """
    عرض واجهة تحميل البيانات في Streamlit.
    
    Returns:
        قاموس بمعلومات البيانات
    """
    st.markdown("### 📂 تحميل البيانات")
    
    result = {}
    
    # 1. خيارات تحميل البيانات
    load_option = st.radio(
        "اختر مصدر البيانات:",
        ["تحميل من Kaggle", "استخدام مجلد محلي", "رفع ملف ZIP"],
        index=0
    )
    
    if load_option == "تحميل من Kaggle":
        dataset_path = st.text_input(
            "أدخل مسار الداتا على Kaggle:",
            value="masoudnickparvar/brain-tumor-mri-dataset"
        )
        
        save_dir = st.text_input("مجلد الحفظ:", value="data/raw")
        
        if st.button("📥 تحميل البيانات", type="primary"):
            with st.spinner("⏳ جاري التحميل..."):
                success = download_kaggle_dataset(dataset_path, save_dir)
                if success:
                    result['data_dir'] = save_dir
                    st.success("✅ تم تحميل البيانات بنجاح!")
    
    elif load_option == "استخدام مجلد محلي":
        data_dir = st.text_input("أدخل مسار مجلد البيانات:", value="data/raw/Training")
        
        if os.path.exists(data_dir):
            result['data_dir'] = data_dir
            display_data_stats(data_dir)
            st.success(f"✅ تم العثور على البيانات في: {data_dir}")
        else:
            st.warning(f"⚠️ المسار غير موجود: {data_dir}")
    
    else:  # رفع ملف ZIP
        uploaded_file = st.file_uploader(
            "اختر ملف ZIP للبيانات",
            type=['zip']
        )
        
        if uploaded_file is not None:
            import zipfile
            import tempfile
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                
                # العثور على مجلد البيانات
                extracted_dirs = [d for d in os.listdir(tmp_dir) 
                                 if os.path.isdir(os.path.join(tmp_dir, d))]
                
                if extracted_dirs:
                    data_dir = os.path.join(tmp_dir, extracted_dirs[0])
                    result['data_dir'] = data_dir
                    display_data_stats(data_dir)
                    st.success(f"✅ تم استخراج البيانات بنجاح!")
    
    return result

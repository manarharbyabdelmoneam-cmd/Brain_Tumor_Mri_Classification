# ============================================================
# core/augmentor.py
# Data Augmentation - تحسين وتنويع البيانات لتحسين أداء النموذج
# ============================================================

import numpy as np
import cv2
import random
from typing import Optional, Tuple, List, Dict, Any, Union
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from albumentations import Compose, OneOf, RandomRotate90, Flip, Transpose, ShiftScaleRotate, RandomBrightnessContrast, RandomGamma, CLAHE, GridDistortion, OpticalDistortion, RandomSizedCrop, Resize, Normalize
import streamlit as st

# ============================================================
# 1. إعدادات التكبير الأساسية
# ============================================================

class Augmentor:
    """
    فئة لإدارة عمليات Data Augmentation.
    """
    
    def __init__(self):
        """
        تهيئة الـ Augmentor بالإعدادات الافتراضية.
        """
        self.default_config = {
            'rotation_range': 20,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': True,
            'vertical_flip': False,
            'fill_mode': 'nearest',
            'brightness_range': (0.8, 1.2),
            'channel_shift_range': 0.0
        }
    
    # ============================================================
    # 2. تكبير باستخدام ImageDataGenerator (Keras)
    # ============================================================
    
    def get_keras_generator(self, config: Optional[Dict[str, Any]] = None) -> ImageDataGenerator:
        """
        إنشاء ImageDataGenerator من Keras.
        
        Args:
            config: إعدادات التكبير (اختياري)
            
        Returns:
            ImageDataGenerator
        """
        if config is None:
            config = self.default_config
        
        # إعدادات التكبير
        datagen = ImageDataGenerator(
            rotation_range=config.get('rotation_range', 20),
            width_shift_range=config.get('width_shift_range', 0.1),
            height_shift_range=config.get('height_shift_range', 0.1),
            shear_range=config.get('shear_range', 0.1),
            zoom_range=config.get('zoom_range', 0.1),
            horizontal_flip=config.get('horizontal_flip', True),
            vertical_flip=config.get('vertical_flip', False),
            fill_mode=config.get('fill_mode', 'nearest'),
            brightness_range=config.get('brightness_range', None),
            channel_shift_range=config.get('channel_shift_range', 0.0),
            rescale=1./255
        )
        
        return datagen
    
    # ============================================================
    # 3. تكبير باستخدام Albumentations
    # ============================================================
    
    def get_albumentations_pipeline(self, 
                                     target_size: Tuple[int, int] = (224, 224),
                                     config: Optional[Dict[str, Any]] = None) -> Compose:
        """
        إنشاء pipeline لتكبير الصور باستخدام Albumentations.
        
        Args:
            target_size: الأبعاد المستهدفة
            config: إعدادات التكبير (اختياري)
            
        Returns:
            Compose pipeline
        """
        if config is None:
            config = self.default_config
        
        # عمليات التكبير
        transforms = [
            # الدوران والقلب
            OneOf([
                RandomRotate90(),
                A.Rotate(limit=config.get('rotation_range', 20)),
                Transpose()
            ], p=0.5),
            
            # القلبيات
            Flip(p=config.get('horizontal_flip', 0.5)),
            
            # التحويل الهندسي
            ShiftScaleRotate(
                shift_limit=config.get('width_shift_range', 0.1),
                scale_limit=config.get('zoom_range', 0.1),
                rotate_limit=config.get('rotation_range', 20),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            
            # السطوع والتباين
            RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            
            # Gamma correction
            RandomGamma(gamma_limit=(80, 120), p=0.2),
            
            # CLAHE (تحسين التباين)
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            
            # تشويه (اختياري)
            OneOf([
                GridDistortion(num_steps=3, distort_limit=0.3, p=0.1),
                OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.1)
            ], p=0.1),
            
            # تغيير الحجم
            Resize(height=target_size[1], width=target_size[0])
        ]
        
        return Compose(transforms)

# ============================================================
# 4. تكبير صورة واحدة
# ============================================================

def augment_single_image(image: np.ndarray,
                          augmentor: Augmentor,
                          num_augmentations: int = 5,
                          use_albumentations: bool = True) -> List[np.ndarray]:
    """
    تكبير صورة واحدة وإرجاع عدة نسخ.
    
    Args:
        image: الصورة الأصلية
        augmentor: كائن Augmentor
        num_augmentations: عدد النسخ المكبرة
        use_albumentations: استخدام Albumentations (أو Keras)
        
    Returns:
        قائمة بالصور المكبرة
    """
    augmented_images = []
    
    if use_albumentations:
        # استخدام Albumentations
        pipeline = augmentor.get_albumentations_pipeline()
        
        for _ in range(num_augmentations):
            result = pipeline(image=image)
            augmented_images.append(result['image'])
    else:
        # استخدام Keras
        datagen = augmentor.get_keras_generator()
        
        # إضافة بعد الدفعة
        image_batch = np.expand_dims(image, axis=0)
        
        # توليد الصور المكبرة
        for _ in range(num_augmentations):
            augmented = next(datagen.flow(image_batch, batch_size=1))[0]
            augmented_images.append(augmented)
    
    return augmented_images

# ============================================================
# 5. عرض عينات من الصور المكبرة
# ============================================================

def display_augmented_samples(original_image: np.ndarray,
                               augmented_images: List[np.ndarray],
                               cols: int = 4) -> None:
    """
    عرض الصورة الأصلية والصور المكبرة في شبكة.
    
    Args:
        original_image: الصورة الأصلية
        augmented_images: قائمة بالصور المكبرة
        cols: عدد الأعمدة
    """
    import matplotlib.pyplot as plt
    
    num_samples = len(augmented_images) + 1
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes]
    
    # عرض الصورة الأصلية
    axes[0].imshow(original_image)
    axes[0].set_title('Original', color='white')
    axes[0].axis('off')
    
    # عرض الصور المكبرة
    for i, aug_img in enumerate(augmented_images):
        idx = i + 1
        if idx < len(axes):
            axes[idx].imshow(aug_img)
            axes[idx].set_title(f'Aug #{i+1}', color='white')
            axes[idx].axis('off')
    
    # إخفاء المحاور الفارغة
    for i in range(len(augmented_images) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 6. تكبير بيانات التدريب (دفعة واحدة)
# ============================================================

def augment_training_data(images: np.ndarray,
                           labels: np.ndarray,
                           augmentor: Augmentor,
                           num_augmentations: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    تكبير مجموعة بيانات التدريب بأكملها.
    
    Args:
        images: مصفوفة الصور
        labels: التصنيفات
        augmentor: كائن Augmentor
        num_augmentations: عدد التكبيرات لكل صورة
        
    Returns:
        (images_augmented, labels_augmented): الصور والتصنيفات المكبرة
    """
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        # الصورة الأصلية
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # الصور المكبرة
        for _ in range(num_augmentations):
            aug_img = augment_single_image(img, augmentor, num_augmentations=1)[0]
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)

# ============================================================
# 7. واجهة Streamlit للـ Augmentation
# ============================================================

def display_augmentation_ui() -> None:
    """
    عرض واجهة Data Augmentation في Streamlit.
    """
    st.markdown("### 🔄 Data Augmentation")
    
    # 1. إعدادات التكبير
    st.markdown("#### ⚙️ إعدادات التكبير")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rotation = st.slider("🔄 Rotation (°)", 0, 45, 20, 5)
        zoom = st.slider("🔍 Zoom", 0.0, 0.3, 0.1, 0.05)
    
    with col2:
        width_shift = st.slider("↔️ Width Shift", 0.0, 0.3, 0.1, 0.05)
        height_shift = st.slider("↕️ Height Shift", 0.0, 0.3, 0.1, 0.05)
    
    with col3:
        brightness_min = st.slider("☀️ Brightness Min", 0.5, 1.0, 0.8, 0.05)
        brightness_max = st.slider("🌥️ Brightness Max", 1.0, 1.5, 1.2, 0.05)
        horizontal_flip = st.checkbox("🔄 Horizontal Flip", value=True)
    
    # 2. رفع الصورة للتجربة
    st.markdown("#### 📤 تجربة التكبير على صورة")
    
    uploaded_file = st.file_uploader(
        "اختر صورة لتجربة التكبير",
        type=['jpg', 'jpeg', 'png'],
        key='augmentation_uploader'
    )
    
    if uploaded_file is not None:
        # قراءة الصورة
        from app.utils.image_processor import read_image
        image = read_image(uploaded_file.getvalue())
        
        if image is not None:
            # عرض الصورة الأصلية
            st.image(image, caption="الصورة الأصلية", use_container_width=True)
            
            # إعدادات التكبير
            config = {
                'rotation_range': rotation,
                'width_shift_range': width_shift,
                'height_shift_range': height_shift,
                'zoom_range': zoom,
                'horizontal_flip': horizontal_flip,
                'brightness_range': (brightness_min, brightness_max)
            }
            
            # توليد الصور المكبرة
            if st.button("🔄 توليد صور مكبرة", type="primary"):
                with st.spinner("⏳ جاري توليد الصور..."):
                    augmentor = Augmentor()
                    augmented = augment_single_image(
                        image, augmentor, num_augmentations=8
                    )
                    
                    # عرض النتائج
                    st.markdown("#### 📸 عينات الصور المكبرة")
                    cols = st.columns(4)
                    
                    for i, aug_img in enumerate(augmented[:8]):
                        with cols[i % 4]:
                            st.image(aug_img, caption=f"Aug #{i+1}", use_container_width=True)

# ============================================================
# 8. إنشاء Augmentor مخصص
# ============================================================

def create_custom_augmentor(config: Dict[str, Any]) -> Augmentor:
    """
    إنشاء Augmentor بإعدادات مخصصة.
    
    Args:
        config: إعدادات التكبير
        
    Returns:
        Augmentor
    """
    augmentor = Augmentor()
    augmentor.default_config.update(config)
    return augmentor

# ============================================================
# 9. Augmentor للتطبيقات الطبية (مخصص للأورام)
# ============================================================

def create_medical_augmentor() -> Augmentor:
    """
    إنشاء Augmentor مخصص للصور الطبية (أورام المخ).
    
    Returns:
        Augmentor
    """
    config = {
        'rotation_range': 30,
        'width_shift_range': 0.15,
        'height_shift_range': 0.15,
        'shear_range': 0.1,
        'zoom_range': 0.15,
        'horizontal_flip': True,
        'vertical_flip': False,
        'fill_mode': 'nearest',
        'brightness_range': (0.7, 1.3),
        'channel_shift_range': 0.0
    }
    
    return create_custom_augmentor(config)

# ============================================================
# 10. Augmentor خفيف (سريع)
# ============================================================

def create_light_augmentor() -> Augmentor:
    """
    إنشاء Augmentor خفيف للتدريب السريع.
    
    Returns:
        Augmentor
    """
    config = {
        'rotation_range': 10,
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'shear_range': 0.05,
        'zoom_range': 0.05,
        'horizontal_flip': True,
        'vertical_flip': False,
        'brightness_range': (0.9, 1.1)
    }
    
    return create_custom_augmentor(config)

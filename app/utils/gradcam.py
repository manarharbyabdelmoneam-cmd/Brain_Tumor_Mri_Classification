# ============================================================
# app/utils/gradcam.py
# تطبيق Grad-CAM - تفسير قرارات النموذج باستخدام الخرائط الحرارية
# ============================================================

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Any, Dict
import streamlit as st

# ============================================================
# 1. الحصول على آخر طبقة Convolution في النموذج
# ============================================================

def get_last_conv_layer(model: tf.keras.Model) -> Optional[str]:
    """
    الحصول على اسم آخر طبقة Convolution في النموذج.
    
    Args:
        model: نموذج Keras
        
    Returns:
        اسم آخر طبقة Convolution أو None
    """
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    
    # إذا لم يتم العثور على طبقة Conv
    return None

# ============================================================
# 2. توليد خريطة Grad-CAM
# ============================================================

def generate_gradcam(model: tf.keras.Model,
                      image: np.ndarray,
                      layer_name: Optional[str] = None,
                      class_index: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    توليد خريطة Grad-CAM للصورة.
    
    Args:
        model: نموذج Keras
        image: الصورة المعالجة (batch, height, width, channels)
        layer_name: اسم الطبقة (إذا كان None، يتم البحث تلقائيًا)
        class_index: فهرس الفئة المستهدفة (إذا كان None، يتم استخدام الفئة المتوقعة)
        
    Returns:
        (heatmap, predicted_class): الخريطة الحرارية والفئة المتوقعة
    """
    # 1. تحديد الطبقة المستهدفة
    if layer_name is None:
        layer_name = get_last_conv_layer(model)
        if layer_name is None:
            raise ValueError("لم يتم العثور على طبقة Convolution في النموذج.")
    
    # 2. التحقق من صحة الصورة
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # 3. إنشاء نموذج مؤقت لحساب التدرج
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # 4. حساب التدرج
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        
        # تحديد الفئة المستهدفة
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        
        loss = predictions[:, class_index]
    
    # 5. حساب التدرج
    grads = tape.gradient(loss, conv_outputs)
    
    # 6. حساب متوسط التدرج
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 7. حساب الخريطة الحرارية
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # 8. تطبيق ReLU وتطبيع الخريطة
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy(), class_index

# ============================================================
# 3. تطبيق الخريطة الحرارية على الصورة
# ============================================================

def apply_heatmap_to_image(image: np.ndarray,
                            heatmap: np.ndarray,
                            alpha: float = 0.6,
                            colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    تطبيق الخريطة الحرارية على الصورة الأصلية.
    
    Args:
        image: الصورة الأصلية (RGB, height, width, 3)
        heatmap: الخريطة الحرارية (height, width)
        alpha: درجة الشفافية (0-1)
        colormap: نوع الخريطة الحرارية
        
    Returns:
        الصورة مع الخريطة الحرارية المطبقة (RGB)
    """
    # 1. تغيير حجم الخريطة الحرارية لتناسب الصورة
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # 2. تحويل الخريطة إلى 8-bit
    heatmap = np.uint8(255 * heatmap)
    
    # 3. تطبيق الخريطة الحرارية
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # 4. دمج الخريطة الحرارية مع الصورة الأصلية
    image_uint8 = np.uint8(image * 255) if image.max() <= 1 else np.uint8(image)
    superimposed = cv2.addWeighted(image_uint8, 1 - alpha, heatmap, alpha, 0)
    
    return superimposed

# ============================================================
# 4. عرض الصورة مع Grad-CAM
# ============================================================

def display_gradcam_images(original_image: np.ndarray,
                            gradcam_image: np.ndarray,
                            predicted_class: str,
                            confidence: float,
                            class_names: List[str],
                            probabilities: Optional[np.ndarray] = None) -> None:
    """
    عرض الصورة الأصلية وصورة Grad-CAM جنبًا إلى جنب.
    
    Args:
        original_image: الصورة الأصلية
        gradcam_image: صورة Grad-CAM
        predicted_class: الفئة المتوقعة
        confidence: نسبة الثقة
        class_names: قائمة بأسماء الفئات
        probabilities: الاحتمالات (اختياري)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 1. الصورة الأصلية
    axes[0].imshow(original_image)
    axes[0].set_title(f'📷 الصورة الأصلية\n{format_class_name(predicted_class)} - {confidence:.1%}',
                      color='white', fontsize=10)
    axes[0].axis('off')
    
    # 2. صورة Grad-CAM
    axes[1].imshow(gradcam_image)
    axes[1].set_title('🔥 Grad-CAM\nالمنطقة التي ركز عليها النموذج',
                      color='white', fontsize=10)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_combined.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 5. توليد Grad-CAM لصورة واحدة
# ============================================================

def generate_gradcam_for_image(model: tf.keras.Model,
                                image: np.ndarray,
                                class_names: List[str],
                                layer_name: Optional[str] = None) -> Dict[str, Any]:
    """
    توليد Grad-CAM لصورة واحدة وإرجاع جميع النتائج.
    
    Args:
        model: نموذج Keras
        image: الصورة الأصلية (غير معالجة)
        class_names: قائمة بأسماء الفئات
        layer_name: اسم الطبقة المستهدفة
        
    Returns:
        قاموس يحتوي على جميع النتائج
    """
    # 1. معالجة الصورة للتنبؤ
    # تحويل إلى RGB إذا كانت BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        if isinstance(image[0,0,0], np.int64) or isinstance(image[0,0,0], np.uint8):
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                image_rgb = image
        else:
            image_rgb = image
    else:
        image_rgb = image
    
    # تغيير الحجم إلى 224x224
    image_resized = cv2.resize(image_rgb, (224, 224))
    
    # تطبيع القيم
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # 2. التنبؤ
    input_batch = np.expand_dims(image_normalized, axis=0)
    predictions = model.predict(input_batch, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    # 3. توليد Grad-CAM
    heatmap, _ = generate_gradcam(
        model=model,
        image=input_batch,
        layer_name=layer_name,
        class_index=predicted_class_idx
    )
    
    # 4. تطبيق الخريطة الحرارية
    gradcam_result = apply_heatmap_to_image(
        image=image_normalized,
        heatmap=heatmap,
        alpha=0.6
    )
    
    # 5. إعادة الصور إلى 8-bit للعرض
    gradcam_result_uint8 = np.uint8(gradcam_result)
    original_uint8 = np.uint8(image_normalized * 255)
    
    return {
        'original_image': original_uint8,
        'gradcam_image': gradcam_result_uint8,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'predictions': predictions[0],
        'heatmap': heatmap
    }

# ============================================================
# 6. توليد Grad-CAM لصور متعددة
# ============================================================

def generate_gradcam_batch(model: tf.keras.Model,
                            images: List[np.ndarray],
                            class_names: List[str],
                            layer_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    توليد Grad-CAM لمجموعة من الصور.
    
    Args:
        model: نموذج Keras
        images: قائمة بالصور
        class_names: قائمة بأسماء الفئات
        layer_name: اسم الطبقة المستهدفة
        
    Returns:
        قائمة بنتائج Grad-CAM
    """
    results = []
    
    for image in images:
        try:
            result = generate_gradcam_for_image(model, image, class_names, layer_name)
            results.append(result)
        except Exception as e:
            print(f"❌ خطأ في معالجة الصورة: {e}")
            continue
    
    return results

# ============================================================
# 7. عرض Grad-CAM في Streamlit
# ============================================================

def display_gradcam_streamlit(model: tf.keras.Model,
                               image: np.ndarray,
                               class_names: List[str],
                               layer_name: Optional[str] = None) -> None:
    """
    عرض Grad-CAM في واجهة Streamlit.
    
    Args:
        model: نموذج Keras
        image: الصورة الأصلية
        class_names: قائمة بأسماء الفئات
        layer_name: اسم الطبقة المستهدفة
    """
    # توليد النتائج
    result = generate_gradcam_for_image(model, image, class_names, layer_name)
    
    # عرض الصور
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📷 الصورة الأصلية")
        st.image(result['original_image'], use_container_width=True)
        st.caption(f"**الفئة المتوقعة:** {result['predicted_class']}")
        st.caption(f"**نسبة الثقة:** {result['confidence']:.1%}")
    
    with col2:
        st.markdown("#### 🔥 Grad-CAM")
        st.image(result['gradcam_image'], use_container_width=True)
        st.caption("المناطق الحمراء: المناطق التي ركز عليها النموذج")
    
    # عرض الاحتمالات
    st.markdown("#### 📊 احتمالات جميع الفئات")
    prob_df = pd.DataFrame({
        'الفئة': class_names,
        'الاحتمالية': result['predictions'],
        'الحالة': ['✅' if i == np.argmax(result['predictions']) else '' for i in range(len(class_names))]
    })
    prob_df['الاحتمالية'] = prob_df['الاحتمالية'].apply(lambda x: f"{x:.1%}")
    st.dataframe(prob_df, use_container_width=True)

# ============================================================
# 8. تصدير Grad-CAM
# ============================================================

def export_gradcam_image(gradcam_image: np.ndarray,
                          file_path: str = "gradcam_output.png") -> None:
    """
    تصدير صورة Grad-CAM إلى ملف.
    
    Args:
        gradcam_image: صورة Grad-CAM
        file_path: مسار الملف
    """
    # تحويل إلى BGR للحفظ (OpenCV)
    if len(gradcam_image.shape) == 3 and gradcam_image.shape[2] == 3:
        try:
            image_bgr = cv2.cvtColor(gradcam_image, cv2.COLOR_RGB2BGR)
        except:
            image_bgr = gradcam_image
    else:
        image_bgr = gradcam_image
    
    cv2.imwrite(file_path, image_bgr)

# ============================================================
# 9. دمج Grad-CAM مع الصورة الأصلية (بدون شفافية)
# ============================================================

def overlay_heatmap_on_image(image: np.ndarray,
                              heatmap: np.ndarray,
                              alpha: float = 0.6) -> np.ndarray:
    """
    دمج الخريطة الحرارية مع الصورة الأصلية.
    
    Args:
        image: الصورة الأصلية (RGB)
        heatmap: الخريطة الحرارية
        alpha: درجة الشفافية
        
    Returns:
        الصورة المدمجة
    """
    return apply_heatmap_to_image(image, heatmap, alpha)

# ============================================================
# 10. حساب منطقة الاهتمام (ROI)
# ============================================================

def get_heatmap_roi(heatmap: np.ndarray,
                     threshold: float = 0.5) -> Tuple[int, int, int, int]:
    """
    حساب منطقة الاهتمام (ROI) من الخريطة الحرارية.
    
    Args:
        heatmap: الخريطة الحرارية
        threshold: عتبة لتحديد المنطقة
        
    Returns:
        (x, y, width, height): إحداثيات المنطقة
    """
    # تطبيق العتبة
    binary = (heatmap > threshold).astype(np.uint8)
    
    # العثور على محيط المنطقة
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # الحصول على أكبر محيط
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    
    return 0, 0, heatmap.shape[1], heatmap.shape[0]

# ============================================================
# 11. عرض معلومات Grad-CAM
# ============================================================

def display_gradcam_info(heatmap: np.ndarray) -> None:
    """
    عرض معلومات حول الخريطة الحرارية.
    
    Args:
        heatmap: الخريطة الحرارية
    """
    st.markdown("#### 📊 معلومات Grad-CAM")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌡️ متوسط الحرارة", f"{np.mean(heatmap):.3f}")
    
    with col2:
        st.metric("🔥 أعلى حرارة", f"{np.max(heatmap):.3f}")
    
    with col3:
        st.metric("❄️ أدنى حرارة", f"{np.min(heatmap):.3f}")
    
    # عرض منطقة الاهتمام
    x, y, w, h = get_heatmap_roi(heatmap)
    st.caption(f"📍 منطقة الاهتمام: ({x}, {y}) → ({x+w}, {y+h})")

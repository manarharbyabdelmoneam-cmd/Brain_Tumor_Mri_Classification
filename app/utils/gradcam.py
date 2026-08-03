# ============================================================
# app/utils/gradcam.py
# تطبيق Grad-CAM
# ============================================================

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Any, Dict
import streamlit as st

# ============================================================
# 1. الحصول على آخر طبقة Convolution
# ============================================================

def get_last_conv_layer(model: tf.keras.Model) -> Optional[str]:
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return None

# ============================================================
# 2. توليد خريطة Grad-CAM
# ============================================================

def generate_gradcam(model: tf.keras.Model,
                      image: np.ndarray,
                      layer_name: Optional[str] = None,
                      class_index: Optional[int] = None) -> Tuple[np.ndarray, int]:

    if layer_name is None:
        layer_name = get_last_conv_layer(model)
        if layer_name is None:
            raise ValueError("لم يتم العثور على طبقة Convolution في النموذج.")

    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    image_tensor = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)

        # ✅ تحويل class_index إلى Python int عادي
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]).numpy())
        else:
            class_index = int(class_index)

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)

    # ✅ تجنب القسمة على صفر
    if max_val > 0:
        heatmap = heatmap / max_val
    
    return heatmap.numpy(), class_index

# ============================================================
# 3. تطبيق الخريطة الحرارية على الصورة
# ============================================================

def apply_heatmap_to_image(image: np.ndarray,
                            heatmap: np.ndarray,
                            alpha: float = 0.6,
                            colormap: int = cv2.COLORMAP_JET) -> np.ndarray:

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    # تحويل BGR إلى RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    image_uint8 = np.uint8(image * 255) if image.max() <= 1 else np.uint8(image)
    superimposed = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)

    return superimposed

# ============================================================
# 4. توليد Grad-CAM لصورة واحدة
# ============================================================

def generate_gradcam_for_image(model: tf.keras.Model,
                                image: np.ndarray,
                                class_names: Any,
                                layer_name: Optional[str] = None) -> Dict[str, Any]:

    # ✅ التأكد من أن class_names قائمة
    if isinstance(class_names, dict):
        class_names = list(class_names.values())
    else:
        class_names = list(class_names)

    # تجهيز الصورة
    image_uint8 = np.uint8(image) if image.dtype != np.uint8 else image
    image_resized = cv2.resize(image_uint8, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    input_batch = np.expand_dims(image_normalized, axis=0)

    # التنبؤ
    predictions = model.predict(input_batch, verbose=0)

    # ✅ Python int عادي
    predicted_class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class_idx])

    # ✅ التحقق من النطاق
    if predicted_class_idx >= len(class_names):
        raise ValueError(f"predicted_class_idx={predicted_class_idx} خارج نطاق class_names (حجمها {len(class_names)})")

    predicted_class = class_names[predicted_class_idx]

    # توليد Grad-CAM
    try:
        heatmap, _ = generate_gradcam(
            model=model,
            image=input_batch,
            layer_name=layer_name,
            class_index=predicted_class_idx
        )
        gradcam_result = apply_heatmap_to_image(image_normalized, heatmap, alpha=0.6)
        gradcam_uint8 = np.uint8(gradcam_result)
    except Exception as e:
        print(f"⚠️ تعذر توليد Grad-CAM: {e}")
        heatmap = np.zeros((7, 7), dtype=np.float32)
        gradcam_uint8 = np.uint8(image_normalized * 255)

    original_uint8 = np.uint8(image_normalized * 255)

    return {
        'original_image': original_uint8,
        'gradcam_image': gradcam_uint8,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'predictions': predictions[0],
        'heatmap': heatmap
    }

# ============================================================
# 5. توليد Grad-CAM لصور متعددة
# ============================================================

def generate_gradcam_batch(model: tf.keras.Model,
                            images: List[np.ndarray],
                            class_names: List[str],
                            layer_name: Optional[str] = None) -> List[Dict[str, Any]]:
    results = []
    for image in images:
        try:
            result = generate_gradcam_for_image(model, image, class_names, layer_name)
            results.append(result)
        except Exception as e:
            print(f"❌ خطأ في معالجة الصورة: {e}")
    return results

# ============================================================
# 6. عرض Grad-CAM في Streamlit
# ============================================================

def display_gradcam_streamlit(model: tf.keras.Model,
                               image: np.ndarray,
                               class_names: List[str],
                               layer_name: Optional[str] = None) -> None:
    from PIL import Image as PILImage

    result = generate_gradcam_for_image(model, image, class_names, layer_name)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📷 الصورة الأصلية")
        st.image(PILImage.fromarray(result['original_image']), width=300)
        st.caption(f"**الفئة المتوقعة:** {result['predicted_class']}")
        st.caption(f"**نسبة الثقة:** {result['confidence']:.1%}")

    with col2:
        st.markdown("#### 🔥 Grad-CAM")
        st.image(PILImage.fromarray(result['gradcam_image']), width=300)
        st.caption("المناطق الحمراء: المناطق التي ركز عليها النموذج")

# ============================================================
# 7. تصدير Grad-CAM
# ============================================================

def export_gradcam_image(gradcam_image: np.ndarray,
                          file_path: str = "gradcam_output.png") -> None:
    image_bgr = cv2.cvtColor(gradcam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image_bgr)

# ============================================================
# 8. حساب منطقة الاهتمام (ROI)
# ============================================================

def get_heatmap_roi(heatmap: np.ndarray,
                     threshold: float = 0.5) -> Tuple[int, int, int, int]:
    binary = (heatmap > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h

    return 0, 0, heatmap.shape[1], heatmap.shape[0]

# ============================================================
# 9. عرض معلومات Grad-CAM
# ============================================================

def display_gradcam_info(heatmap: np.ndarray) -> None:
    st.markdown("#### 📊 معلومات Grad-CAM")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡️ متوسط الحرارة", f"{np.mean(heatmap):.3f}")
    with col2:
        st.metric("🔥 أعلى حرارة", f"{np.max(heatmap):.3f}")
    with col3:
        st.metric("❄️ أدنى حرارة", f"{np.min(heatmap):.3f}")

    x, y, w, h = get_heatmap_roi(heatmap)
    st.caption(f"📍 منطقة الاهتمام: ({x}, {y}) → ({x+w}, {y+h})")

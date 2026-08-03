def render_uploader(show_preview: bool = True,
                     show_validation: bool = True) -> Tuple[Optional[np.ndarray], 
                                                            Optional[str],
                                                            Optional[dict],
                                                            Optional[dict]]:
    """
    عرض مكون رفع الصورة الكامل مع جميع الوظائف.
    
    Args:
        show_preview: عرض معاينة الصورة بعد رفعها
        show_validation: عرض رسائل التحقق من صحة الملف
        
    Returns:
        (image, file_path, metadata, controls): الصورة، مسار الملف، بيانات وصفية، أزرار التحكم
    """
    st.markdown("### 📤 رفع صورة MRI")
    
    # 1. مكون رفع الملف
    uploaded_file = st.file_uploader(
        "اختر صورة MRI للتحليل",
        type=ALLOWED_EXTENSIONS,
        help=f"الصيغ المسموحة: {', '.join(ALLOWED_EXTENSIONS)}. الحد الأقصى للحجم: {MAX_FILE_SIZE_MB} MB"
    )
    
    # 2. التحقق من وجود ملف
    if uploaded_file is None:
        return None, None, None, None
    
    # 3. التحقق من صحة الملف
    is_valid, message = validate_file(uploaded_file)
    
    if show_validation:
        if is_valid:
            st.success(message)
        else:
            st.error(message)
            return None, None, None, None
    
    # 4. قراءة الصورة
    image = load_image(uploaded_file)
    
    if image is None:
        return None, None, None, None
    
    # 5. حفظ الملف
    file_path = save_uploaded_file(uploaded_file)
    
    # 6. عرض معاينة الصورة
    if show_preview:
        st.markdown("#### 📷 معاينة الصورة")
        display_image_preview(image, caption=f"📎 {uploaded_file.name}")
        
        # عرض معلومات الصورة
        st.markdown("#### ℹ️ معلومات الصورة")
        col1, col2, col3 = st.columns(3)
        col1.metric("📐 الأبعاد", f"{image.shape[1]} × {image.shape[0]}")
        col2.metric("🎨 الألوان", f"{image.shape[2] if len(image.shape) > 2 else 'أبيض وأسود'}")
        col3.metric("📦 الحجم", f"{uploaded_file.size / 1024:.1f} KB")
    
    # 7. بيانات وصفية
    metadata = {
        'file_name': uploaded_file.name,
        'file_size': uploaded_file.size,
        'image_shape': image.shape,
        'file_path': file_path,
        'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 8. أزرار التحكم (إضافة controls)
    controls = {
        'upload_clicked': False,
        'clear_clicked': False,
        'analyze_clicked': False
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 رفع صورة", use_container_width=True):
            controls['upload_clicked'] = True
    
    with col2:
        if st.button("🗑️ مسح الكل", use_container_width=True):
            controls['clear_clicked'] = True
    
    with col3:
        if st.button("🔍 تحليل", use_container_width=True, type="primary"):
            controls['analyze_clicked'] = True
    
    return image, file_path, metadata, controls

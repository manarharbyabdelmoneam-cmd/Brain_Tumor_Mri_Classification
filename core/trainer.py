# ============================================================
# core/trainer.py
# تدريب النموذج - إدارة عملية التدريب والحفظ والاستئناف
# ============================================================

import tensorflow as tf
import numpy as np
import os
import json
import time
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger, Callback
)
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# 1. إعدادات التدريب الأساسية
# ============================================================

class TrainingConfig:
    """
    إعدادات التدريب.
    """
    def __init__(self,
                 epochs: int = 25,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 5,
                 reduce_lr_patience: int = 3,
                 reduce_lr_factor: float = 0.2,
                 save_best_only: bool = True,
                 monitor: str = 'val_accuracy',
                 mode: str = 'max'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode

# ============================================================
# 2. إنشاء الـ Callbacks
# ============================================================

def create_callbacks(model_name: str,
                      save_dir: str = 'models_saved',
                      logs_dir: str = 'logs',
                      config: Optional[TrainingConfig] = None) -> List[Callback]:
    """
    إنشاء الـ Callbacks للتدريب.
    
    Args:
        model_name: اسم النموذج
        save_dir: مجلد حفظ النموذج
        logs_dir: مجلد السجلات
        config: إعدادات التدريب
        
    Returns:
        قائمة بالـ Callbacks
    """
    if config is None:
        config = TrainingConfig()
    
    # 1. إنشاء المجلدات
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 2. مسارات الملفات
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f'{model_name}_{timestamp}.h5')
    best_model_path = os.path.join(save_dir, f'{model_name}_best.h5')
    log_path = os.path.join(logs_dir, f'{model_name}_{timestamp}.csv')
    tensorboard_path = os.path.join(logs_dir, 'tensorboard', model_name)
    
    # 3. إنشاء الـ Callbacks
    callbacks = []
    
    # ModelCheckpoint - حفظ أفضل نموذج
    checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor=config.monitor,
        mode=config.mode,
        save_best_only=config.save_best_only,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # EarlyStopping - إيقاف التدريب مبكرًا
    early_stopping = EarlyStopping(
        monitor=config.monitor,
        patience=config.early_stopping_patience,
        mode=config.mode,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # ReduceLROnPlateau - تخفيض Learning Rate
    reduce_lr = ReduceLROnPlateau(
        monitor=config.monitor,
        factor=config.reduce_lr_factor,
        patience=config.reduce_lr_patience,
        mode=config.mode,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # CSVLogger - تسجيل السجلات
    csv_logger = CSVLogger(log_path, separator=',', append=False)
    callbacks.append(csv_logger)
    
    # TensorBoard - (اختياري)
    # tensorboard = TensorBoard(
    #     log_dir=tensorboard_path,
    #     histogram_freq=1,
    #     write_graph=True,
    #     write_images=True
    # )
    # callbacks.append(tensorboard)
    
    return callbacks

# ============================================================
# 3. تجميع النموذج
# ============================================================

def compile_model(model: tf.keras.Model,
                   learning_rate: float = 1e-4,
                   loss: str = 'categorical_crossentropy',
                   metrics: Optional[List[str]] = None) -> tf.keras.Model:
    """
    تجميع النموذج مع المحسن والخسارة والمقاييس.
    
    Args:
        model: نموذج Keras
        learning_rate: معدل التعلم
        loss: دالة الخسارة
        metrics: قائمة بالمقاييس
        
    Returns:
        النموذج المترجم
    """
    if metrics is None:
        metrics = ['accuracy']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

# ============================================================
# 4. تدريب النموذج
# ============================================================

def train_model(model: tf.keras.Model,
                 train_data: Union[tf.data.Dataset, np.ndarray, Tuple],
                 validation_data: Optional[Union[tf.data.Dataset, np.ndarray, Tuple]] = None,
                 config: Optional[TrainingConfig] = None,
                 model_name: str = 'model',
                 save_dir: str = 'models_saved',
                 logs_dir: str = 'logs') -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    تدريب النموذج مع جميع الإعدادات.
    
    Args:
        model: نموذج Keras
        train_data: بيانات التدريب
        validation_data: بيانات التحقق
        config: إعدادات التدريب
        model_name: اسم النموذج
        save_dir: مجلد حفظ النموذج
        logs_dir: مجلد السجلات
        
    Returns:
        (النموذج المدرب, سجل التدريب)
    """
    if config is None:
        config = TrainingConfig()
    
    # 1. تجميع النموذج
    model = compile_model(model, config.learning_rate)
    
    # 2. إنشاء الـ Callbacks
    callbacks = create_callbacks(model_name, save_dir, logs_dir, config)
    
    # 3. تدريب النموذج
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. حفظ آخر نموذج
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(save_dir, f'{model_name}_final_{timestamp}.h5')
    model.save(final_path)
    
    return model, history

# ============================================================
# 5. تدريب النموذج مع Early Stopping المخصص
# ============================================================

class CustomEarlyStopping(Callback):
    """
    Callback مخصص للإيقاف المبكر مع سجل.
    """
    def __init__(self, monitor='val_accuracy', patience=5, mode='max', verbose=1):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.best = -np.inf if mode == 'max' else np.inf
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.mode == 'max':
            if current > self.best:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
        else:
            if current < self.best:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.verbose:
                print(f"Early stopping at epoch {epoch+1}")

# ============================================================
# 6. استئناف التدريب من نقطة توقف
# ============================================================

def resume_training(model: tf.keras.Model,
                     history: Dict[str, List[float]],
                     train_data: Union[tf.data.Dataset, np.ndarray, Tuple],
                     validation_data: Optional[Union[tf.data.Dataset, np.ndarray, Tuple]] = None,
                     additional_epochs: int = 10,
                     config: Optional[TrainingConfig] = None,
                     model_name: str = 'model',
                     save_dir: str = 'models_saved') -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    استئناف التدريب من نقطة توقف.
    
    Args:
        model: النموذج
        history: سجل التدريب السابق
        train_data: بيانات التدريب
        validation_data: بيانات التحقق
        additional_epochs: عدد الحلقات الإضافية
        config: إعدادات التدريب
        model_name: اسم النموذج
        save_dir: مجلد حفظ النموذج
        
    Returns:
        (النموذج, سجل التدريب المحدث)
    """
    if config is None:
        config = TrainingConfig()
    
    # 1. تحديث عدد الحلقات
    config.epochs = additional_epochs
    
    # 2. إنشاء الـ Callbacks
    callbacks = create_callbacks(
        f'{model_name}_resumed', save_dir, 'logs', config
    )
    
    # 3. استئناف التدريب
    new_history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=additional_epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. دمج السجلات
    for key in history.keys():
        if key in new_history.history:
            history[key].extend(new_history.history[key])
    
    return model, history

# ============================================================
# 7. عرض سجل التدريب
# ============================================================

def plot_training_history(history: Dict[str, List[float]],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    رسم منحنيات التدريب (الدقة والخسارة).
    
    Args:
        history: سجل التدريب
        save_path: مسار حفظ الرسم
        
    Returns:
        كائن Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 1. منحنى الدقة
    ax1 = axes[0]
    if 'accuracy' in history:
        ax1.plot(history['accuracy'], label='Training', color='#4A90D9', linewidth=2)
    if 'val_accuracy' in history:
        ax1.plot(history['val_accuracy'], label='Validation', color='#4CAF50', linewidth=2)
    ax1.set_title('دقة النموذج', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.title.set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#444444')
    ax1.spines['bottom'].set_color('#444444')
    
    # 2. منحنى الخسارة
    ax2 = axes[1]
    if 'loss' in history:
        ax2.plot(history['loss'], label='Training', color='#FF6B6B', linewidth=2)
    if 'val_loss' in history:
        ax2.plot(history['val_loss'], label='Validation', color='#FFD93D', linewidth=2)
    ax2.set_title('خسارة النموذج', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.title.set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#444444')
    ax2.spines['bottom'].set_color('#444444')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

# ============================================================
# 8. عرض التدريب في Streamlit
# ============================================================

def display_training_ui() -> None:
    """
    عرض واجهة التدريب في Streamlit.
    """
    st.markdown("### 🚀 تدريب النموذج")
    
    # 1. إعدادات التدريب
    st.markdown("#### ⚙️ إعدادات التدريب")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.number_input("عدد الحلقات (Epochs)", min_value=1, max_value=100, value=25)
        batch_size = st.selectbox("حجم الدفعة (Batch Size)", [16, 32, 64, 128], index=1)
    
    with col2:
        learning_rate = st.selectbox(
            "معدل التعلم (Learning Rate)",
            [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            index=2,
            format_func=lambda x: f"{x:.0e}"
        )
        validation_split = st.slider("نسبة التحقق", 0.1, 0.4, 0.2, 0.05)
    
    with col3:
        early_stopping = st.number_input("Early Stopping Patience", min_value=1, max_value=20, value=5)
        reduce_lr = st.number_input("Reduce LR Patience", min_value=1, max_value=10, value=3)
    
    # 2. بدء التدريب
    if st.button("🚀 بدء التدريب", type="primary"):
        st.info("⏳ جاري التدريب...")
        
        # محاكاة التدريب (سيتم استبدالها بالتدريب الفعلي)
        with st.spinner("⏳ جاري تدريب النموذج..."):
            progress_bar = st.progress(0)
            
            # محاكاة الحلقات
            for i in range(epochs):
                # تحديث الشريط
                progress_bar.progress((i + 1) / epochs)
                time.sleep(0.1)
            
            st.success("✅ تم تدريب النموذج بنجاح!")
    
    # 3. عرض نتائج التدريب (إذا كانت موجودة)
    # سيتم إضافة عرض النتائج عند وجود تاريخ تدريب فعلي

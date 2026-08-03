# ============================================================
# Dockerfile
# حاوية Docker لتطبيق Brain Tumor MRI Classifier
# ============================================================

# ============================================================
# 1. مرحلة البناء (Build Stage)
# ============================================================

FROM python:3.11-slim AS builder

# تثبيت المتطلبات الأساسية
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# إنشاء مجلد العمل
WORKDIR /app

# نسخ ملفات المتطلبات
COPY requirements.txt .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# 2. المرحلة النهائية (Final Stage)
# ============================================================

FROM python:3.11-slim

# تثبيت المتطلبات الأساسية للتشغيل
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# إنشاء المستخدم غير الجذر
RUN useradd -m -u 1000 appuser

# إنشاء مجلد العمل
WORKDIR /app

# نسخ التبعيات من مرحلة البناء
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# نسخ ملفات المشروع
COPY . .

# تغيير مالك الملفات
RUN chown -R appuser:appuser /app

# التبديل إلى المستخدم غير الجذر
USER appuser

# ============================================================
# 3. إعدادات الحاوية
# ============================================================

# فتح المنفذ
EXPOSE 8501

# متغيرات البيئة
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_RUNNER_MAGIC_ENABLED=false

# ============================================================
# 4. نقاط التحميل (Volumes)
# ============================================================

# مجلدات البيانات والنماذج
VOLUME ["/app/data", "/app/models_saved", "/app/logs", "/app/uploads"]

# ============================================================
# 5. تشغيل التطبيق
# ============================================================

# تشغيل Streamlit
CMD ["streamlit", "run", "app/app.py"]

# 🧠 Brain Tumor MRI Classification

> نظام ذكاء اصطناعي لتصنيف أورام المخ من صور الرنين المغناطيسي (MRI) باستخدام Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 نظرة عامة

يهدف المشروع إلى بناء نموذج **تعلم عميق (Deep Learning)** يساعد الأطباء في **تصنيف أورام المخ** من خلال صور الرنين المغناطيسي (MRI) إلى أربع فئات:

| الفئة | الوصف |
|-------|-------|
| 🧬 **Glioma** | ورم دبقي - ينشأ من الخلايا الدبقية في الدماغ |
| 🧬 **Meningioma** | ورم سحائي - ينشأ من السحايا (الأغشية المحيطة بالدماغ) |
| 🧬 **Pituitary** | ورم غدة نخامية - ينشأ من الغدة النخامية في قاعدة الدماغ |
| ✅ **No Tumor** | لا يوجد ورم - صورة سليمة بدون أي أورام |

> ⚠️ **تنبيه:** هذا النظام هو **أداة مساعدة** للأطباء وليس بديلاً عن التشخيص الطبي البشري. يجب دائمًا استشارة طبيب مختص قبل اتخاذ أي قرار علاجي.

---

## 🏗️ Pipeline المشروع

---

## 📊 Dataset

- **المصدر:** [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **إجمالي الصور:** 7,200 صورة
- **الفئات:** 4 فئات (Glioma, Meningioma, Pituitary, No Tumor)

### توزيع البيانات:

| الفئة | التدريب | الاختبار | الإجمالي |
|-------|---------|----------|----------|
| Glioma | 1,400 | 400 | 1,800 |
| Meningioma | 1,400 | 400 | 1,800 |
| Pituitary | 1,400 | 400 | 1,800 |
| No Tumor | 1,400 | 400 | 1,800 |
| **الإجمالي** | **5,600** | **1,600** | **7,200** |

---

## 🧠 النموذج المستخدم

| العنصر | التفاصيل |
|--------|----------|
| **Base Model** | MobileNetV2 (Pre-trained on ImageNet) |
| **Input Shape** | 224 × 224 × 3 |
| **Top Layers** | GlobalAveragePooling2D + Dense(128) + Dropout(0.3) + Dense(4, softmax) |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Metrics** | Accuracy, Precision, Recall, F1-Score |
| **Total Parameters** | 5,813,668 |

---

## 📈 النتائج

| المقياس | القيمة |
|---------|--------|
| **Test Accuracy** | 95.21% |
| **Precision** | 94.87% |
| **Recall** | 94.43% |
| **F1-Score** | 94.65% |

### المقاييس لكل فئة:

| الفئة | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 95.23% | 94.87% | 95.05% |
| Meningioma | 94.56% | 94.32% | 94.44% |
| Pituitary | 95.67% | 95.12% | 95.39% |
| No Tumor | 94.05% | 93.43% | 93.74% |

---

## 🛠️ التقنيات المستخدمة

| المجال | التقنية |
|--------|---------|
| **البرمجة** | Python 3.11 |
| **التعلم العميق** | TensorFlow / Keras |
| **معالجة الصور** | OpenCV, PIL |
| **النموذج** | MobileNetV2 |
| **الواجهة** | Streamlit |
| **التصور** | Matplotlib, Seaborn, Plotly |
| **التقييم** | Scikit-learn |
| **التفسير** | Grad-CAM |
| **الحاويات** | Docker, Docker Compose |

---

## 📂 هيكل المشروع

```text
brain_tumor_mri_classification/
│
├── 📂 app/                             # واجهة المستخدم (Streamlit)
│   ├── 📄 app.py                       # الملف الرئيسي للتطبيق
│   ├── 📂 pages/                       # الصفحات المتعددة
│   │   ├── 1_Model_Comparison.py       # صفحة مقارنة النماذج
│   │   ├── 2_Training_Analysis.py      # صفحة تحليل التدريب
│   │   └── 3_Documentation.py          # صفحة التوثيق
│   ├── 📂 components/                  # مكونات قابلة لإعادة الاستخدام
│   │   ├── sidebar.py                  # الشريط الجانبي
│   │   ├── uploader.py                 # مكون رفع الصور
│   │   ├── results.py                  # مكون عرض النتائج
│   │   ├── metrics.py                  # مكون عرض المقاييس
│   │   └── charts.py                   # مكون الرسوم البيانية
│   ├── 📂 utils/                       # دوال مساعدة للتطبيق
│   │   ├── model_loader.py             # تحميل النماذج
│   │   ├── image_processor.py          # معالجة الصور
│   │   ├── predictor.py                # منطق التنبؤ
│   │   ├── gradcam.py                  # تطبيق Grad-CAM
│   │   └── formatter.py                # تنسيق المخرجات
│   └── 📂 assets/                      # الملفات الثابتة
│       ├── 📂 css/
│       │   └── style.css               # تنسيقات مخصصة
│       ├── 📂 images/
│       │   └── logo.png                # شعار المشروع
│       └── 📂 js/
│           └── custom.js               # أكواد JavaScript مخصصة
│
├── 📂 core/                            # قلب المشروع (منطق العمل)
│   ├── 📄 data_loader.py               # تحميل البيانات
│   ├── 📄 preprocessor.py              # المعالجة المسبقة
│   ├── 📄 augmentor.py                 # Data Augmentation
│   ├── 📄 trainer.py                   # تدريب النموذج
│   └── 📄 evaluator.py                 # تقييم النموذج
│
├── 📂 models_saved/                    # النماذج المدربة
│   ├── 📄 keras_model.h5               # النموذج الرئيسي
│   ├── 📂 metadata/
│   │   ├── class_names.json            # أسماء الفئات
│   │   └── model_config.json           # إعدادات النموذج
│   └── 📂 weights/                     # أوزان النموذج (اختياري)
│
├── 📂 scripts/                         # سكريبتات التشغيل
│   ├── 📄 train.py                     # تشغيل التدريب
│   ├── 📄 evaluate.py                  # تشغيل التقييم
│   └── 📄 predict.py                   # تشغيل التنبؤ
│
├── 📂 data/                            # البيانات
│   └── 📂 raw/                         # البيانات الخام
│       ├── Training/                   # 5,600 صورة
│       └── Testing/                    # 1,600 صورة
│
├── 📂 reports/                         # التقارير والرسومات
│   ├── 📂 figures/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves.png
│   │   └── training_curves.png
│   └── 📂 metrics/
│       └── performance_report.html
│
├── 📂 logs/                            # سجلات التطبيق
│   ├── app.log
│   └── prediction_log.csv
│
├── 📂 uploads/                         # الصور المرفوعة (مؤقت)
│   └── 📂 temp/
│
├── 📄 config.yaml                      # إعدادات المشروع
├── 📄 requirements.txt                 # المكتبات المطلوبة
├── 📄 .env.example                     # مثال للمتغيرات البيئية
├── 📄 .gitignore                       # الملفات المستثناة من GitHub
├── 📄 Dockerfile                       # حاوية Docker
├── 📄 docker-compose.yml               # تشغيل الخدمات
└── 📄 README.md                        # وثائق المشروع (هذا الملف)

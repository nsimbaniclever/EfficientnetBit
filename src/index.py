# labo1.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

# ---------------------------
# Конфигурация пользователя (ИСПРАВЛЕННАЯ)
# ---------------------------
SAMPLE_SIZE = 10000  # Увеличено для лучших результатов
VAL_SIZE = 2000  # Размер валидации
TEST_SIZE = 2000  # Размер теста
IMG_SIZE = (128, 128)  # Менее агрессивное увеличение размера
BATCH_SIZE = 32
NUM_CLASSES = 10
FE_EPOCHS = 8  # Больше эпох для FE
FT_EPOCHS = 6
FE_LR = 3e-4  # Learning rate ИСПРАВЛЕН
FT_LR = 1e-5  # Ниже для fine-tuning
FINE_TUNE_LAYERS = 30  # Более консервативно
OUTPUT_DIR = "outputs_labo"

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)
tf.random.set_seed(42)

print("Исправленная конфигурация активирована...")

# ---------------------------
# 1) Загрузка данных С ВАЛИДАЦИЕЙ
# ---------------------------
print("\n Загрузка данных CIFAR-10...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = [
    "самолет",
    "автомобиль",
    "птица",
    "кошка",
    "олень",
    "собака",
    "лягушка",
    "лошадь",
    "корабль",
    "грузовик",
]

# Использовать больше данных
x_train = x_train[:SAMPLE_SIZE]
y_train = y_train[:SAMPLE_SIZE]
x_test = x_test[:TEST_SIZE]
y_test = y_test[:TEST_SIZE]

# Разделение train/validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

print(
    f" Данные: Train {x_train.shape[0]}, Val {x_val.shape[0]}, Test {x_test.shape[0]}"
)

# ---------------------------
# 2) Предобработка ИСПРАВЛЕНА для EfficientNet
# ---------------------------
print("\n Специальная предобработка для EfficientNet...")


def preprocess_efficientnet(images):
    """Специфическая предобработка для EfficientNet"""
    images = tf.image.resize(images, IMG_SIZE)
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    return images


# Применить предобработку
x_train_processed = preprocess_efficientnet(x_train)
x_val_processed = preprocess_efficientnet(x_val)
x_test_processed = preprocess_efficientnet(x_test)

# One-hot encoding
y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# Datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train_processed, y_train_cat))
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val_processed, y_val_cat))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test_processed, y_test_cat))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------------------------
# 3) Аугментация данных КРИТИЧЕСКИ ВАЖНА
# ---------------------------
print("\n Создание аугментации данных...")

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)


# ---------------------------
# 4) Вспомогательные функции для outputs
# ---------------------------
def save_training_plots(history_fe, history_ft, model_name):
    """Сохраняет графики обучения"""
    # Объединить истории
    acc = history_fe.history["accuracy"] + history_ft.history["accuracy"]
    val_acc = history_fe.history["val_accuracy"] + history_ft.history["val_accuracy"]
    loss = history_fe.history["loss"] + history_ft.history["loss"]
    val_loss = history_fe.history["val_loss"] + history_ft.history["val_loss"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Точность обучения", linewidth=2)
    plt.plot(val_acc, label="Точность валидации", linewidth=2)
    plt.axvline(
        x=len(history_fe.history["accuracy"]) - 1,
        c="r",
        linestyle="--",
        label="Начало Fine-Tuning",
        alpha=0.7,
    )
    plt.title(f"{model_name} - Точность")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Потери обучения", linewidth=2)
    plt.plot(val_loss, label="Потери валидации", linewidth=2)
    plt.axvline(
        x=len(history_fe.history["loss"]) - 1,
        c="r",
        linestyle="--",
        label="Начало Fine-Tuning",
        alpha=0.7,
    )
    plt.title(f"{model_name} - Потери")
    plt.xlabel("Эпоха")
    plt.ylabel("Потери")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"training_curves_{model_name.lower()}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f" Графики сохранены: training_curves_{model_name.lower()}.png")


def evaluate_and_save_results(
    model, x_test, y_test, y_test_cat, class_names, model_name
):
    """Оценивает модель и сохраняет все результаты"""
    print(f"\n Оценка {model_name}...")

    # Базовые метрики
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f" {model_name} -> Потери: {test_loss:.4f}, Точность: {test_accuracy:.4f}")

    # Предсказания
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_cat, axis=1)

    # Classification Report
    report = classification_report(
        y_true_classes, y_pred_classes, target_names=class_names, digits=4
    )
    report_path = os.path.join(
        OUTPUT_DIR, f"classification_report_{model_name.lower()}.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    print(f" Отчет сохранен: {report_path}")

    # Матрица ошибок
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Матрица ошибок - {model_name}", fontsize=14, pad=20)
    plt.ylabel("Истинные метки")
    plt.xlabel("Предсказания")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Матрица ошибок: {cm_path}")

    return test_loss, test_accuracy, report, cm


# ---------------------------
# 5) EfficientNetB0 ИСПРАВЛЕННЫЙ
# ---------------------------
print("\n Построение исправленного EfficientNetB0...")


def create_efficientnet_model():
    """Создает модель EfficientNet с аугментацией данных"""
    # Базовая модель
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False

    # Архитектура с аугментацией
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)  # АУГМЕНТАЦИЯ ДАННЫХ
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    # CORRECTION: Utiliser un nom ASCII pour TensorFlow
    model = models.Model(inputs, outputs, name="efficientnet_corrected")
    return model, base_model


# Создание модели
efficient_model, efficient_base = create_efficientnet_model()

# Компиляция с исправленным LR
efficient_model.compile(
    optimizer=optimizers.Adam(learning_rate=FE_LR),  # ИСПРАВЛЕНО
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print(" Сводка EfficientNet:")
efficient_model.summary()

# Callbacks
efficient_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
    ),
    callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_efficientnet_fe.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
    ),
]

# ОБУЧЕНИЕ FE
print("\n EfficientNet - Feature Extraction...")
start_time = time.time()
history_e_fe = efficient_model.fit(
    train_ds,
    epochs=FE_EPOCHS,
    validation_data=val_ds,
    callbacks=efficient_callbacks,
    verbose=1,
)
time_e_fe = time.time() - start_time

# FINE-TUNING
print("\n EfficientNet - Fine-Tuning...")
efficient_base.trainable = True
for layer in efficient_base.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

# Перекомпиляция с очень низким LR
efficient_model.compile(
    optimizer=optimizers.Adam(learning_rate=FT_LR),  # ОЧЕНЬ НИЗКИЙ
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks FT
ft_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True
    ),
    callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_efficientnet_ft.h5"),
        monitor="val_accuracy",
        save_best_only=True,
    ),
]

start_time = time.time()
history_e_ft = efficient_model.fit(
    train_ds,
    epochs=FE_EPOCHS + FT_EPOCHS,
    initial_epoch=history_e_fe.epoch[-1],
    validation_data=val_ds,
    callbacks=ft_callbacks,
    verbose=1,
)
time_e_ft = time.time() - start_time

# ФИНАЛЬНОЕ СОХРАНЕНИЕ
efficient_model.save(os.path.join(OUTPUT_DIR, "efficientnet_final.h5"))

# ---------------------------
# 6) ОЦЕНКА EFFICIENTNET
# ---------------------------
print("\n Оценка EfficientNet...")

# Графики обучения
save_training_plots(history_e_fe, history_e_ft, "EfficientNetB0")

# Детальные результаты
e_loss, e_acc, e_report, e_cm = evaluate_and_save_results(
    efficient_model,
    x_test_processed,
    y_test,
    y_test_cat,
    class_names,
    "EfficientNet_FT",
)

# ---------------------------
# 7) СРАВНЕНИЕ И ФИНАЛЬНЫЙ ОТЧЕТ
# ---------------------------
print("\n Генерация финального отчета...")

# Сводный отчет
summary_path = os.path.join(OUTPUT_DIR, "FINAL_REPORT.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== ОТЧЕТ ЛАБОРАТОРНАЯ 1 - ИСПРАВЛЕННАЯ ВЕРСИЯ ===\n\n")
    f.write("КОНФИГУРАЦИЯ:\n")
    f.write(f"- Размер изображения: {IMG_SIZE}\n")
    f.write(
        f"- Обучающая: {x_train.shape[0]} | Валидация: {x_val.shape[0]} | Тест: {x_test.shape[0]}\n"
    )
    f.write(f"- Learning Rates: FE={FE_LR}, FT={FT_LR}\n")
    f.write(f"- Аугментация данных: ДА\n\n")

    f.write("РЕЗУЛЬТАТЫ EFFICIENTNET:\n")
    f.write(f"- Финальная точность: {e_acc:.4f}\n")
    f.write(f"- Финальные потери: {e_loss:.4f}\n")
    f.write(f"- Время FE: {time_e_fe:.1f}с | Время FT: {time_e_ft:.1f}с\n")
    f.write(f"- Параметры: {efficient_model.count_params()}\n\n")

    f.write("ПРИМЕНЕННЫЕ УЛУЧШЕНИЯ:\n")
    f.write("1. Уменьшены learning rates (1e-4/1e-5 вместо 1e-3)\n")
    f.write("2. Добавлена аугментация данных\n")
    f.write("3. Специфический препроцессинг для EfficientNet\n")
    f.write("4. Больше обучающих данных\n")
    f.write("5. Чистое разделение train/val/test\n")
    f.write("6. Early stopping и callbacks\n")

print(f"ФИНАЛЬНЫЙ ОТЧЕТ: {summary_path}")

# ---------------------------
# 8) ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ
# ---------------------------
print("\nГенерация визуализации предсказаний...")

# Примеры правильных/неправильных предсказаний
y_pred = efficient_model.predict(x_test_processed[:20], verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat[:20], axis=1)

plt.figure(figsize=(15, 8))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[i].astype("uint8"))  # Оригинальное изображение
    pred_class = class_names[y_pred_classes[i]]
    true_class = class_names[y_true_classes[i]]
    color = "green" if y_pred_classes[i] == y_true_classes[i] else "red"
    plt.title(
        f"Истина: {true_class}\nПредсказание: {pred_class}", color=color, fontsize=10
    )
    plt.axis("off")

plt.suptitle("Примеры предсказаний (Зеленый=Правильно, Красный=Ошибка)", fontsize=14)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "predictions_examples.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print("Примеры предсказаний: predictions_examples.png")

# ---------------------------
# 9) ФИНАЛЬНОЕ РЕЗЮМЕ
# ---------------------------
print("\n" + "=" * 50)
print("ЛАБОРАТОРНАЯ 1 ЗАВЕРШЕНА - ИСПРАВЛЕННАЯ ВЕРСИЯ")
print("=" * 50)
print(f"ВСЕ РЕЗУЛЬТАТЫ В: {OUTPUT_DIR}/")
print(f"Точность EfficientNet: {e_acc:.4f}")
print(f"Общее время: {(time_e_fe + time_e_ft)/60:.1f} мин")
print("Графики, матрицы, отчеты сгенерированы!")
print("=" * 50)

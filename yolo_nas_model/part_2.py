import os
from super_gradients.training import Trainer, dataloaders, models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
# from super_gradients.training.dataloaders import DetectionDataLoader

# Параметры для сохранения чекпоинтов
CHECKPOINT_DIR = '/home/user/MoleScane/checkpoints'
trainer = Trainer(experiment_name='my_second_yolonas_run', ckpt_root_dir=CHECKPOINT_DIR)

# Параметры датасета
dataset_params = {
    'data_dir': '/home/user/MoleScane/dataset',
    'train_images_dir': 'images',     # Путь к папке с изображениями
    'train_labels_dir': 'labels',     # Путь к папке с аннотациями
    'val_images_dir': 'images/valid',       # Используем те же изображения для валидации
    'val_labels_dir': 'labels/valid',       # Используем те же аннотации для валидации
    'classes': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']  # Классы из GroundTruth.csv
}

# DataLoader для тренировочного набора
train_data = dataloaders.coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': 16,
        'num_workers': 4
    }
)

# DataLoader для валидационного набора
val_data = dataloaders.coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': 16,
        'num_workers': 4
    }
)

# Инициализация модели для обучения
model = models.get(
    'yolo_nas_l',
    num_classes=len(dataset_params['classes']),
    pretrained_weights="coco"
)

# Параметры обучения
train_params = {
    'silent_mode': False,                # Отключаем silent_mode, чтобы видеть вывод
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": 10,                    # Задаем количество эпох
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "train_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}

# Запуск тренировки
trainer.train(
    model=model,
    training_params=train_params,
    train_loader=train_data,
    valid_loader=val_data
)

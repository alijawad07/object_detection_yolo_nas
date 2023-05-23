import argparse
import os
import yaml
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models import get
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback


parser = argparse.ArgumentParser(description='Training script for PP-YOLO-E model on custom dataset.')
parser.add_argument('--data', type=str, default='data.yaml',
                    help='Path to the data configuration file (default: data.yaml)')
parser.add_argument('--name', type=str, required=True,
                    help='Name of the experiment')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training and validation (default: 4)')
parser.add_argument('--num_workers', type=int, default=2,
                    help='Number of workers for data loading (default: 2)')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs for training (default: 30)')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = 'checkpoints'
MODEL_ARCH = 'yolo_nas_l'

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

trainer = Trainer(experiment_name=args.name, ckpt_root_dir=CHECKPOINT_DIR)

with open(args.data) as f:
    data_config = yaml.load(f, Loader=yaml.FullLoader)

dataset_params = {
    'data_dir': data_config['dataset'],
    'train_images_dir': f'{data_config["train"]}/images',
    'train_labels_dir': f'{data_config["train"]}/labels',
    'val_images_dir': f'{data_config["val"]}/images',
    'val_labels_dir': f'{data_config["val"]}/labels',
    'test_images_dir': f'{data_config["test"]}/images',
    'test_labels_dir': f'{data_config["test"]}/labels',
    'classes': data_config['names']
}

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes'],
    },
    dataloader_params={
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }
)

train_data.dataset.transforms

model = get('yolo_nas_l',
            num_classes=len(dataset_params['classes']),
            pretrained_weights='coco')

#optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)

lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=5 * len(train_data), anneal_strategy='cos', pct_start=0.2)

train_params = {
   'silent_mode': False,
    "average_best_models":True,
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
    "max_epochs": args.epochs,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    
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

trainer.train(model=model, 
              training_params=train_params, 
              train_loader=train_data, 
              valid_loader=val_data)

print('Training finished successfully')


#evaluation on test data

best_model = models.get(
    MODEL_ARCH,
    num_classes=len(dataset_params['classes']),
    checkpoint_path=f"{CHECKPOINT_DIR}/{args.name}/average_model.pth"
).to(device)

trainer.test(
    model=best_model,
    test_loader=test_data,
    test_metrics_list=DetectionMetrics_050(
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
)

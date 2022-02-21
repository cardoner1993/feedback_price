import torch

config = dict()
config['competition'] = "feedback-prize-2021"
config['name'] = "Longformer-Baseline"
config['debug'] = False
config['inference_only'] = False

config['model_name'] = "allenai/longformer-base-4096"
config['max_length'] = 1024
config['hidden_size'] = 768

config['output_labels'] = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

config['labels_to_ids'] = {v:k for k,v in enumerate(config['output_labels'])}
config['ids_to_labels'] = {k:v for k,v in enumerate(config['output_labels'])}

config['n_fold'] = 10
config['trn_fold'] = [0]
config['seed'] = 2022

config['max_epochs'] = 8
config['gradient_clip_val'] = 100
config['accumulate_grad_batches'] = 1
config['early_stopping'] = False

config['optimizer'] = dict(
    optimizer="AdamW", 
    lr=2e-5, 
    weight_decay=1e-5
    )

config['scheduler'] = dict(
    interval = "epoch",
    scheduler = "MultiStepLR",
    # Epochs where the LR is updated.
    milestones = [2, 3, 4, 5, 6, 7],
    gamma = 0.5
)

config['monitor_metric'] = 'val_f1_score'
config['mode'] = 'max'

config['train_batch_size'] = 4
config['valid_batch_size'] = 4
config['num_workers'] = 4
config['resume_from_checkpoint'] = None

config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# colab_dir = ""
# api_path = colab_dir + "kaggle.json"
# drive_path = colab_dir + "mst8823"
# upload_from_colab = False

# kaggle_dataset_path = None

"""
- step scheduler example
scheduler = dict(
    interval = "step",
    scheduler="get_cosine_schedule_with_warmup",
    num_warmup_steps=256, 
    num_cycles=0.5)

"""

if config['debug']:
    config['train_batch_size'] = 2
    config['valid_batch_size'] = 2
    config['max_epochs'] = 2
    config['max_length'] = 128

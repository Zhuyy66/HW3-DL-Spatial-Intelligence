# Week3 Day6 Topic2 Hyperparameters

| group | parameter | value | source |
| --- | --- | --- | --- |
| model | policy | act | act_calvin_week3_150k.yaml |
| model | vision_backbone | resnet18 | act_default_v0.4.0.dump.txt |
| model | pretrained_backbone_weights | ResNet18_Weights.IMAGENET1K_V1 | act_default_v0.4.0.dump.txt |
| model | dim_model | 512 | act_default_v0.4.0.dump.txt |
| model | n_heads | 8 | act_default_v0.4.0.dump.txt |
| model | n_encoder_layers | 4 | act_default_v0.4.0.dump.txt |
| model | n_decoder_layers | 1 | act_default_v0.4.0.dump.txt |
| model | dim_feedforward | 3200 | act_default_v0.4.0.dump.txt |
| model | use_vae | True | act_default_v0.4.0.dump.txt |
| model | latent_dim | 32 | act_default_v0.4.0.dump.txt |
| model | dropout | 0.1 | act_default_v0.4.0.dump.txt |
| model | kl_weight | 10.0 | act_default_v0.4.0.dump.txt |
| data | A-only episodes/frames | 6089 / 366693 | a_only run_manifest.json |
| data | ABC episodes/frames | 17870 / 1071743 | abc run_manifest.json |
| data | state/action shape | [15] / [7] | act_calvin_week3_150k.yaml |
| data | static/wrist image shape | [200, 200, 3] / [84, 84, 3] | act_calvin_week3_150k.yaml |
| training | optimizer | adamw | act_calvin_week3_150k.yaml |
| training | learning_rate | 1e-05 | act_calvin_week3_150k.yaml |
| training | weight_decay | 0.0001 | act_calvin_week3_150k.yaml |
| training | batch_size | 8 | run_manifest.json |
| training | gradient_steps | 150000 | run_manifest.json |
| training | seed | 20260529 | act_calvin_week3_150k.yaml |
| training | chunk_size | 100 | act_calvin_week3_150k.yaml |
| training | n_action_steps | 100 | act_calvin_week3_150k.yaml |
| training | num_workers | 16 | run_manifest.json |
| training | prefetch_factor | 4 | run_manifest.json |
| training | persistent_workers | True | run_manifest.json |
| training | save_freq | 50000 | run_manifest.json |
| logging | A-only WandB | https://wandb.ai/ares-core-ai/hw3-topic2/runs/cnfwf30s | a_only run_manifest.json |
| logging | ABC WandB | https://wandb.ai/ares-core-ai/hw3-topic2/runs/eysfzlqn | abc run_manifest.json |

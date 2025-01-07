# Risultati

```json
"args": [
    "--epochs", "400",
    "--window_size", "100",
    "--window_delta", "100",
    "--learning_rate", "0.0001",
    "--batch_size", "16",
    "--dropout", "0.2",
    "--temporal_kernel_size", "5",
    "--temporal_layers", "32,16,8",
    "--spatial_layers", "8",
    "--joint_features", "6",
    "--scheduler_step", "200"
]
```

Using Kinect V2 joint position.
Trained for 400 epochs.

| Ex | Our | Mourchid<br>et al. | Deb<br>et al. | Song<br>et al. | Zhang<br>et al. | Liao<br>et al. |
|----|-----|-----------------|------------|-------------|--------------|-------------|
|  1 |**0.928**|0.641|0.799|0.977|1.757|1.141|
|  2 |**1.352**|0.753|0.774|1.282|3.139|1.528|
|  3 |**0.846**|0.210|0.369|1.105|1.737|0.845|
|  4 |**0.971**|0.206|0.347|0.715|1.202|0.468|

```json
"r_min":0.8,
"r_max":0.99,
"args": [
                "--epochs",
                "800",
                "--window_size",
                "200",
                "--window_delta",
                "200",
                "--learning_rate",
                "0.001",
                "--weight_decay",
                "0.01",
                "--batch_size",
                "16",
                "--dropout",
                "0.40",
                "--model_dim",
                "32",
                "--temporal_state_dim",
                "32",
                "--temporal_layers",
                "2",
                "--spatial_layers",
                "2",
                "--joint_features",
                "6",
                "--lru_radius_min",
                "0.8",
                "--lru_radius_max",
                "0.99",
                "--scheduler_step",
                "300",
                //"--dashboard"
            ]
```

1.850 es1
1.570 es2
0.958 es3
1.340 es4


Conv+LRU+gMLP, con bias di selezione...

```json
"r_min":0.60,
"r_max":0.99,
"phase_max":"math.pi",
"args": [
    "--epochs",                 "800",
    "--window_size",            "200",
    "--window_delta",           "200",
    "--learning_rate",          "0.002",
    "--weight_decay",           "0.001",
    "--batch_size",             "10",
    "--dropout",                "0.3",
    "--model_dim",              "32",
    "--temporal_state_dim",     "48",
    "--temporal_layers",        "2",
    "--spatial_layers",         "2",
    "--joint_features",         "6",
    "--scheduler_step",         "300",
    "--temporal_method",        "LRU"
]
```

3.050 es1
3.100 es2
3.670 es3
3.110 es4
3.230 mean

## MIGLIORE IN ASSOLUTO!

```json
{
    'model_dim': 96, 
    'temporal_state_dim': 96, 
    'temporal_layers': 4, 
    'spatial_layers': 8, 
    'dropout': 0.70, 
    'batch_size': 12, 
    'learning_rate': 0.002, 
    'weight_decay': 0.001, 
    'maximum_quality': 50.0, 
    'joint_count': 19, 
    'joint_features': 6, 
    'scheduler_step': 100, 
    'temporal_method': 'GRU', 
    'spatial_method': 'gMLP', 
    'no_conv': False, 
    'epochs': 200
}
```

'aggregated_mean': 4.854, 
'parameters': 615745, 
'ex1_mean': 4.040, 
'ex2_mean': 6.224, 
'ex3_mean': 4.616, 
'ex4_mean': 5.051, 
'ex5_mean': 4.337
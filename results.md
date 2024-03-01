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
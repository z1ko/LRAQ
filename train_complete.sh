#/bin/bash
python train_complete.py 		\
	--epochs 200 				\
	--learning_rate 0.005 		\
	--weight_decay 0.001 		\
	--batch_size 10 			\
	--dropout 0.60 				\
	--model_dim 80 				\
	--temporal_state_dim 200 	\
	--temporal_layers 6 		\
	--spatial_layers 8 			\
	--joint_features 6			\
	--scheduler_step 50		\
	--spatial_method gMLP 		\
	--temporal_method LSTM

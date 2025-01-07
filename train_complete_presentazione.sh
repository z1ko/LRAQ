#/bin/bash
python train_complete.py 		\
	--epochs 200 				\
	--learning_rate 0.001 		\
	--weight_decay 0.001 		\
	--batch_size 10 			\
	--dropout 0.6 				\
	--model_dim 48 				\
	--temporal_state_dim 48 	\
	--temporal_layers 3 		\
	--spatial_layers 2 			\
	--joint_features 6			\
	--scheduler_step 250		\
	--spatial_method gMLP 		\
	--temporal_method GRU

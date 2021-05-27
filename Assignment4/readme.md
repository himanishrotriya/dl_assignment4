# Link of wandb report: 
https://wandb.ai/himani-shrotriya/assignment_4/reports/Deep-Generative-Models-Assignment-4--Vmlldzo3MjEwOTk

# contrastive_divergence_a4.py
  Contains code for contrastive divergence without sweep setup.

1) To run rbm for a particular hyper-parameter configuration , set the parameters while calling train in the last line :
    `train(num_hidden_nodes=num_hidden_nodes,num_visible_nodes=num_visible_nodes,learning_rate=learning_rate,num_epochs=num_epochs,k=k,drop_probability=0)`

2) The line which creates an instance of the classifier  is :  
    `classifier = FeedForwardNN(hidden_layer_size=rbm.num_hidden_nodes,output_layer_size=10,
                                   input_layer_size=rbm.num_hidden_nodes, learning_rate=0.001, num_epochs=10,
                                   optimisation="adam", batch_size=64,
                                   activation_function="lrelu",drop_probability=drop_probability)`
                                   
     i) Parameter: `optimisation` can take following values:
      1. adam
      2. nadam
      3. nesterov
      4. rmsprop

      ii) Parameter: `activation_function` can take following values:
      1. relu
      2. lrelu
      3. selu
  
     iii) The below line will train the classifier once the desired hyperparameters are set:    
             `classifier.fit(classifier_input, y_val, test_input, y_test, alpha)`
      where alpha is weight decay parameter which we can set to any desired value.

      iv) To add dropout in the classifier,set drop_probability in the below line:  
          `train(num_hidden_nodes=num_hidden_nodes,num_visible_nodes=num_visible_nodes,learning_rate=learning_rate,num_epochs=num_epochs,k=k,drop_probability=0)`
  
  # cd_sweep_a4.py
  This file contains code for contrastive divergence with sweep set up.
  # gibbs_sampling_a4.py
  1) set the hyperparameters for gibbs sampling in the line:  
      `rbm = RBMGibbsSampling(num_visible_nodes=num_visible_nodes, num_hidden_nodes=num_hidden_nodes, k=k, r=r,
                       learning_rate=learning_rate, num_epochs=num_epochs)`  

2) set the hyper-parameters for classifier in the line:  
    `classifier = FeedForwardNN(hidden_layer_size=256, output_layer_size=10, input_layer_size=rbm.num_hidden_nodes,
                               learning_rate=0.001, num_epochs=10, optimisation="adam", batch_size=64,activation_function="lrelu",drop_probability=0)`  
                               
    i) The below line will train the classifier once the desired hyperparameters are set:  
         `classifier.fit(classifier_input, y_val, test_input, y_test, alpha)`
  where alpha is weight decay parameter which we can set to any desired value.
   
   
# gs_sweep_a4.py
Contains code for gibbs sampling with seep stup

# visulaisation.ipynb

1) You only need to set the hyper-parameters for rbm in the line :  
    `rbm=RBMContrastiveDivergence(num_visible_nodes=num_visible_nodes,num_hidden_nodes=num_hidden_nodes,k=k,learning_rate=learning_rate,num_epochs=num_epochs)`

2) `samples` conatin the generated samples after every (m/64) steps
3) `true_images` contain the true_images after every (m/64) steps
4) `test_hidden_representation` contains the n-dimensional hidden representation of the test data

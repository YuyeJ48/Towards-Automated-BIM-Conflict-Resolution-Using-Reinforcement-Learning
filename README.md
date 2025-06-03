# Towards Automated BIM Conflict Resolution Using RL

This code base contains implementation of the paper: [Towards Automated BIM Conflict Resolution Using Reinforcement Learning](https://mediatum.ub.tum.de/doc/1781883/1781883.pdf) This paper combines Gym-based RL environment and Solibri APIs to train an agent in real BIM environment to solve conflicts in IFC model. The PPO algorithm is the main focus, but DQN is also included, both algorithms are employed from the Stable-Baselines3.

## Installation

### 1. Install the Solibri Java API:

Unzip view-examples.zip, then follow the [official Java API instruction](https://solibri.github.io/Developer-Platform/latest/getting-started.html) from Solibri to maven install the folder. 

Please note:

1. Please change the _smc_dir_ variable in the _pom.xml_ file according to your Solibri installation folder
   ```json
   	<properties>
		<!-- Solibri installation path on Windows -->
		<smc-dir> D:/Program/solibri/SOLIBRI</smc-dir>
   ```
2. Change the _outputFilePath_ variable in the _CheckingResultsExporterView.java_ file if necessary. This variable is where the exported csv file, which contains the information of detected issues, will be saved. And the same location should be applied in the following training.
   ```json
      private void exportResults() {
		Collection<Result> results = SMC.getChecking().getResults();
		LOG.info("Results size: {}", results.size());
		String outputFilePath = "D:/Thesis/5.IFCworkspace/Solibri_JavaAPI/checking_results/checking_results.csv";
   ```

After the successful installtion, lauch the Solibri software. If the installation is successful, a _CHECKING EXPORTER_ should appear in _VIEWS_ tab. Click it and a window will pop up with the _Export automatically_ option selected. 
### 2. Install Python libraries.

Use conda to install all the necessary libraries, including [Gym framework](https://gymnasium.farama.org/), [IFcOpenSell](https://ifcopenshell.org/), [Stable-Baselines3](https://ifcopenshell.org/) and so on.

```bash
conda env create -f env.yml
```
*Please install the v0.7.0-6c9e130ca version of ifcopenshell

## Training

### Lauch Solirbi using REST API
The Solibri Software should be run using its REST API, please follow the [official REST API instrunction](https://solibri.github.io/Developer-Platform/latest/RestApiUsage.html) for lauching Solibri. Please also pay attention to the location of the Solirbi installation directory here.

The default port is 10876, if you choose to use a different one, please replace it all in the code.

After the Solirbi is successfully opened, sign in with a Solibri Office account. Then open the corresponding smc model you would like to train. The required rulesets and classifications are prepared.

### Adjust Customized Parameters in Code
Please go through the code and replace all the path parameter accordingly.
1. This is where you save the training model, x_train.ifc will be created constantly during the training.
```bash 
self.destination_path = 'D:\\Thesis\\5.IFCworkspace\\Model\\AC20-FZK-Haus_toilet_train.ifc'
```
2. This is where you save the original AC20-FZK-Haus_x.ifc model from this branch, this ifc remains untouch during the whole training process.
```bash 
source_path = 'D:\\Thesis\\5.IFCworkspace\\Model\\AC20-FZK-Haus_toilet.ifc'
```
3. This is the same csv path for the _outputFilePath_ variable in the Installation chapter.
```bash 
file_path = 'D:/Thesis/5.IFCworkspace/Solibri_JavaAPI/checking_results/checking_results.csv'
```
4. This is where you save the original x_AC20-FZK-Haus.smc file from this branch.
```bash 
smc_file_path = r"D:\Thesis\5.IFCworkspace\NEW\Toilet-Wall_AC20-FZK-Haus.smc"
```

### Start Training
Before the training, turn on this part of code, and set all the code afterwards as comment to test the environment first.
```bash 
# # for testing the env
# env = GymEnv_IFC_Toilet()
# check_env(env, warn=True, skip_render_check=True)
# observation, info = env.reset()
# for _ in range(10):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     print("action:", action)
#     print("observation:", observation)
#     print("reward:", reward)
#     print("terminated:", terminated)
#     print("truncated:", truncated)
#     print("info:", info)
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()
```

If the test is successful, restore the commented state of the code and start the training.
What you can train: 
1. In the code, there is two different algorithm options, PPO is the priority, if the training is successful, then you can go with the DQN.
   
```bash
#choose one algorithm
# train with PPO
model = PPO("MlpPolicy", env, learning_rate=0.0001, n_steps=128, batch_size=128, n_epochs=4, verbose=1, tensorboard_log=save_path + "tensorboard/", device="cpu")
model.learn(total_timesteps=8192, callback=callback, progress_bar=True)
model.save(save_path + "final_model/")
env.close()

# #train with DQN
# model = DQN("MlpPolicy", env, batch_size=128,buffer_size=2048,gamma=0.99,learning_starts=128,learning_rate=0.00063,target_update_interval=64,train_freq=4,gradient_steps=-1,exploration_fraction=0.5, exploration_final_eps=0.1,verbose=1, tensorboard_log=save_path + "dqn_tensorboard/", device="auto")
# model.learn(total_timesteps=16384, callback=callback, progress_bar=True)
# model.save(save_path + "final_model/")
# env.close()
```
2. You can increase the total_timesteps if the training works well.
3. If possible, you can try to tune the hyperparameters, especially the learning_rate(maybe try 0.0005 or 0.001), to see if it improves the results.

*All training data will be saved in a folder named _databank_ in the same location of the main code, including final model, best performance model, model after different intervals alongside with different training and evaluation data.

### TODO
- We adopted a strategy of restarting Solibri every 256 steps to reduce its memory consumption and speed up training. This strategy was effective in the early stages of training, but as the number of restarts increased, Solibri checks became very slow.
- In `Toilet-Wall\Toilet-Wall_RL_Code_250317.py`, we explored 
	1. using a smaller observation space
	2. relative normalized coordinates
	3. introducing intermediate rewards
	4. optimizing the reward logic
	5. changing the network dimension of the policy MLP
	6. reducing the randomness of the initialization position

	The goal is to accelerate the convergence speed of training. However, the memory usage of Solibri increase significantly with the number of training steps, making training still very inefficient.

## Citation
```
@inproceedings{Jiang:2025:RL4BIMConflict,
	author = {Jiang, Y. and Du, C. and Wu, J. and Nousias, S. and Borrmann, A.},
	title = {Towards Automated BIM Conflict Resolution Using Reinforcement Learning},
	booktitle = {Proc. of the 32nd Int. Workshop on Intelligent Computing in Engineering (EG-ICE)},
	year = {2025}
}
```

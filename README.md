# Towards-Automated-BIM-Conflict-Resolution-Using-Reinforcement-Learning

This code combines Gym-based RL environment and Solibri APIs to train an agent in real BIM environment to solve the conlicts in IFC model. The PPO algorithm is the main focus, but DQN was also included, both algorithms employed from the Stable-Baselines3.

## Installation

### 1. Install the Solibri Java API:

Follow the [official Java API instruction](https://solibri.github.io/Developer-Platform/latest/getting-started.html) from Solibri to maven install the folder _Solibri_JavaAPI_. 

Please note:

1. If the Solibri installtion folder is not the default one, the _smc_dir_ variable need to be updated in the _pom.xml_ file. 
2. Change the _outputFilePath_ variable in the _CheckingResultsExporterView.java_ file if necessary. This variable is where the exported csv file which contains the information for detected issues will be saved. And the same location should be applied in the following training.

After the installtion, lauch the Solibri software. If the installation is successful, a _CHECKING EXPORTER_ should appear in _VIEWS_ tab. Click it and a window will pop up with the _Export automatically_ option selected. 
### 2. Install Python libraries.

manager [pip](https://pip.pypa.io/en/stable/) to install all the necessary libraries, including [Gym framework](https://gymnasium.farama.org/), [IFcOpenSell](https://ifcopenshell.org/), [Stable-Baselines3](https://ifcopenshell.org/) and so on.

```bash
pip install csv
pip install os
pip install sys
pip install shutil
pip install numpy
pip install requests
pip install pandas

pip install gymnasium

pip install ifcopenshell
pip install ifcopenshell.api
pip install ifcopenshell.geom
pip install ifcopenshell.util

pip install stable_baselines3
pip install stable_baselines3.common
```

## Training

### Lauch Solirbi using REST API
The Solibri Software should be run using its REST API, please follow the [official REST API instrunction](https://solibri.github.io/Developer-Platform/latest/RestApiUsage.html) for lauching Solibri. Please also pay attention to the location of the Solirbi installation directory here.

After the Solirbi is successfully opened, sign in with a Solibri Office account. Then open the smc model that you want to use as training environment and adjust all required rulesets and classifications if necessary. 

### Adjust Customized Parameters in Code
1. The UUID of the current solibri model:
   
    You can follow the official instrunction to call the _GET/models_ request to do so using Swagger UI or run the following code:
    ```python
    import requests
    solibri_server_url = 'http://localhost:10876/solibri/v1'
    url = f'{solibri_server_url}/models'
    response = requests.get(url, headers={'accept': 'application/json'})
    print(response)
    response_body = response.json()
    print(response_body)
    ```
   The UUID of the model should then be copied to the main code as the variable _modelUUID_ in the the function __update(self)_ 

2. The path of the csv file
   
   The same path for the _outputFilePath_ variable in the Installation chapter should be used again for the variable _file_path_ in the __current_results(self)_ function in the main code.

3. The file location of the IFC model

     The last thing need to be customized in the Python code before training the model is the file location of the IFC model in the function _reset(self, seed=None)_.
     The variable _source_path_ is the file path of the original IFC model while the _self.destination_path_ is the file path of the copied IFC model for training. The latter one will be modified continuously during the training while the original one remains untouched.

### Start Training
Tune the hyperparameters if you want to, set the code after the comment _# for testing the model_ to be commented, and start training:
```python
# for testing the model
env = GymEnv_IFC_AirTerminal()
observation, info = env.reset()
...
else:
    env.close()
```
All training data will be saved in a folder named _databank_ in the same location of the main code, including final model, best performance model, model after different intervals alongside with different training and evaluation data.

## Test the Model
For testing the trained RL model, set the code between _# for testing the env_ and _# for testing the model_ to be commented and run the code:

```python
# for testing the env
env = GymEnv_IFC_Toilet()
check_env(env, warn=True, skip_render_check=True)
...
# for testing the model
```

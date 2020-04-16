# Aida-Gym training environment

This is a fork from Aida Gym environment created in order to create tools to make Aida robot learn to walk.
We implemented Stable-baselines support in order to use their proven-to-work algorythms.
So far we didn't manage to make the robot walk in a decent manner, try it yourself and let us know if you can.

## Installation

First clone the repo

``` Python
git clone url
```
Then you need to install a few libraries, using pip and a [virtualenv](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html) is recommended since stable baselines doesn't work with TensorFlow 2.0


``` Python
pip install tensorflow==1.15.0rc2
pip install gym
pip install pybullet
pip install stable-baselines
pip install imageio
pip install dash
pip install dash_core_components
pip install dash_html_components
pip install dash_renderer
pip install pip install dash-bootstrap-components
```

If you get an missing package error, use pip install and the name of the package to get it to work.

## Usage

### Training
Using your virtualenv, go to aida_gym folder and type :
``` Python
python3 train.py --name NAME --algo ppo2
```
This will launch a training sessions, their are plenty of options you can use, type `python3 train.py --help` to kown more about them. Here we wrote the requiered ones : a name and the algorithm used (choose between ppo2, sac, and a2c). It is really easy to add more if you will.

The directories are set up so that each script knows were to look. The folder resultat contains the zip of the saved models and a gif created at the end of the simulation (more gifs can be found in server/assets) but you can also make your one gifs (see monitoring section)

### Monitoring

Using tensorboard (still in aida_gym folder):
``` Python
tensorboard --logdir ./log
```
It will launch a tensorboard server on port 6006 of your local machine

Using our custom made monitoring server (which will display GIFs and let you ajust several parameters), go to server folder
``` Python
python3 app2.py
```
It will launch a server on port 8050.

To create your own gif of a saved model you can type (in aida_gym folder) :
``` Python
python3 makegif.py --name NAME
```
Since the directories are well made, you don't need to specify the location of the zip model, it will know were to look just using the name. The gif is saved in the video folder of the model.

### Optimizing

In order to optimize the hyperparameters, you can use the opti.py scripts : 
``` Python
python3 optiPPO2.py
```
This will launch a series of simulations with different hyperparameters, log them into tensorboard (lodgir ./optimisation/logOPTI). You can also gather the results from the sqlite database that is created in order to visualize them on Excel. This optimisation uses Optuna, features TPE sampler and Median pruning.

## Note on using distant computer

You can of course launch all those commands via SSH and get the resulting files using FTP on a distant computer. The idea is to free your personnal computer from heavy computation while you need it for your daily activities.
A great tip is to create a [SSH tunnel](https://www.abyssproject.net/2015/05/tunnel-ssh-et-port-forwarding-avec-putty-sous-windows/) in order to access the servers, for example you can map your localhost:8001 port to the distant localhost:8050 in order to access the monitoring server from a different network.

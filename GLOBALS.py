import yaml
import logging

from uuid import getnode as get_mac
mac = get_mac()


# params_fn = r"C:\Users\e2p2\git\HebLingStudy\exp_params.yml"

if mac == 62438188043965 or mac == 202351123943195:
    params_fn = "/Users/ExpRoom1/Documents/OrenKobo/Final_codes/HebLingStudy/exp_params.yml"

if mac in [62438188057035,95474511659208] :
    params_fn = "/Users/orenkobo/Desktop/PhD/Aim1/Aim1_New/exp_params.yml"
    # params_fn = "/Users/orenkobo/Desktop/PhD/HebLingStudy/exp_params.yml"
# params_fn = "/Users/ExpRoom1/Documents/OrenKobo/Final_codes/HebLingStudy/exp_params.yml"


if mac == 57407536525872:
    params_fn = "/Users/schonberglab_laptop1/Documents/OrenKobo/repos/HebLingStudy/exp_params.yml"
    comp = "Laptop1"


with open(params_fn, 'r') as f:
    params = yaml.safe_load(f)
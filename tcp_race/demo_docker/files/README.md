# ridersai_f1tenth_gym

This is Stony Brook University's submission to the riders.ai challenge at IROS (and maybe after that), using the f1tenth gym.

To run, first install the gym (see the old readme). Note we use a slightly custom version of the gym enviornment where you can have a render callback ect.

set the current working directory to "./pkg/src", then run `python3 -m pkg.main_gap` or `python3 -m pkg.main_unc` (depending on the controller you want). These run the tuning code, in the appropriate main file (see the pkg/src/main_gap.py or /pkg/src/main_unc.py files for implementation details). You can set the map (OBS or regular) in the main files as well. Both of these use pickled gain files. If you want to start from scratch, first delete `gains_gap.pkl` or `gains_unc.pkl`.


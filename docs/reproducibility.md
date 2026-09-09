# Reproducibility

Every run is determined by:

- dimension and boundary condition;
- formatted rule;
- grid dimensions;
- concrete integer seed;
- initial density;
- number of steps; and
- software version and repository commit.

Sending `seed: null` to a session endpoint means “choose a new random seed.”
The response stores and returns the generated integer, so the same initial
state can be recreated later. The CLI requires an integer seed for recorded
experiment runs.

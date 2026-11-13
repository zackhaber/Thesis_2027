# Interface notes

- Classical planners calculate routes dynamically. In Unknown mode, they use
  sensor-limited map knowledge.
  
- For non-CTF Unknown-mode planners, switching on detection requires an
  evader to be visible in a cell explored before the current sensor update.
  Newly observed cells become eligible for detection on a later update.
  
- Capture the Flag MARL is available in the simulation and benchmark panels
  when a matching checkpoint exists for the selected configuration.

- Checkpoint warnings in the panels still indicate missing or mismatched
  training configurations.

- In Capture the Flag, the gold flag marks the flag at home. A purple attacker
  is carrying it. Once the flag is taken, defenders pursue the carrier when
  detected; otherwise, in Unknown mode, they search using their beliefs.

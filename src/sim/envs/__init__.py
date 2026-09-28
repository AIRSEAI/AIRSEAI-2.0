"""Scenes: the robot, and every task's furniture on top of it.

`bimanual_tactile` is the robot plus its tactile pads. Each task module
(`blind_peg_insertion`, `stack_cube`, ...)
subclasses `BimanualTactileSceneCfg` and holds that task's world constants.

Importing this package pulls in Isaac Lab, so it must be done after
`AppLauncher` has started the app.
"""

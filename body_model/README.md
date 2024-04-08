1. Clone the body model code from [EgoEgo](https://github.com/lijiaman/egoego_release/tree/main/body_model).
2. Download [SMPL-H](https://mano.is.tue.mpg.de/login.php) (the extended SMPL+H model).

Overall, the structure of `body_model/` folder should be:
```
body_model/
|--README.md
|--__init__.py
|--body_model.py
|--utils.py
|--smplh/
|----info.txt/
|----LICENSE.txt/
|----female/
|------model.npz
|----male/
|------model.npz
|----neutral/
|------model.npz
```
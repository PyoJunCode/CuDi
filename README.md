# CuDi



## Train





**Train student model**

Modify `config` in train_student.py

You can use pretrained teacher weight in `./weights/` folder.

```
python3 train_student.py -e <experiment_name> --gpus <gpu_num>
```





## Test

use `-t` argument to test inference time only.



**Test student model**

Modify `config` in train_student.py

```
python3 test_student.py <model_ckpt_path>
```





## Todo

- [ ] Checkpoint callback
- [ ] Add evaluation metric in test, predict
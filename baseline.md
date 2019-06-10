1. model_zoo.py
pytorch model_zoo.py 63line Permission deny,need to change path :

```
# cached_file = os.path.join(model_dir, filename) #/home/stage/.torch/models
cached_file = os.path.join('/home/stage/yuan/models', filename)
```

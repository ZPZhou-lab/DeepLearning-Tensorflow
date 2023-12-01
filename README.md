# DeepLearning-Tensorflow
Basic knowledge of deep learning (based on Tensorflow implementation)

## Config
In order to install and deploy the project code correctly, we need to configure the `source` folder of this project. The following are configuration schemes for different usage scenarios. 

1. **Using in a `*.py`  script**
   
    We need to set the `$PYTHONPATH` environment variable in system, suppose `/path` is your parent path of folder `source` (i.e. `/path/source`), then add the new line to the end of your system file `~/.bashrc` or `~/.zshrc`：

    ```bash
    export PYTHONPATH=/path:$PYTHONPATH
    ```

    Save the file and exit the text editor. Restart your terminal or run `source ~/.bashrc` (or `source ~/.zshrc` if you're using Zsh) to apply the changes to your current terminal session.

    Now, you can import and use the `source` package of this project like other Python packages in any `*.py` script or in ternimal.

    ```Python
    from source.code import utils
    ```

2. **Using in a `*.ipynb` Jupyter Notebook in Web browser**



3. **Using in a `*.ipynb` Jupyter Notebook in VSCode**



4. **General Solution**
   
   A common solution is to add the parent path to the `source` package through `sys.path.append` before running other code whether you are writing `*.py` scripts or using Jupyter Notebook. 

   ```Python
   import sys
   sys.path.append("your parent path of source")
   ```

    **The disadvantage of this method is that it is temporary** and must be performed every time you write a new project. 


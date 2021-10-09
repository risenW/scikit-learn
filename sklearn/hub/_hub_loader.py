"""
The :mod:`sklearn.hub.load_model` can be used to save a model to huggingface.co/models.
"""

# Author: Rising Odegua <risingodegua@gmail.com>,
# License: BSD 3 clause

from huggingface_hub import hf_hub_download
from joblib import load
from pickle import load as pload

class HubLoader:
    """Loads a model from huggingface.co/models.

    Parameters
    ----------

    repo_id : str
        The id of the model repository.
    
    filename : str, optional
        The filename of the model. If not specified, the default filename is used.

    serialization_method : str, optional
        The serialization method used to save the model. Supports joblib and pickle. The default is "joblib".

    revision : str, optional
        The revision of the model. If not specified, the latest revision will be used.

    cache_dir : str, optional
        The directory to cache the model. If not specified, the default cache directory will be used.

    Returns
    -------
    Scikit-learn model object
        The model loaded from huggingface.co/models.
                
    """
    def __init__(
        self,
        repo_id,
        filename=None,
        serialization_method="joblib",
        revision=None,
        cache_dir=None
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.serialization_method = serialization_method
        self.revision = revision
        self.cache_dir = cache_dir

    def load(self):
        
        model_path = hf_hub_download(self.repo_id, self.filename, self.revision, self.cache_dir)

        if self.serialization_method == "joblib":
            model = load(model_path)
        elif self.serialization_method == "pickle":
            model = pload(model_path)
        else:
            raise ValueError("serialization_method must be 'joblib' or 'pickle'")
            
        return model
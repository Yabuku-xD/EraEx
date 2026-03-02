import threading
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class LazyLoader(Generic[T]):

    # Initialize class state.
    def __init__(self, factory: Callable[[], T], name: str = ""):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self._factory = factory
        self._name = name
        self._instance = None
        self._lock = threading.Lock()
        self._loaded = False

    # Handle instance.
    @property
    def instance(self) -> T:
        """
        Execute instance.
        
        This method implements the instance step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._instance = self._factory()
                    self._loaded = True
        return self._instance

    # Handle is loaded.
    @property
    def is_loaded(self) -> bool:
        """
        Return whether loaded.
        
        This method implements the is loaded step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return self._loaded

    # Preload this operation.
    def preload(self):
        """
        Execute preload.
        
        This method implements the preload step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        _ = self.instance


# Preload in background.
def preload_in_background(*loaders: LazyLoader):

    # Internal helper to worker.
    """
    Execute preload in background.
    
    This function implements the preload in background step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    def _worker():
        """
        Execute worker.
        
        This function implements the worker step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        for loader in loaders:
            try:
                loader.preload()
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
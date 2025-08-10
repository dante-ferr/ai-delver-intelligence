import multiprocessing as std_mp

SESSION_REGISTRY = {}
REGISTRY_LOCK = std_mp.Lock()

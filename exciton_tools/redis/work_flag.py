from datetime import datetime

import redis


def get_worker_flag(
    host: str, port: str, password: str, work_flag: str, time_threshold: int = 300
) -> bool:
    """Get if there is an existing work flag.

    Args:
        host (str): redis host
        port (str): redis port
        password (str): redis password
        work_flag (str): work flag
        time_threshold (int, optional): time to reset. Defaults to 300.

    Returns:
        bool: if there is an existing worker.
    """
    redis_client = redis.Redis(
        host=host, port=port, db=0, password=password, decode_responses=True
    )
    mytime = redis_client.get(work_flag)
    if mytime is None:
        return False
    else:
        mytime = datetime.fromisoformat(mytime)
        nowtime = datetime.utcnow()
        difftime = nowtime - mytime
        difftime = difftime.seconds
        if difftime > time_threshold:
            redis_client.delete(work_flag)
            return False
        else:
            return True


def set_work_flag(host: str, port: str, password: str, work_flag: str):
    """set work flag.

    Args:
        host (str): redis host
        port (str): redis port
        password (str): redis password
        work_flag (str): work flag
    """
    redis_client = redis.Redis(
        host=host, port=port, db=0, password=password, decode_responses=True
    )
    redis_client.set(work_flag, datetime.utcnow().isoformat())


def delete_work_flag(host: str, port: str, password: str, work_flag: str):
    """delete a work flag.

    Args:
        host (str): redis host
        port (str): redis port
        password (str): redis password
        work_flag (str): work flag
    """
    redis_client = redis.Redis(
        host=host, port=port, db=0, password=password, decode_responses=True
    )
    redis_client.delete(work_flag)

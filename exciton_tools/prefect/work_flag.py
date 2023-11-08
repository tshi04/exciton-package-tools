from typing import Callable

from exciton_tools.redis import delete_work_flag, get_worker_flag, set_work_flag


def decorator_work_flag(myfunction: Callable) -> Callable:
    """a work flag decorator.

    Args:
        myfunction (Callable): the function to pass.

    Returns:
        Callable: decorate function.
    """

    def work_flag_worker(
        redis_host: str,
        redis_port: str,
        redis_password: str,
        work_flag: str,
        time_refresh: int = 300,
    ) -> bool:
        """work flag

        Args:
            redis_host (str): redis host
            redis_port (str): redis port
            redis_password (str): password
            work_flag (str): work flag sign
            time_refresh (int, optional): time to refresh. Defaults to 300.

        Returns:
            bool: exising worker.
        """
        status = get_worker_flag(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            work_flag=work_flag,
            time_refresh=time_refresh,
        )
        if not status:
            set_work_flag(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                work_flag=work_flag,
            )
            myfunction(
                redis_host,
                redis_port,
                redis_password,
                work_flag,
                time_refresh,
            )
            delete_work_flag(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                work_flag=work_flag,
            )
        return status

    return work_flag_worker

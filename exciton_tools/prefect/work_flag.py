from typing import Callable

from exciton_tools.redis import delete_work_flag, get_worker_flag, set_work_flag


def decorator_work_flag(myfunction: Callable):
    def work_flag_worker(
        redis_host: str, redis_port: str, redis_password: str, work_flag: str
    ) -> bool:
        status = get_worker_flag(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            work_flag=work_flag,
        )
        if not status:
            set_work_flag(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                work_flag=work_flag,
            )
            myfunction(redis_host, redis_port, redis_password, work_flag)
            delete_work_flag(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                work_flag=work_flag,
            )
        return status

    return work_flag_worker

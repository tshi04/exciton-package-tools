import logging

from minio import Minio


def download_file_from_minio(
    host: str,
    port: str,
    access_key: str,
    secret_key: str,
    use_ssl: bool,
    bucket_name: str,
    object_name: str,
    file_path: str,
):
    """Download data from Minio.

    Args:
        host (str): minio host
        port (str): minio port
        access_key (str): access key
        secret_key (str): secret key
        use_ssl (bool): use ssl
        bucket_name (str): bucket name
        object_name (str): object name
        file_path (str): local file full path.
    """
    endpoint = f"{host}:{port}"
    minio_client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=use_ssl,
    )
    logging.info(f"Download file from {bucket_name}...")
    minio_client.fget_object(
        bucket_name=bucket_name,
        object_name=object_name,
        file_path=file_path,
    )


def upload_file_to_minio(
    host: str,
    port: str,
    access_key: str,
    secret_key: str,
    use_ssl: bool,
    bucket_name: str,
    object_name: str,
    file_path: str,
):
    endpoint = f"{host}:{port}"
    minio_client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=use_ssl,
    )
    logging.info(f"upload file to {bucket_name}...")
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    minio_client.fput_object(
        bucket_name=bucket_name,
        object_name=object_name,
        file_path=file_path,
    )

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from kubernetes import client, config
from kubernetes.stream import stream
from kubernetes.client.exceptions import ApiException
import sys, time, logging

NAMESPACE = "llm-test"
POD_NAME = "warm-pytorch-worker"
GIT_REPO = "https://github.com/yhjh5302/DNN-Testbed"
WORKDIR = "/root/DNN-Testbed"
TRAIN_SCRIPT = "/root/DNN-Testbed/horovod_test/train.py"


def ensure_warm_pod(**context):
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    conf = context["dag_run"].conf or {}

    image = conf.get("image", "nvcr.io/nvidia/pytorch:25.12-py3")
    gpu = conf.get("gpu", 1)
    cpu = conf.get("cpu", "2")
    memory = conf.get("memory", "8Gi")

    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(
            name=POD_NAME,
            namespace=NAMESPACE,
            labels={"app": "warm-pytorch"}
        ),
        spec=client.V1PodSpec(
            restart_policy="Always",
            containers=[
                client.V1Container(
                    name="worker",
                    image=image,
                    command=["/bin/bash", "-c"],
                    args=["while true; do sleep 3600; done"],
                    resources=client.V1ResourceRequirements(
                        limits={
                            "nvidia.com/gpu": str(gpu),
                            "cpu": cpu,
                            "memory": memory,
                        }
                    ),
                )
            ],
        ),
    )

    try:
        v1.read_namespaced_pod(name=POD_NAME, namespace=NAMESPACE)
        print("Warm pod already exists")
    except ApiException as e:
        if e.status == 404:
            v1.create_namespaced_pod(
                namespace=NAMESPACE,
                body=pod,
            )
            print("Warm pod created")
        else:
            raise


def wait_for_pod_ready():
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    while True:
        pod = v1.read_namespaced_pod(POD_NAME, NAMESPACE)
        phase = pod.status.phase

        if phase == "Running":
            for cs in pod.status.container_statuses or []:
                if cs.ready:
                    return
        if phase in ("Failed", "Unknown"):
            raise RuntimeError(f"Pod in bad state: {phase}")

        time.sleep(5)


def exec_in_warm_pod(cmd: str, **context):
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    full_cmd = f"""
    set -e
    if [ ! -d "{WORKDIR}" ]; then
      git clone {GIT_REPO} {WORKDIR}
    fi
    cd {WORKDIR}
    exec {cmd}
    """

    resp = stream(
        v1.connect_get_namespaced_pod_exec,
        name=POD_NAME,
        namespace=NAMESPACE,
        container="worker",
        command=["/bin/bash", "-c", full_cmd],
        stderr=True,
        stdin=False,
        stdout=True,
        tty=False,
    )

    print(resp)


with DAG(
    dag_id="warm_pod_ml_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
) as dag:

    ensure_pod = PythonOperator(
        task_id="ensure_warm_pod",
        python_callable=ensure_warm_pod,
    )

    wait_pod = PythonOperator(
        task_id="wait_warm_pod",
        python_callable=wait_for_pod_ready,
    )

    preprocess = PythonOperator(
        task_id="preprocessing",
        python_callable=exec_in_warm_pod,
        op_kwargs={
            "cmd": "sleep 30 && echo '[PREPROCESS] start'; python horovod_test/train.py --stage preprocess"
        },
    )

    train = PythonOperator(
        task_id="training",
        python_callable=exec_in_warm_pod,
        op_kwargs={
            "cmd": "sleep 30 && echo '[TRAIN] start'; python horovod_test/train.py --stage train"
        },
    )

    evaluate = PythonOperator(
        task_id="evaluation",
        python_callable=exec_in_warm_pod,
        op_kwargs={
            "cmd": "sleep 30 && echo '[EVAL] start'; python horovod_test/train.py --stage eval"
        },
    )

    ensure_pod >> wait_pod >> preprocess >> train >> evaluate

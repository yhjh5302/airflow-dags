from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

NAMESPACE = "ml"
POD_NAME = "warm-pytorch-worker"
GIT_REPO = "https://github.com/yhjh5302/DNN-Testbed"
WORKDIR = "/root/DNN-Testbed"
TRAIN_SCRIPT = "/root/DNN-Testbed/horovod_test/train.py"


def ensure_warm_pod(**context):
    conf = context["dag_run"].conf or {}

    image = conf.get("image", "nvcr.io/nvidia/pytorch:25.12-py3")
    gpu = conf.get("gpu", 1)
    cpu = conf.get("cpu", "2")
    memory = conf.get("memory", "8Gi")

    pod_yaml = f"""
apiVersion: v1
kind: Pod
metadata:
  name: {POD_NAME}
  namespace: {NAMESPACE}
spec:
  containers:
  - name: worker
    image: {image}
    command: ["/bin/bash", "-c"]
    args: ["while true; do sleep 3600; done"]
    resources:
      limits:
        nvidia.com/gpu: {gpu}
        cpu: {cpu}
        memory: {memory}
"""
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=pod_yaml.encode(),
        check=True,
    )


def exec_cmd(cmd: str) -> str:
    return f"""
    kubectl exec {POD_NAME} -n {NAMESPACE} -- bash -c '
      set -e
      if [ ! -d "{WORKDIR}" ]; then
        git clone {GIT_REPO} /root/DNN-Testbed
      fi
      cd {WORKDIR}
      {cmd}
    '
    """


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
        provide_context=True,
    )

    preprocess = BashOperator(
        task_id="preprocessing",
        bash_command=exec_cmd(
            "echo '[PREPROCESS] start'; python horovod_test/train.py --stage preprocess"
        ),
    )

    train = BashOperator(
        task_id="training",
        bash_command=exec_cmd(
            "echo '[TRAIN] start'; python horovod_test/train.py --stage train"
        ),
    )

    evaluate = BashOperator(
        task_id="evaluation",
        bash_command=exec_cmd(
            "echo '[EVAL] start'; python horovod_test/train.py --stage eval"
        ),
    )

    ensure_pod >> preprocess >> train >> evaluate

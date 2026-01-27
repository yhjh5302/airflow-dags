from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowFailException
from datetime import datetime, timezone
from kubernetes import client, config
from kubernetes.stream import stream
from kubernetes.client.exceptions import ApiException
import sys, time, logging

NAMESPACE = "llm-test"
POD_NAME = "warm-pytorch-worker"
GIT_REPO = "https://github.com/yhjh5302/DNN-Testbed"
GIT_BRANCH = "master"
WORKDIR = "/root/DNN-Testbed"
TRAIN_SCRIPT = "/root/DNN-Testbed/horovod_test/train.py"


def get_pod_events(v1, pod_name, namespace):
    try:
        events = v1.list_namespaced_event(
            namespace, 
            field_selector=f"involvedObject.name={pod_name}"
        )

        def get_event_time(event):
            dt = event.last_timestamp or event.first_timestamp
            return dt if dt else datetime.min.replace(tzinfo=timezone.utc)

        sorted_events = sorted(events.items, key=get_event_time, reverse=True)
        return [f"[{e.reason}] {e.message}" for e in sorted_events]
    except Exception as e:
        return [f"[Error] Cannot get event messages: {e}"]


def ensure_warm_pod(**context):
    log = logging.getLogger("task")
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    conf = context["dag_run"].conf or {}
    image = conf.get("image", "nvcr.io/nvidia/pytorch:25.12-py3")
    cpu = conf.get("cpu", "2")
    memory = conf.get("memory", "8Gi")
    gpu = conf.get("gpu", "1")

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
                            "cpu": str(cpu),
                            "memory": str(memory),
                            "nvidia.com/gpu": str(gpu),
                        }
                    ),
                )
            ],
        ),
    )

    try:
        v1.read_namespaced_pod(name=POD_NAME, namespace=NAMESPACE)
        log.info("Warm pod already exists")
    except ApiException as e:
        if e.status == 404:
            v1.create_namespaced_pod(
                namespace=NAMESPACE,
                body=pod,
            )
            log.info("Warm pod created")
        else:
            raise


def wait_for_pod_ready():
    log = logging.getLogger("task")
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    FATAL_REASONS = [
        "CrashLoopBackOff",
        "ImagePullBackOff",
        "ErrImagePull",
        "CreateContainerConfigError",
        "InvalidImageName",
        "CreateContainerError"
    ]

    while True:
        pod = v1.read_namespaced_pod(POD_NAME, NAMESPACE)
        phase = pod.status.phase

        if phase in ("Succeeded", "Failed", "Unknown"):
            log.error(f"Pod entered a terminal. (Phase: {phase})")
            raise RuntimeError(f"Pod entered a terminal. (Phase: {phase})")

        if pod.metadata.deletion_timestamp is not None:
            log.error(f"Pod is terminating. (Phase: Terminating)")
            raise RuntimeError(f"Pod is terminating. (Phase: Terminating)")

        container_statuses = pod.status.container_statuses or []
        for cs in container_statuses:
            state = cs.state
            
            if cs.ready:
                log.info(f"Pod {POD_NAME} is Ready!")
                return

            if state.waiting:
                reason = state.waiting.reason
                if reason in FATAL_REASONS:
                    log.error(f"Pod failed with fatal reason. (Reason: {reason})")
                    events = get_pod_events(v1, POD_NAME, NAMESPACE)
                    for msg in events[:5]:
                        log.error(f"{msg}")
                    raise RuntimeError(f"Pod failed with fatal reason. (Reason: {reason})")
                else:
                    log.info(f"Container is waiting. (Reason: {reason})")

            if state.terminated and state.terminated.exit_code != 0:
                log.error(f"Container terminated with exit code {state.terminated.exit_code}")
                raise RuntimeError(f"Container terminated with exit code {state.terminated.exit_code}")

        log.info(f"Waiting for Pod {POD_NAME} to be ready. (Current Phase: {phase})")
        time.sleep(5)


def exec_in_warm_pod(cmd: str, **context):
    log = logging.getLogger("task")
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    full_cmd = f"""
    set -e
    if [ -d "{WORKDIR}/.git" ]; then
        cd {WORKDIR}
        git fetch origin {GIT_BRANCH}
        git checkout {GIT_BRANCH}
        git pull origin {GIT_BRANCH}
    else
        git clone -q {GIT_REPO} {WORKDIR}
        cd {WORKDIR}
        git checkout {GIT_BRANCH}
    fi
    {cmd}
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
        _preload_content=False,
    )

    stdout = []
    stderr = []

    while resp.is_open():
        resp.update(timeout=1)
        if resp.peek_stdout():
            out = resp.read_stdout()
            stdout.append(out)
            log.info(out.rstrip())
        if resp.peek_stderr():
            err = resp.read_stderr()
            stderr.append(err)
            log.error(err.rstrip())

    resp.close()

    exit_code = resp.returncode
    if exit_code != 0:
        raise AirflowFailException(f"Pod exec failed with exit code {exit_code}")


with DAG(
    dag_id="warm_pod_ml_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    params={
        "git_repo": Param(default="https://github.com/yhjh5302/DNN-Testbed", type="string", pattern=r"^https?:\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?$"),
        "git_commit": Param(default="ffffffffffffffffffffffffffffffffffffffff", type="string", pattern=r"^[a-f0-9]{40}$"),
        "image": Param(default="nvcr.io/nvidia/pytorch:25.12-py3", type="string", pattern=r"^[\w\.\-/]+:[\w\.\-]+$"),
        "cpu": Param(default="100m", type="string", pattern=r"^[0-9]+m$"),
        "memory": Param(default="256Mi", type="string"),
        "gpu": Param(default="1", type="string", pattern=r"^[0-9]+$"),
        "volumes": Param(
            default=[
                {"volume": "prj-0asd3f92sdw29ds7-", "path": "/data/public"},
                {"volume": "private", "path": "/data/private"},
            ],
            type="array",
            items={
                "type": "object",
                "properties": {
                    "volume": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "minLength": 1},
                },
                "required": ["volume", "path"],
            },
        ),
    },
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
            "cmd": "sleep 3 && echo '[PREPROCESS] start cpu {{ params.cpu }} memory {{ params.memory }}'; python horovod_test/train.py --stage preprocess"
        },
    )

    train = PythonOperator(
        task_id="training",
        python_callable=exec_in_warm_pod,
        op_kwargs={
            "cmd": "sleep 3 && echo '[TRAIN] start'; python horovod_test/train.py --stage train"
        },
    )

    evaluate = PythonOperator(
        task_id="evaluation",
        python_callable=exec_in_warm_pod,
        op_kwargs={
            "cmd": "sleep 3 && echo '[EVAL] start'; python horovod_test/train.py --stage eval"
        },
    )

    ensure_pod >> wait_pod >> preprocess >> train >> evaluate

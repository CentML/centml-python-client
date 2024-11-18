import sys
from functools import wraps
from typing import Callable, Dict
import click
from tabulate import tabulate
from centml.sdk import DeploymentType, DeploymentStatus, ServiceStatus, ApiException, HardwareInstanceResponse
from centml.sdk.api import get_centml_client


depl_name_to_type_map = {
    "inference": DeploymentType.INFERENCE_V2,
    "compute": DeploymentType.COMPUTE_V2,
    "cserve": DeploymentType.CSERVE,
}
depl_type_to_name_map = {v: k for k, v in depl_name_to_type_map.items()}


def handle_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ApiException as e:
            click.echo(f"Error: {e.reason}")
            return None

    return wrapper


def _get_hw_to_id_map(cclient, cluster_id):
    response = cclient.get_hardware_instances(cluster_id)

    # Initialize hashmap for hardware to id or vice versa mapping
    hw_to_id_map: Dict[str, int] = {}
    id_to_hw_map: Dict[int, HardwareInstanceResponse] = {}

    for hw in response:
        hw_to_id_map[hw.name] = hw.id
        id_to_hw_map[hw.id] = hw
    return hw_to_id_map, id_to_hw_map


def _format_ssh_key(ssh_key):
    if not ssh_key:
        return "No SSH Key Found"
    return ssh_key[:10] + '...'


def _get_ready_status(cclient, deployment):
    api_status = deployment.status
    service_status = (
        cclient.get_status(deployment.id).service_status if deployment.status == DeploymentStatus.ACTIVE else None
    )

    status_styles = {
        (DeploymentStatus.PAUSED, None): ("paused", "yellow", "black"),
        (DeploymentStatus.DELETED, None): ("deleted", "white", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.HEALTHY): ("ready", "green", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.INITIALIZING): ("starting", "black", "white"),
        (DeploymentStatus.ACTIVE, ServiceStatus.MISSING): ("starting", "black", "white"),
        (DeploymentStatus.ACTIVE, ServiceStatus.ERROR): ("error", "red", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.CREATECONTAINERCONFIGERROR): (
            "createContainerConfigError",
            "red",
            "black",
        ),
        (DeploymentStatus.ACTIVE, ServiceStatus.CRASHLOOPBACKOFF): ("crashLoopBackOff", "red", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.IMAGEPULLBACKOFF): ("imagePullBackOff", "red", "black"),
        (DeploymentStatus.ACTIVE, ServiceStatus.PROGRESSDEADLINEEXCEEDED): ("progressDeadlineExceeded", "red", "black"),
    }

    style = status_styles.get((api_status, service_status), ("unknown", "black", "white"))
    # Handle foreground and background colors
    return click.style(style[0], fg=style[1], bg=style[2])


@click.command(help="List all deployments")
@click.argument("type", type=click.Choice(list(depl_name_to_type_map.keys())), required=False, default=None)
def ls(type):
    with get_centml_client() as cclient:
        depl_type = depl_name_to_type_map[type] if type in depl_name_to_type_map else None
        deployments = cclient.get(depl_type)
        rows = [
            [d.id, d.name, depl_type_to_name_map[d.type], d.status.value, d.created_at.strftime("%Y-%m-%d %H:%M:%S")]
            for d in deployments
        ]

        click.echo(
            tabulate(
                rows,
                headers=["ID", "Name", "Type", "Status", "Created at"],
                tablefmt="rounded_outline",
                disable_numparse=True,
            )
        )


@click.command(help="Get deployment details")
@click.argument("type", type=click.Choice(list(depl_name_to_type_map.keys())))
@click.argument("id", type=int)
@handle_exception
def get(type, id):
    with get_centml_client() as cclient:
        depl_type = depl_name_to_type_map[type]

        if depl_type == DeploymentType.INFERENCE_V2:
            deployment = cclient.get_inference(id)
        elif depl_type == DeploymentType.COMPUTE_V2:
            deployment = cclient.get_compute(id)
        elif depl_type == DeploymentType.CSERVE:
            deployment = cclient.get_cserve(id)
        else:
            sys.exit("Please enter correct deployment type")

        ready_status = _get_ready_status(cclient, deployment)
        _, id_to_hw_map = _get_hw_to_id_map(cclient, deployment.cluster_id)
        hw = id_to_hw_map[deployment.hardware_instance_id]

        click.echo(
            tabulate(
                [
                    ("Name", deployment.name),
                    ("Status", ready_status),
                    ("Endpoint", deployment.endpoint_url),
                    ("Created at", deployment.created_at.strftime("%Y-%m-%d %H:%M:%S")),
                    ("Hardware", f"{hw.name} ({hw.num_gpu}x {hw.gpu_type})"),
                    ("Cost", f"{hw.cost_per_hr/100} credits/hr"),
                ],
                tablefmt="rounded_outline",
                disable_numparse=True,
            )
        )

        click.echo("Additional deployment configurations:")
        if depl_type == DeploymentType.INFERENCE_V2:
            click.echo(
                tabulate(
                    [
                        ("Image", deployment.image_url),
                        ("Container port", deployment.container_port),
                        ("Healthcheck", deployment.healthcheck or "/"),
                        ("Replicas", {"min": deployment.min_scale, "max": deployment.max_scale}),
                        ("Environment variables", deployment.env_vars or "None"),
                        ("Max concurrency", deployment.concurrency or "None"),
                    ],
                    tablefmt="rounded_outline",
                    disable_numparse=True,
                )
            )
        elif depl_type == DeploymentType.COMPUTE_V2:
            click.echo(
                tabulate(
                    [("Username", "centml"), ("SSH key", _format_ssh_key(deployment.ssh_public_key))],
                    tablefmt="rounded_outline",
                    disable_numparse=True,
                )
            )
        elif depl_type == DeploymentType.CSERVE:
            click.echo(
                tabulate(
                    [
                        ("Hugging face model", deployment.model),
                        (
                            "Parallelism",
                            {"tensor": deployment.tensor_parallel_size, "pipeline": deployment.pipeline_parallel_size},
                        ),
                        ("Replicas", {"min": deployment.min_scale, "max": deployment.max_scale}),
                        ("Max concurrency", deployment.concurrency or "None"),
                    ],
                    tablefmt="rounded_outline",
                    disable_numparse=True,
                )
            )


@click.command(help="Delete a deployment")
@click.argument("id", type=int)
@handle_exception
def delete(id):
    with get_centml_client() as cclient:
        cclient.delete(id)
        click.echo("Deployment has been deleted")


@click.command(help="Pause a deployment")
@click.argument("id", type=int)
@handle_exception
def pause(id):
    with get_centml_client() as cclient:
        cclient.pause(id)
        click.echo("Deployment has been paused")


@click.command(help="Resume a deployment")
@click.argument("id", type=int)
@handle_exception
def resume(id):
    with get_centml_client() as cclient:
        cclient.resume(id)
        click.echo("Deployment has been resumed")


from kubernetes.client.rest import ApiException
from kubernetes import client as k8s_client, config as k8s_config
import kubernetes.client.models as k8s_models

# TODO make logging work like rest of cli
import logging
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def create_service_account(service_account_name: str, namespace: str) -> None:
    """
    Creates a ServiceAccount in a given namespace.

    :param clientset: The Kubernetes API client instance.
    :param service_account_name: Name of the ServiceAccount to create.
    :param namespace: Namespace in which to create the ServiceAccount.
    :return: None if successful, otherwise raises an exception.
    """
    # Define the ServiceAccount object
    service_account = k8s_client.V1ServiceAccount(
        api_version="v1",
        kind="ServiceAccount",
        metadata=k8s_client.V1ObjectMeta(
            name=service_account_name,
            namespace=namespace
        )
    )
    
    try:
        # Create the ServiceAccount in the specified namespace
        api_instance = k8s_client.CoreV1Api()
        api_instance.create_namespaced_service_account(
            namespace=namespace,
            body=service_account
        )
        logging.info(f"ServiceAccount {service_account_name} created in namespace {namespace}")
    except ApiException as e:
        if is_already_exists_error(e):
            logging.info(f"ServiceAccount {service_account_name} already exists in namespace {namespace}")
        else: 
            raise e

def is_already_exists_error(e: Exception) -> bool:
    """
    Determines if the error indicates that a specified resource already exists.
    Supports wrapped errors and returns False when the error is None.
    """    
    # https://github.com/kubernetes/apimachinery/blob/96b97de8d6ba49bc192968551f2120ef3881f42d/pkg/apis/meta/v1/types.go#L895
    return isinstance(e, ApiException) and e.status == 409 and e.reason == "Conflict"

def upsert(kind: str, name: str, create: Callable[[], None], update: Callable[[], None]) -> None:
    """
    Tries to create a resource, and if it already exists, attempts to update it.

    :param kind: The kind of resource (e.g., "ServiceAccount", "ConfigMap").
    :param name: The name of the resource.
    :param create: A callable function that creates the resource.
    :param update: A callable function that updates the resource.
    :return: None if successful, otherwise raises an exception.
    """
    try:
        create()
        logging.info(f"{kind} {name} created")
    except ApiException as e:
        if not is_already_exists_error(e):
            raise RuntimeError(f"Failed to create {kind} {name}: {e}") from e
        try:
            # else upsert it by updating
            update()
            logging.info(f"{kind} {name} updated")
        except ApiException as update_err:
            raise RuntimeError(f"Failed to update {kind} {name}: {update_err}") from update_err


def upsert_cluster_role(name: str, rules: list[k8s_models.V1PolicyRule]) -> None:
    """
    Upserts a ClusterRole.
    """
    cluster_role = k8s_client.V1ClusterRole(
        api_version="rbac.authorization.k8s.io/v1",
        kind="ClusterRole",
        metadata=k8s_client.V1ObjectMeta(name=name),
        rules=rules
    )

    rbac_api = k8s_client.RbacAuthorizationV1Api() 

    def create():
        rbac_api.create_cluster_role(cluster_role)

    def update():
        rbac_api.replace_cluster_role(name, cluster_role)

    upsert("ClusterRole", name, create, update)


def upsert_role(name: str, namespace: str, rules: list[k8s_models.V1PolicyRule]) -> None:
    """
    Upserts a Role in a specific namespace.
    """
    role = k8s_client.V1Role(
        api_version="rbac.authorization.k8s.io/v1",
        kind="Role",
        metadata=k8s_client.V1ObjectMeta(name=name),
        rules=rules
    )
    
    rbac_api = k8s_client.RbacAuthorizationV1Api() 

    def create():               
        rbac_api.create_namespaced_role(namespace, role)

    def update():
        rbac_api.replace_namespaced_role(name, namespace, role)

    upsert("Role", f"{namespace}/{name}", create, update)


def upsert_cluster_role_binding(name: str, cluster_role_name: str, subject: k8s_models.RbacV1Subject) -> None:
    """
    Upserts a ClusterRoleBinding.
    """
    cluster_role_binding = k8s_client.V1ClusterRoleBinding(
        api_version="rbac.authorization.k8s.io/v1",
        kind="ClusterRoleBinding",
        metadata=k8s_client.V1ObjectMeta(name=name),
        role_ref=k8s_client.V1RoleRef(
            api_group="rbac.authorization.k8s.io",
            kind="ClusterRole",
            name=cluster_role_name
        ),
        subjects=[subject]
    )

    rbac_api = k8s_client.RbacAuthorizationV1Api()

    def create():
        rbac_api.create_cluster_role_binding(cluster_role_binding)

    def update():
        rbac_api.replace_cluster_role_binding(name, cluster_role_binding)

    upsert("ClusterRoleBinding", name, create, update)


def upsert_role_binding(name: str, role_name: str, namespace: str, subject: k8s_models.RbacV1Subject) -> None:
    """
    Upserts a RoleBinding in a specific namespace.
    """
    role_binding = k8s_client.V1RoleBinding(
        api_version="rbac.authorization.k8s.io/v1",
        kind="RoleBinding",
        metadata=k8s_client.V1ObjectMeta(name=name),
        role_ref=k8s_client.V1RoleRef(
            api_group="rbac.authorization.k8s.io",
            kind="Role",
            name=role_name
        ),
        subjects=[subject]
    )

    rbac_api = k8s_client.RbacAuthorizationV1Api() 

    def create():
        rbac_api.create_namespaced_role_binding(namespace, role_binding)

    def update():
        rbac_api.replace_namespaced_role_binding(name, namespace, role_binding)

    return upsert("RoleBinding", f"{namespace}/{name}", create, update)


def create_service_account_token_secret(service_account: k8s_models.V1ServiceAccount) -> str:
    """
    Creates a bearer token secret for a given ServiceAccount.

    :param clientset: The Kubernetes API client instance.
    :param service_account: The ServiceAccount object.
    :return: The name of the created secret.
    :raises: RuntimeError if an error occurs during secret creation or ServiceAccount patching.
    """

    assert isinstance(service_account.metadata, k8s_models.V1ObjectMeta)

    # Define the secret
    secret = k8s_client.V1Secret(        
        metadata=k8s_models.V1ObjectMeta(
            generate_name=f"{service_account.metadata.name}-token-",
            namespace=service_account.metadata.namespace,
            annotations={
                "kubernetes.io/service-account.name": service_account.metadata.name
            }
        ),
        type="kubernetes.io/service-account-token"
    )

    try:
        # Create the secret
        api_instance = k8s_client.CoreV1Api()
        created_secret = api_instance.create_namespaced_secret(
            namespace=service_account.metadata.namespace,
            body=secret
        )

        # typing is bad in the kubernetes python client, this is for typing only
        assert isinstance(created_secret, k8s_models.V1Secret)
        assert isinstance(created_secret.metadata, k8s_models.V1ObjectMeta)
        
        logging.info(f"Created bearer token secret for ServiceAccount {service_account.metadata.name}")

        # Update the ServiceAccount to include the secret
        service_account.secrets = [k8s_models.V1ObjectReference(
            name=created_secret.metadata.name,
            namespace=created_secret.metadata.namespace
        )]

        # Create a patch to update the ServiceAccount
        patch = {
            "secrets": [
                {
                    "name": created_secret.metadata.name,
                    "namespace": created_secret.metadata.namespace
                }
            ]
        }

        api_instance.patch_namespaced_service_account(
            name=service_account.metadata.name,
            namespace=service_account.metadata.namespace,
            body=patch
        )
        logging.info(f"Patched ServiceAccount {service_account.metadata.name} with the bearer token secret")

        assert isinstance(created_secret.metadata, k8s_models.V1ObjectMeta)
        assert isinstance(created_secret.metadata.name, str)        
        return created_secret.metadata.name

    except ApiException as e:
        raise RuntimeError(f"Failed to handle token secret for ServiceAccount {service_account.metadata.name}: {e}")


def get_or_create_service_account_token_secret(service_account_name: str, namespace: str) -> str:
    """
    Checks if a ServiceAccount already has a 'kubernetes.io/service-account-token' secret associated with it.
    If not, creates one.

    :param clientset: The Kubernetes API client instance.
    :param sa_name: The name of the ServiceAccount.
    :param namespace: The namespace of the ServiceAccount.
    :return: The name of the service account token secret.
    :raises: RuntimeError if an error occurs.
    """
    core_api = k8s_client.CoreV1Api()
    
    # Polling to wait for the ServiceAccount to have a secret
    start_time = time.time()
    timeout_seconds = 30
    poll_interval = 0.5

    service_account = None
    while time.time() - start_time < timeout_seconds:
        try:
            service_account = core_api.read_namespaced_service_account(name=service_account_name, namespace=namespace)
            if service_account:
                break
        except ApiException as e:
            logging.error(f"Failed to retrieve ServiceAccount {service_account_name}: {e}")
            time.sleep(poll_interval)
    else:
        raise RuntimeError(f"Timed out waiting for ServiceAccount {service_account_name} in namespace {namespace}")

    if not service_account:
        raise RuntimeError(f"ServiceAccount {service_account_name} not found in namespace {namespace}")

    assert isinstance(service_account, k8s_models.V1ServiceAccount)
    service_account_secrets: list[k8s_models.V1ObjectReference] = service_account.secrets or []
    
    # Check if the ServiceAccount already has a token secret
    for secret_ref in service_account_secrets:
        try:
            secret: k8s_models.V1Secret = core_api.read_namespaced_secret(name=secret_ref.name, namespace=namespace) # type: ignore
            if secret.type == "kubernetes.io/service-account-token":
                assert isinstance(secret.metadata, k8s_models.V1ObjectMeta)
                assert isinstance(secret.metadata.name, str)
                return secret.metadata.name
        except ApiException as e:
            logging.warning(f"Failed to retrieve secret {secret_ref.name}: {e}")
    
    # If no token secret is found, create one
    return create_service_account_token_secret(service_account)


def get_service_account_bearer_token(namespace: str, service_account_name: str, timeout: float) -> str:
    """
    Determines if a ServiceAccount has a bearer token secret or creates one if necessary.
    Waits for the secret to contain a token and returns it in base64-encoded form.

    :param clientset: The Kubernetes API client instance.
    :param namespace: The namespace of the ServiceAccount.
    :param service_account_name: The name of the ServiceAccount.
    :param timeout: The maximum time to wait for the bearer token (seconds)
    :return: The bearer token in base64-encoded form.
    :raises: RuntimeError if an error occurs or the token is not available within the timeout.
    """
    try:
        secret_name = get_or_create_service_account_token_secret(service_account_name, namespace)
    except Exception as e:
        raise RuntimeError(f"Failed to get or create token secret for ServiceAccount {service_account_name}: {e}") from e

    start_time = time.time()
    poll_interval = 0.5

    core_api = k8s_client.CoreV1Api()

    while time.time() - start_time < timeout:
        try:
            secret = core_api.read_namespaced_secret(secret_name, namespace)            
            assert isinstance(secret, k8s_models.V1Secret)
            assert isinstance(secret.data, dict)
            if 'token' in secret.data:
                logging.info(f"Bearer token retrieved for ServiceAccount {service_account_name}")
                return secret.data['token']
        except ApiException as e:
            logging.warning(f"Failed to get secret {secret_name}: {e}")
        
        time.sleep(poll_interval)

    raise RuntimeError(f"Timed out waiting for bearer token for ServiceAccount {service_account_name} in namespace {namespace}")


# Constants for resource names
CENTML_PLATFORM_SERVICE_ACCOUNT = "centml-platform"
CENTML_PLATFORM_CLUSTER_ROLE = "centml-platform-role"
CENTML_PLATFORM_CLUSTER_ROLE_BINDING = "centml-platform-role-binding"

# Cluster-wide policy rules for the CentML Platform service account
CENTML_PLATFORM_CLUSTER_POLICY_RULES = [
    k8s_client.V1PolicyRule(
        api_groups=["*"],
        resources=["*"],
        verbs=["*"]
    ),
    k8s_client.V1PolicyRule(
        non_resource_ur_ls=["*"],
        verbs=["*"]
    )
]

# Namespace-specific policy rules for CentML Platform service account
CENTML_PLATFORM_NAMESPACE_POLICY_RULES = [
    k8s_client.V1PolicyRule(
        api_groups=["*"],
        resources=["*"],
        verbs=["*"]
    )
]

def install_cluster_manager_rbac(service_account_namespace: str, rbac_namespaces: list[str], bearer_token_timeout) -> str:
    """
    Installs RBAC resources for a cluster manager to operate a cluster.
    Returns a bearer token for the service account.

    :param clientset: The Kubernetes API client instance.
    :param service_account_namespace: The namespace for the service account and related RBAC resources.
    :param rbac_namespaces: A list of namespaces for namespace-specific roles and bindings.
    :param bearer_token_timeout: Timeout for obtaining the bearer token.
    :return: The service account bearer token.
    """
    # Create the service account
    try:
        create_service_account(CENTML_PLATFORM_SERVICE_ACCOUNT, service_account_namespace)
    except Exception as e:
        raise RuntimeError(f"Failed to create service account: {e}")

    if not rbac_namespaces:
        # Create cluster-wide RBAC resources
        try:
            upsert_cluster_role(CENTML_PLATFORM_CLUSTER_ROLE, CENTML_PLATFORM_CLUSTER_POLICY_RULES)
            upsert_cluster_role_binding(
                CENTML_PLATFORM_CLUSTER_ROLE_BINDING,
                CENTML_PLATFORM_CLUSTER_ROLE,
                k8s_client.RbacV1Subject(
                    kind="ServiceAccount",
                    name=CENTML_PLATFORM_SERVICE_ACCOUNT,
                    namespace=service_account_namespace
                )
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create cluster-wide RBAC resources: {e}")
    else:
        # Create namespace-specific RBAC resources
        for service_account_namespace in rbac_namespaces:
            try:
                upsert_role(CENTML_PLATFORM_CLUSTER_ROLE, service_account_namespace, CENTML_PLATFORM_NAMESPACE_POLICY_RULES)
                upsert_role_binding(
                    CENTML_PLATFORM_CLUSTER_ROLE_BINDING,
                    CENTML_PLATFORM_CLUSTER_ROLE,
                    service_account_namespace,
                    k8s_client.RbacV1Subject(
                        kind="ServiceAccount",
                        name=CENTML_PLATFORM_SERVICE_ACCOUNT,
                        namespace=service_account_namespace
                    )
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create namespace-specific RBAC resources in {service_account_namespace}: {e}")

    # Get the bearer token for the service account
    try:
        return get_service_account_bearer_token(service_account_namespace, CENTML_PLATFORM_SERVICE_ACCOUNT, bearer_token_timeout)
    except Exception as e:
        raise RuntimeError(f"Failed to get service account bearer token: {e}")



def uninstall_cluster_manager_rbac() -> None:
    """
    Removes RBAC resources for a cluster manager to operate a cluster.
    """
    uninstall_rbac(
        "kube-system",
        CENTML_PLATFORM_CLUSTER_ROLE_BINDING,
        CENTML_PLATFORM_CLUSTER_ROLE,
        CENTML_PLATFORM_SERVICE_ACCOUNT,
    )

def uninstall_rbac(namespace: str, binding_name: str, role_name: str, service_account_name: str) -> None:
    """
    Uninstalls RBAC related resources: ClusterRoleBinding, ClusterRole, and ServiceAccount.

    :param clientset: The Kubernetes API client instance.
    :param namespace: The namespace for the ServiceAccount.
    :param binding_name: The name of the ClusterRoleBinding to delete.
    :param role_name: The name of the ClusterRole to delete.
    :param service_account_name: The name of the ServiceAccount to delete.
    :return: None if successful, otherwise raises an exception.
    """
    try:
        k8s_client.RbacAuthorizationV1Api().delete_cluster_role_binding(binding_name)
        logging.info(f"ClusterRoleBinding {binding_name} deleted")
    except ApiException as e:
        if e.status != 404:  # Not Found
            raise RuntimeError(f"Failed to delete ClusterRoleBinding {binding_name}: {e}") from e
        logging.info(f"ClusterRoleBinding {binding_name} not found")

    try:
        k8s_client.RbacAuthorizationV1Api().delete_cluster_role(role_name)
        logging.info(f"ClusterRole {role_name} deleted")
    except ApiException as e:
        if e.status != 404:
            raise RuntimeError(f"Failed to delete ClusterRole {role_name}: {e}") from e
        logging.info(f"ClusterRole {role_name} not found")

    try:
        k8s_client.CoreV1Api().delete_namespaced_service_account(service_account_name, namespace)
        logging.info(f"ServiceAccount {service_account_name} in namespace {namespace} deleted")
    except ApiException as e:
        if e.status != 404:
            raise RuntimeError(f"Failed to delete ServiceAccount {service_account_name} in namespace {namespace}: {e}") from e
        logging.info(f"ServiceAccount {service_account_name} in namespace {namespace} not found")



@click.command(name="add", help="Add cluster to CentML Platform")
@click.argument("name", type=str)
@handle_exception
def add_cluster(name: str):
    k8s_config.load_kube_config() 
    service_account_token = install_cluster_manager_rbac("kube-system", [], 30)   


@click.command(name="remove", help="Unimport cluster")
@click.argument("name", type=str)
@handle_exception
def remove_cluster(name: str):
    k8s_config.load_kube_config()     
    uninstall_cluster_manager_rbac()   

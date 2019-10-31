# masters2

[![Binder](https://binder.cern.ch/badge_logo.svg)](https://binder.cern.ch/v2/gh/tghartland/masters2-binder/py3?urlpath=%2Flab)

### With binder

Set up a cluster as in the steps below, and launch the jupyterlab instance with the binder link above.

Upload a kubeconfig file (either cluster admin or for a specific RBAC account) and move it to ~/.kube/config in jupyterlab.

Start a redis db using [this manifest](./redis/redis.yaml) and the [associated service](./redis/redis-svc.yaml).
The service IP needs to be copied to two places:

* in masters-workflow.yaml, the parameter `redis-host`
* in masters/db.py, the variable `HOST`

Launch the notebook masters-notebook.ipynb and start going through the steps.

Leave a few seconds after the `plot.setup_figure()` step for the figure to initialize or it will
not resize properly.

After submitting the workflow to argo, copy the workflow name from the first line of output and paste it
into the workflow variable on the first line of the next step.

The last step will loop and continuously update the figure with the data pulled from redis.

### Argo installation

```bash
$ kubectl create namespace argo
$ kubectl apply -n argo -f https://raw.githubusercontent.com/argoproj/argo/stable/manifests/install.yaml

$ kubectl apply -f workflow-rolebinding.yaml
```

The workflow rolebinding is needed by the argo sidecar container to be able to monitor the pod it is running in.

The same rolebinding will need to be created for any other namespaces that will be used for running argo workflows.

### Setup S3 storage

```bash
$ S3_HOST=$(openstack catalog show s3 -f value -c endpoints | grep public | cut -d '/' -f3)
$ ACCESS_KEY=$(openstack ec2 credentials create -f value -c access)
$ SECRET_KEY=$(openstack ec2 credentials show $ACCESS_KEY -f value -c secret)

$ kubectl create secret generic cern-s3-cred --from-literal=accessKey=$ACCESS_KEY --from-literal=secretKey=$SECRET_KEY

$ echo $S3_HOST
s3.cern.ch
```

`S3_HOST` value goes into the workflow yaml as a parameter.

A bucket for artifact storage should be created in the S3 storage and its name
should also be given as a parameter in the workflow.

### Start workflow

```bash
$ argo submit --watch masters-workflow.yaml
```

### Workflow overview

In this example the workflow is finding expected limits at two mass points,
and running two pods in parallel at each.

```bash
$ kubectl -n argo port-forward service/argo-ui 8001:80
```

![](img/workflow-shape.png)

### Result

When the simulated peaks from all data points are used, and with a higher parallelism,
the final plot is produced and stored in the s3 bucket in directory `{{workflow.name}}/results`,
taking only a few minutes to complete in a large cluster.

![](img/brazil-masters-workflow-l2k4r.png)

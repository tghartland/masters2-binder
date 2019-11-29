REGISTRY?=gitlab-registry.cern.ch/thartlan/containers

PYROOT_VERSION=v1
PYROOT_IMAGE=${REGISTRY}/pyroot:${PYROOT_VERSION}

MASTERS_VERSION=v1
MASTERS_IMAGE=${REGISTRY}/masters2:${MASTERS_VERSION}

INIT_REDIS_VERSION=v1
INIT_REDIS_IMAGE=${REGISTRY}/masters-init-redis:${INIT_REDIS_VERSION}

JUPYTERLAB_VERSION=v3
JUPYTERLAB_IMAGE=${REGISTRY}/jupyterlab:${JUPYTERLAB_VERSION}

build-pyroot:
	docker build -t ${PYROOT_IMAGE} -f Dockerfile.pyroot .

#compress-pyroot: build-pyroot
#	docker export $$(docker run -d ${PYROOT_IMAGE}) | docker import - ${PYROOT_IMAGE}

push-pyroot: build-pyroot
	docker push ${PYROOT_IMAGE}

build-masters2: build-pyroot
	docker build -t ${MASTERS_IMAGE} --build-arg PYROOT_IMAGE=${PYROOT_IMAGE} -f src/Dockerfile .

push-masters2: push-pyroot build-masters2
	docker push ${MASTERS_IMAGE}

build-init-redis:
	docker build -t ${INIT_REDIS_IMAGE} -f redis/Dockerfile.init_redis redis

push-init-redis: build-init-redis
	docker push ${INIT_REDIS_IMAGE}

build-jupyterlab:
	docker build -t ${JUPYTERLAB_IMAGE} -f Dockerfile.jupyterlab .

compress-jupyterlab: build-jupyterlab
	docker tag ${JUPYTERLAB_IMAGE} ${JUPYTERLAB_IMAGE}-uncompressed
	docker export $$(docker run -d ${JUPYTERLAB_IMAGE}-uncompressed) | docker import - ${JUPYTERLAB_IMAGE}
	docker image list ${JUPYTERLAB_IMAGE}-uncompressed --format "{{.Repository}}:{{.Tag}} {{.Size}}"
	docker image list ${JUPYTERLAB_IMAGE} --format "{{.Repository}}:{{.Tag}} {{.Size}}"

push-jupyterlab: compress-jupyterlab
	docker push ${JUPYTERLAB_IMAGE}

.PHONY: build-pyroot push-pyroot build-masters2 push-masters2 build-init-redis push-init-redis build-jupyterlab compress-jupyterlab push-jupyterlab

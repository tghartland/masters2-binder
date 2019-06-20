REPOSITORY?=thartland

.PHONY: build
build:
	docker build -t ${REGISTRY}/centos-root .

.PHONY: push
push: build
	docker push ${REGISTRY}/centos-root

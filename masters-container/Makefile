REGISTRY?=thartland

.PHONY: build
build:
	docker build -t ${REGISTRY}/masters .

.PHONY: push
push: build
	docker push ${REGISTRY}/masters

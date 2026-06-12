IMAGE_NAME := cnapcloud/nyc-mlops

.PHONY: docker-build docker-run docker-run-interactive

build:
	docker build -t $(IMAGE_NAME):latest .

run:
	docker run --rm \
		--add-host mlflow.cnapcloud.com:192.168.0.180 \
		$(IMAGE_NAME):latest

push: build
	docker push $(IMAGE_NAME):latest

run-interactive:
	docker run --rm -it \
		--add-host mlflow.cnapcloud.com:192.168.0.180 \
		$(IMAGE_NAME):latest /bin/bash
